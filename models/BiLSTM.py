#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp 
from collections import OrderedDict
from torch.nn.utils.rnn import pad_sequence

def init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias,0)
    elif isinstance(m, nn.Parameter):
        nn.init.xavier_uniform_(m)

class BiLSTM(nn.Module):
    def __init__(self, conf, raw_graph, features):
        super(BiLSTM, self).__init__()
        self.conf = conf
        device = self.conf["device"]
        self.device = device

        self.ui_graph, self.bi_graph_train, self.bi_graph_seen = raw_graph
        self.content_feature, self.text_feature, _ = features

        self.num_user = self.conf["num_users"]
        self.num_bundle = self.conf["num_bundles"]
        self.num_item = self.conf["num_items"]
        self.embedding_size = 64
        self.component = ["mm"] 
        self.sigmoid = conf["sigmoid"]

        if "id" in self.component:
            self.item_embeddings = nn.Parameter(torch.FloatTensor(self.num_item, self.embedding_size))
            init(self.item_embeddings)
            self.item_embeddings_retrival = nn.Parameter(torch.FloatTensor(self.num_item, self.embedding_size))
            init(self.item_embeddings_retrival)
        elif "mm" in self.component:
            self.content_feature = nn.functional.normalize(self.content_feature,dim=-1)
            self.text_feature = nn.functional.normalize(self.text_feature,dim=-1)

            def dense(feature):
                module = nn.Sequential(OrderedDict([
                    ('w1', nn.Linear(feature.shape[1], feature.shape[1])),
                    ('act1', nn.ReLU()),
                    ('w2', nn.Linear(feature.shape[1], 256)),
                    ('act2', nn.ReLU()),
                    ('w3', nn.Linear(256, 64)),
                    ]))
                
                for m in module:
                    init(m)
                
                return module
            
            # encoders for media feature
            self.c_encoder = dense(self.content_feature)
            self.t_encoder = dense(self.text_feature)

            self.c_encoder_retrival = dense(self.content_feature)
            self.t_encoder_retrival = dense(self.text_feature)

        # init gru
        # generate_sequence

        self.seqs, self.lens = self.__generate_sequences(self.bi_graph_seen)
        self.seq_model = nn.LSTM(self.embedding_size, self.embedding_size, 2, bidirectional=True) # inputsize, hiddensize, layer

        self.fc = nn.Linear(self.embedding_size * 2, self.embedding_size) 

    def __generate_sequences(self, graph):
        # tensor
        
        # length = graph.sum(axis=0)
        # print(length.shape)
        graph = graph.tocoo()
        values = graph.data
        indices = np.vstack((graph.row, graph.col))
        graph = torch.sparse.FloatTensor(torch.LongTensor(indices), torch.LongTensor(values), torch.Size(graph.shape))

        sequence = [ i.coalesce().indices().squeeze(0) for i in graph ]
        lens = torch.LongTensor([len(seq)-1 for seq in sequence]).to(self.device)

        sequence = pad_sequence(
            sequence, batch_first=True,padding_value=0).to(self.device)

        return sequence, lens

    def encode(self, input):
        input = input.long()

        if "id" in self.component:
            item_feature = self.item_embeddings
        elif "mm" in self.component:
            item_feature_c = self.c_encoder(self.content_feature)
            item_feature_t = self.t_encoder(self.text_feature)
            item_feature = F.normalize(item_feature_c) + F.normalize(item_feature_t)

        h = item_feature[self.seqs[input]] # [bs, N, d]  
        ends = self.lens[input]
        h = h.transpose(0,1) # [N, bs, d]

        output, hn = self.seq_model(h) # [N, bs, d]
        output = output[ends, torch.arange(ends.shape[0])]
        # print(output.shape)
        output = self.fc(output)
        return output
    
    def decode(self, z):
        if "id" in self.component:
            item_feature = self.item_embeddings_retrival
        elif "mm" in self.component:
            item_feature_c = self.c_encoder_retrival(self.content_feature)
            item_feature_t = self.t_encoder_retrival(self.text_feature)
            item_feature = F.normalize(item_feature_c) + F.normalize(item_feature_t)

        logits = z @ item_feature.transpose(0,1)
        if self.sigmoid:
            logits = torch.sigmoid(logits)
        return logits
    
    def propagate(self, test=False):
        return None
    
    def forward(self, batch):
        idx, x  = batch[:2]
        z = self.encode(idx)
        recon_x = self.decode(z)
        loss = recon_loss_function(recon_x, x)
        return {
            'loss': loss,
        }

    def evaluate(self, propagate_result, batch):
        idx, x = batch[:2]
        z = self.encode(idx)
        recon_x = self.decode(z)

        return recon_x

def recon_loss_function(recon_x, x):
    negLogLike = torch.sum(F.log_softmax(recon_x, 1) * x, -1) / x.sum(dim=-1)
    negLogLike = -torch.mean(negLogLike) 
    return negLogLike
