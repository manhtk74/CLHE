#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp 
from collections import OrderedDict
from tqdm import tqdm

def init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias,0)
    elif isinstance(m, nn.Parameter):
        nn.init.xavier_uniform_(m)
        
eps = 1e-9
def laplace_transform(graph):
    rowsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=1).A.ravel()) + eps))
    colsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=0).A.ravel()) + eps))
    graph = rowsum_sqrt @ graph @ colsum_sqrt
    
    graph = graph.tocoo()
    values = graph.data
    indices = np.vstack((graph.row, graph.col))
    graph = torch.sparse.FloatTensor(torch.LongTensor(
        indices), torch.FloatTensor(values), torch.Size(graph.shape))
    return graph

class HyperGraph(nn.Module):
    def __init__(self, conf, raw_graph, features):
        super(HyperGraph, self).__init__()
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
        
        self.num_layer = 1
        
        self.graph = self.create_graph(self.bi_graph_train).to(device)
        
    def create_graph(self, graph):
        graph = graph.T @ graph # [#items. #items]

        rowsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=1).A.ravel()) + 1e-8))
        colsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=0).A.ravel()) + 1e-8))
        graph = rowsum_sqrt @ graph @ colsum_sqrt

        graph = graph.tocoo()
        values = graph.data
        indices = np.vstack((graph.row, graph.col))
        graph = torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(graph.shape))
        
        return graph
    
    def encode(self, input, idx):
        h = F.normalize(input, dim=-1)        
        # graphs = self.graphs[idx] # [bs, #I, #I]
        if "id" in self.component:
            item_feature = self.item_embeddings
        elif "mm" in self.component:
            item_feature_c = self.c_encoder(self.content_feature)
            item_feature_t = self.t_encoder(self.text_feature)
            item_feature = F.normalize(item_feature_c) + F.normalize(item_feature_t)

            features = item_feature
            for i in range(self.num_layer):
                features = torch.spmm(self.graph, features)
        
        return F.normalize(h,dim=-1) @ features
    
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
        idx, x = batch[:2]
        z = self.encode(x, idx)
        recon_x = self.decode(z)
        loss = recon_loss_function(recon_x, x)
        return {
            'loss': loss,
        }
    
    def evaluate(self, propagate_result, batch):
        idx, x = batch[:2]
        z = self.encode(x,idx)
        recon_x = self.decode(z)
        return recon_x

def recon_loss_function(recon_x, x):
    negLogLike = torch.sum(F.log_softmax(recon_x, 1) * x, -1) / x.sum(dim=-1)
    negLogLike = -torch.mean(negLogLike) 
    return negLogLike
