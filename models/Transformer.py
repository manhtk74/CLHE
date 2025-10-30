#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp 
from collections import OrderedDict
from models.utils import TransformerEncoder


def init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias,0)
    elif isinstance(m, nn.Parameter):
        nn.init.xavier_uniform_(m)

class Transformer(nn.Module):
    def __init__(self, conf, raw_graph, features):
        super(Transformer, self).__init__()
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
        self.transformer = TransformerEncoder(conf={
                "n_layer": conf["trans_layer"],
                "dim": 64,
                "num_token": 100,
                "device": self.device,
            }, data={"sp_graph": self.bi_graph_seen})
        

    def propagate(self, test=False):
        return None
    
    def encode(self, idx, x):
        if "id" in self.component:
            item_feature = self.item_embeddings
        elif "mm" in self.component:
            item_feature_c = self.c_encoder(self.content_feature)
            item_feature_t = self.t_encoder(self.text_feature)
            item_feature = F.normalize(item_feature_c) + F.normalize(item_feature_t)

        z = self.transformer(idx, x, item_feature)
        return z
    
    def decode(self, z):
        if "id" in self.component:
            item_feature = self.item_embeddings_retrival
        elif "mm" in self.component:
            item_feature_c = self.c_encoder_retrival(self.content_feature)
            item_feature_t = self.t_encoder_retrival(self.text_feature)
            item_feature = F.normalize(item_feature_c) + F.normalize(item_feature_t)

        return z @ item_feature.transpose(0,1)
    def forward(self, batch):
        idx, x = batch[:2]

        z = self.encode(idx, x)
        recon_x = self.decode(z)
        loss = recon_loss_function(recon_x, x)
        return {
            'loss': loss,
        }

    def evaluate(self, propagate_result, batch):
        idx, x = batch[:2]
        z = self.encode(idx, x)
        recon_x = self.decode(z)
        return recon_x

def recon_loss_function(recon_x, x):
    negLogLike = torch.sum(F.log_softmax(recon_x, 1) * x, -1) / x.sum(dim=-1)
    negLogLike = -torch.mean(negLogLike) 
    return negLogLike
