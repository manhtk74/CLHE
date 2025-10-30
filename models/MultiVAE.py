#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp 
from collections import OrderedDict

def init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias,0)
    elif isinstance(m, nn.Parameter):
        nn.init.xavier_uniform_(m)

class MultiVAE(nn.Module):
    def __init__(self, conf, raw_graph, features):
        super(MultiVAE, self).__init__()
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
        self.anneal = conf["anneal"] if "anneal" in conf else 0
        self.sigmoid = conf["sigmoid"]

        if "id" in self.component:
            self.item_embeddings = nn.Parameter(torch.FloatTensor(self.num_item, self.embedding_size*2))
            init(self.item_embeddings)
            self.item_embeddings_retrival = nn.Parameter(torch.FloatTensor(self.num_item, self.embedding_size))
            init(self.item_embeddings_retrival)
        elif "mm" in self.component:
            # import pdb;pdb.set_trace()
            self.content_feature = F.normalize(self.content_feature,dim=-1)
            self.text_feature = F.normalize(self.text_feature,dim=-1)

            def dense(feature, ouput_dim=64):
                module = nn.Sequential(OrderedDict([
                    ('w1', nn.Linear(feature.shape[1], feature.shape[1])),
                    ('act1', nn.ReLU()),
                    ('w2', nn.Linear(feature.shape[1], 256)),
                    ('act2', nn.ReLU()),
                    ('w3', nn.Linear(256, ouput_dim)),
                    ]))
                
                for m in module:
                    init(m)
                
                return module
            
            # encoders for media feature
            self.c_encoder = dense(self.content_feature, ouput_dim=self.embedding_size*2)
            self.t_encoder = dense(self.text_feature, ouput_dim=self.embedding_size*2)
            self.c_encoder_retrival = dense(self.content_feature)
            self.t_encoder_retrival = dense(self.text_feature)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
        
    def encode(self, input):
        
        if "id" in self.component:
            item_feature = self.item_embeddings
        elif "mm" in self.component:
            item_feature_c = self.c_encoder(self.content_feature)
            item_feature_t = self.t_encoder(self.text_feature)
            item_feature = F.normalize(item_feature_c) + F.normalize(item_feature_t)
            
        output = (input @ item_feature) / torch.sum(input, dim=-1).view(-1,1)
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
        idx, x = batch[:2]

        h = self.encode(x)
        
        mu = h[:, :self.embedding_size]
        logvar = h[:, self.embedding_size:]
        z = self.reparameterize(mu, logvar)

        recon_x = self.decode(z)

        negLogLike = torch.sum(F.log_softmax(recon_x, 1) * x, -1) / x.sum(dim=-1)
        negLogLike = -torch.mean(negLogLike) 
        KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        return {
            'loss': negLogLike + self.anneal * KLD, 
            'KLD': KLD.detach()
        }

    def evaluate(self, propagate_result, batch):
        idx, x = batch[:2]
        h = self.encode(x)
        mu = h[:, :self.embedding_size]
        logvar = h[:, self.embedding_size:]
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x

