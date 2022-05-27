#!/usr/bin/env python
# coding:utf-8

import torch
import numpy as np
import os
from torch import nn
from models.matching_network import MatchingNet
import torch.nn.functional as F

class HiMatchTP(nn.Module):
    def __init__(self, config, label_map, graph_model, device, model_mode, graph_model_label=None):
        """
        :param config: helper.configure, Configure Object
        :param label_map: helper.vocab.Vocab.v2i['label'] -> Dict{str:int}
        :param graph_model: computational graph for graph model
        :param device: torch.device, config.train.device_setting.device
        :param graph_model_label: computational graph for label graph
        """
        super(HiMatchTP, self).__init__()

        self.config = config
        self.device = device
        self.label_map = label_map

        self.graph_model = graph_model
        self.graph_model_label = graph_model_label
        self.dataset = config.data.dataset
        
        self.linear = nn.Linear(len(self.label_map) * config.model.linear_transformation.node_dimension,
                                    len(self.label_map))

        
        # linear transform
        self.transformation = nn.Linear(config.model.linear_transformation.text_dimension,
                                            len(self.label_map) * config.model.linear_transformation.node_dimension)
        
        # dropout
        self.transformation_dropout = nn.Dropout(p=config.model.linear_transformation.dropout)
        self.dropout = nn.Dropout(p=config.model.classifier.dropout)

        self.matching_model = MatchingNet(config, self.graph_model_label, label_map)

    def forward(self, inputs):
        """
        forward pass of text feature propagation and matching learning
        :param text_feature ->  torch.FloatTensor, (batch_size, kernel_size, text_dim)
        :param ranking_positive_mask ->  torch.LongTensor, (batch_size, positive_num)
        :param ranking_negative_mask ->  torch.LongTensor, (batch_size, negative_num)
        :param label_repre ->  torch.FloatTensor, (batch_size, label_size, label_feature_dim)
        :return: logits ->  torch.FloatTensor, (batch, N)
        :return: text_repre ->  torch.FloatTensor, (batch, text_feature_dim)
        :return: label_repre ->  torch.FloatTensor, (batch, label_size, label_feature_dim)
        """
        if inputs[1] == "TRAIN":
            text_feature, mode, ranking_positive_mask, ranking_negative_mask, label_repre = inputs
        else:
            text_feature, mode = inputs[0], inputs[1]
            
        text_feature = text_feature.view(text_feature.shape[0], -1)
        #original_text_feature = text_feature
        
        text_feature = self.transformation_dropout(self.transformation(text_feature))
        text_feature = text_feature.view(text_feature.shape[0],
                                                 len(self.label_map),
                                                 self.config.model.linear_transformation.node_dimension)

        label_wise_text_feature = self.graph_model(text_feature)
        
        logits = self.linear(label_wise_text_feature.view(label_wise_text_feature.shape[0], -1))
        if self.config.model.classifier.output_drop:
            logits = self.dropout(logits)
        
        if inputs[1] == "TRAIN":
            text_repre, label_repre_positive, label_repre_negative = self.matching_model(label_wise_text_feature.view(label_wise_text_feature.shape[0], -1),
                                                                              ranking_positive_mask,
                                                                              ranking_negative_mask, label_repre)
            return logits, text_repre, label_repre_positive, label_repre_negative
        else:
            return logits

