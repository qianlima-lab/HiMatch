import torch.nn as nn
import math
import torch
import numpy as np
import os
from models.structure_model.structure_encoder import StructureEncoder

class MatchingNet(nn.Module):
    def __init__(self, config, graph_model_label=None, label_map=None):
        super(MatchingNet, self).__init__()
        self.config = config
        self.dataset = config.data.dataset
        self.label_map = label_map
        self.positive_sample_num = config.data.positive_num
        self.negative_sample_num = config.data.negative_ratio * self.positive_sample_num
        self.dimension = config.matching_net.dimension
        self.dropout_ratio = config.matching_net.dropout

        self.embedding_net1 = nn.Sequential(nn.Linear(len(self.label_map) * config.model.linear_transformation.node_dimension, self.dimension),
                                                nn.ReLU(),
                                                nn.Dropout(self.dropout_ratio),
                                                nn.Linear(self.dimension, self.dimension))
        self.label_encoder = nn.Sequential(nn.Linear(config.embedding.label.dimension, self.dimension),
                                               nn.ReLU(),
                                               nn.Dropout(self.dropout_ratio),
                                               nn.Linear(self.dimension, self.dimension))
        self.graph_model = graph_model_label
        self.label_map = label_map

        if self.config.matching_net.output_relu != 0:
            self.output_relu = nn.ReLU()
        if self.config.matching_net.output_dropout != 0:
            self.output_dropout = nn.Dropout(self.config.matching_net.output_dropout)
        self.l2_norm = self.config.matching_net.l2_norm

    def forward(self, text, gather_positive, gather_negative, label_repre):
        """
        forward pass of matching learning
        :param gather_positive ->  torch.BoolTensor, (batch_size, positive_sample_num), index of positive label
        :param gather_negative ->  torch.BoolTensor, (batch_size, negative_sample_num), index of negative label
        :param label_repre ->  torch.FloatTensor, (batch_size, label_size, label_feature_dim)
        """
        #label_repre = torch.nn.functional.normalize(label_repre)
        #text = torch.nn.functional.normalize(text)
        
        gather_positive = gather_positive.to(text.device)
        gather_negative = gather_negative.to(text.device)

        label_repre = label_repre.unsqueeze(0)
        label_repre = self.graph_model(label_repre)
        label_repre = label_repre.repeat(text.size(0), 1, 1)

        label_repre = self.label_encoder(label_repre)

        if self.config.matching_net.output_relu != 0:
            label_repre = self.output_relu(label_repre)
        if self.config.matching_net.output_dropout != 0:
            label_repre = self.output_dropout(label_repre)
        if self.l2_norm:
            label_repre = torch.nn.functional.normalize(label_repre, dim=-1)
        
        label_positive = torch.gather(label_repre, 1, gather_positive.view(text.size(0), self.positive_sample_num, 1).expand(text.size(0), self.positive_sample_num, label_repre.size(-1)))

        label_negative = torch.gather(label_repre, 1, gather_negative.view(text.size(0), self.negative_sample_num, 1).expand(text.size(0), self.negative_sample_num, label_repre.size(-1)))
        text_encoder = self.embedding_net1(text)
        
        if self.config.matching_net.output_relu != 0:
            text_encoder = self.output_relu(text_encoder)
        if self.config.matching_net.output_dropout != 0:
            text_encoder = self.output_dropout(text_encoder)
        if self.l2_norm:
            text_encoder = torch.nn.functional.normalize(text_encoder, dim=-1)
        
        return text_encoder, label_positive, label_negative

