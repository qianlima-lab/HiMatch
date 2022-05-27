#!/usr/bin/env python
# coding: utf-8

import torch
from helper.utils import get_hierarchy_relations
import torch.nn.functional as F
import numpy as np

class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.5, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.elipson = 0.000001

    def forward(self, logits, labels):
        pt = torch.sigmoid(logits)
        alpha = self.alpha
        loss = - alpha * (1 - pt) ** self.gamma * labels * torch.log(pt) - \
               (1 - alpha) * pt ** self.gamma * (1 - labels) * torch.log(1 - pt)
        return torch.mean(loss)
        
class ClassificationLoss(torch.nn.Module):
    def __init__(self,
                 taxonomic_hierarchy,
                 label_map,
                 recursive_penalty,
                 recursive_constraint=True, loss_type="bce"):
        """
        Criterion class, classfication loss & recursive regularization
        :param taxonomic_hierarchy:  Str, file path of hierarchy taxonomy
        :param label_map: Dict, label to id
        :param recursive_penalty: Float, lambda value <- config.train.loss.recursive_regularization.penalty
        :param recursive_constraint: Boolean <- config.train.loss.recursive_regularization.flag
        """
        super(ClassificationLoss, self).__init__()
        self.loss_type = loss_type
        self.focal_loss_fn = FocalLoss()
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.recursive_relation = get_hierarchy_relations(taxonomic_hierarchy,
                                                          label_map)
        self.recursive_penalty = recursive_penalty
        self.recursive_constraint = recursive_constraint

    def _recursive_regularization(self, params, device):
        """
        recursive regularization: constraint on the parameters of classifier among parent and children
        :param params: the parameters on each label -> torch.FloatTensor(N, hidden_dim)
        :param device: torch.device -> config.train.device_setting.device
        :return: loss -> torch.FloatTensor, ()
        """
        rec_reg = 0.0
        for i in range(len(params)):
            if i not in self.recursive_relation.keys():
                continue
            child_list = self.recursive_relation[i]
            if not child_list:
                continue
            child_list = torch.tensor(child_list).to(device)
            child_params = torch.index_select(params, 0, child_list)
            parent_params = torch.index_select(params, 0, torch.tensor(i).to(device))
            parent_params = parent_params.repeat(child_params.shape[0], 1)
            _diff = parent_params - child_params
            diff = _diff.view(_diff.shape[0], -1)
            rec_reg += 1.0 / 2 * torch.norm(diff, p=2) ** 2
        return rec_reg

    def forward(self, logits, targets, recursive_params):
        """
        :param logits: torch.FloatTensor, (batch, N)
        :param targets: torch.FloatTensor, (batch, N)
        :param recursive_params: the parameters on each label -> torch.FloatTensor(N, hidden_dim)
        """
        device = logits.device
        if self.loss_type == "bce":
            loss = self.loss_fn(logits, targets)
        elif self.loss_type == "focal":
            loss = self.focal_loss_fn(logits, targets)
        else:
            loss = 0
        if self.recursive_constraint:
            loss += self.loss_fn(logits, targets) + \
                   self.recursive_penalty * self._recursive_regularization(recursive_params,
                                                                           device)
        return loss

class MarginRankingLoss(torch.nn.Module):
    def __init__(self, config):
        """
        Criterion loss
        default torch.nn.MarginRankingLoss(0.01)
        """
        super(MarginRankingLoss, self).__init__()
        self.dataset = config.data.dataset
        base = 0.2
        self.ranking = [torch.nn.MarginRankingLoss(margin=base*0.1), torch.nn.MarginRankingLoss(margin=base * 0.5),
                        torch.nn.MarginRankingLoss(margin=base)]
        self.negative_ratio = config.data.negative_ratio


    def forward(self, text_repre, label_repre_positive, label_repre_negative, mask=None):
        """
        :param text_repre: torch.FloatTensor, (batch, hidden)
        :param label_repre_positive: torch.FloatTensor, (batch, hidden)
        :param label_repre_negative: torch.FloatTensor, (batch, sample_num, hidden)
        :param mask: torch.BoolTensor, (batch, negative_ratio, negative_number), the index of different label
        """
        loss_inter_total, loss_intra_total = 0, 0

        text_score = text_repre.unsqueeze(1).repeat(1, label_repre_positive.size(1), 1)
        loss_inter = (torch.pow(text_score - label_repre_positive, 2)).sum(-1)
        loss_inter = F.relu(loss_inter / text_repre.size(-1))
        loss_inter_total += loss_inter.mean()

        for i in range(self.negative_ratio):
            m = mask[:, i]
            m = m.unsqueeze(-1).repeat(1, 1, label_repre_negative.size(-1))
            label_n_score = torch.masked_select(label_repre_negative, m)
            label_n_score = label_n_score.view(text_repre.size(0), -1, label_repre_negative.size(-1))
            text_score = text_repre.unsqueeze(1).repeat(1, label_n_score.size(1), 1)

            # index 0: parent node
            if i == 0:
                loss_inter_parent = (torch.pow(text_score - label_n_score, 2)).sum(-1)
                loss_inter_parent = F.relu((loss_inter_parent-0.01) / text_repre.size(-1))
                loss_inter_total += loss_inter_parent.mean()
            else:
                # index 1: wrong sibling, index 2: other wrong label
                loss_intra = (torch.pow(text_score - label_n_score, 2)).sum(-1)
                loss_intra = F.relu(loss_intra / text_repre.size(-1))
                loss_gold = loss_inter.view(1, -1)
                loss_cand = loss_intra.view(1, -1)
                ones = torch.ones(loss_gold.size()).to(loss_gold.device)
                loss_intra_total += self.ranking[i](loss_cand, loss_gold, ones)
        return loss_inter_total, loss_intra_total