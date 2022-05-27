#!/usr/bin/env python
# coding: utf-8

import torch
import random
import numpy as np

class Collator(object):
    def __init__(self, config, vocab, mode="TRAIN"):
        """
        Collator object for the collator_fn in data_modules.data_loader
        :param config: helper.configure, Configure Object
        :param vocab: data_modules.vocab, Vocab Object
        """
        super(Collator, self).__init__()
        self.device = config.train.device_setting.device
        self.label_size = len(vocab.v2i['label'].keys())
        self.mode = mode
        self.positive_sample_num = config.data.positive_num
        self.negative_sample_num = config.data.negative_ratio * self.positive_sample_num
        self.negative_ratio = config.data.negative_ratio
        self.vocab = vocab

        self.version_higher_11 = True
        version = torch.__version__
        version_num = version.split('.')
        if (int(version_num[0]) == 1 and int(version_num[1]) <= 1) or int(version_num[0]) == 0:
            self.version_higher_11 = False


    def _multi_hot(self, batch_labels):
        """
        :param batch_labels: label idx list of one batch, List[List[int]], e.g.  [[1,2],[0,1,3,4]]
        :return: multi-hot value for classification -> List[List[int]], e.g. [[0,1,1,0,0],[1,1,0,1,1]
        """
        batch_size = len(batch_labels)
        max_length = max([len(sample) for sample in batch_labels])
        aligned_batch_labels = []
        for sample_label in batch_labels:
            aligned_batch_labels.append(sample_label + (max_length - len(sample_label)) * [sample_label[0]])
        aligned_batch_labels = torch.Tensor(aligned_batch_labels).long()
        batch_labels_multi_hot = torch.zeros(batch_size, self.label_size).scatter_(1, aligned_batch_labels, 1)
        return batch_labels_multi_hot

    def __call__(self, batch):
        """
        transform data for training
        :param batch: Dict{'token': List[List[int]],
                           'label': List[List[int]],
                            'token_len': List[int],
                            'positive_sample': List[List[int]],
                            'negative_sample': List[List[int]],
                            'margin': List[List[int]]}
        :return: batch -> Dict{'token': torch.FloatTensor,
                               'label': torch.FloatTensor,
                               'token_len': torch.FloatTensor,
                               'label_list': List[List[int]],
                               'ranking_positive_mask': torch.LongTensor,
                               'ranking_negative_mask': torch.LongTensor,
                               'margin_mask': torch.BoolTensor}
        """
        batch_token = []
        batch_label = []
        batch_doc_len = []
        batch_ranking_positive_mask = []
        batch_ranking_negative_mask = []
        batch_margin_mask = []

        batch_negative_mask = []
        batch_negative_mask_label = []

        batch_input_ids = []
        batch_input_mask = []
        batch_segment_ids = []
        batch_input_len = []
        
        for sample_i, sample in enumerate(batch):
            batch_token.append(sample['token'])
            batch_label.append(sample['label'])
            batch_doc_len.append(sample['token_len'])
            if self.mode == "TRAIN":
                positive = np.zeros((self.positive_sample_num))
                negative = np.zeros((self.negative_sample_num))
                margin = np.zeros((self.negative_ratio, self.negative_sample_num))
                for i in range(self.positive_sample_num):
                    positive[i] = sample['positive_sample'][i]
                for i in range(self.negative_sample_num):
                    negative[i] = sample['negative_sample'][i]
                    # use mask matrix 'margin' to differentiate different sampling label
                    margin[sample['margin'][i], i] = 1
                batch_ranking_positive_mask.append(positive)
                batch_ranking_negative_mask.append(negative)
                batch_margin_mask.append(margin)
            batch_input_ids.append(sample['input_ids'])
            batch_input_mask.append(sample['input_mask'])
            batch_segment_ids.append(sample['segment_ids'])
            batch_input_len.append(sample['input_len'])

        batch_token = torch.tensor(batch_token)
        batch_multi_hot_label = self._multi_hot(batch_label)
        batch_doc_len = torch.FloatTensor(batch_doc_len)

        batch_input_ids = torch.LongTensor(batch_input_ids)
        batch_input_mask = torch.LongTensor(batch_input_mask)
        batch_segment_ids = torch.LongTensor(batch_segment_ids)
        batch_input_len = torch.LongTensor(batch_input_len)
        
        if self.mode == "TRAIN":
            batch_ranking_positive_mask = torch.LongTensor(batch_ranking_positive_mask)
            batch_ranking_negative_mask = torch.LongTensor(batch_ranking_negative_mask)

            batch_margin_mask = torch.BoolTensor(batch_margin_mask)
            batch_res = {
                'token': batch_token,
                'label': batch_multi_hot_label,
                'token_len': batch_doc_len,
                'label_list': batch_label,
                'ranking_positive_mask': batch_ranking_positive_mask,
                'ranking_negative_mask': batch_ranking_negative_mask,
                'margin_mask': batch_margin_mask,
                'input_ids': batch_input_ids,
                'input_mask': batch_input_mask,
                'segment_ids': batch_segment_ids,
                'input_len': batch_input_len
            }
        else:
            batch_res = {
                'token': batch_token,
                'label': batch_multi_hot_label,
                'token_len': batch_doc_len,
                'label_list': batch_label,
                'input_ids': batch_input_ids,
                'input_mask': batch_input_mask,
                'segment_ids': batch_segment_ids,
                'input_len': batch_input_len
            }
        return batch_res
