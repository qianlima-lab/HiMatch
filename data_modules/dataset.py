#!/usr/bin/env python
# coding: utf-8

from torch.utils.data.dataset import Dataset
from models.structure_model.tree import Tree
import helper.logger as logger
import json
import os
import random
from helper.utils import get_hierarchy_relations, get_parent, get_sibling

def get_sample_position(corpus_filename, on_memory, corpus_lines, stage):
    """
    position of each sample in the original corpus File or on-memory List
    :param corpus_filename: Str, directory of the corpus file
    :param on_memory: Boolean, True or False
    :param corpus_lines: List[Str] or None, on-memory Data
    :param mode: Str, 'TRAIN'/'DEV'/'TEST', log the corpus
    :return: sample_position -> List[int]
    """
    sample_position = [0]
    if not on_memory:
        print('Loading files for ' + stage + ' Dataset...')
        with open(corpus_filename, 'r') as f_in:
            sample_str = f_in.readline()
            while sample_str:
                sample_position.append(f_in.tell())
                sample_str = f_in.readline()
            sample_position.pop()
    else:
        assert corpus_lines
        sample_position = range(len(corpus_lines))
    return sample_position


class ClassificationDataset(Dataset):
    def __init__(self, config, vocab, stage='TRAIN', on_memory=True, corpus_lines=None, mode="TRAIN", bert_tokenizer=None):
        """
        Dataset for text classification based on torch.utils.data.dataset.Dataset
        :param config: helper.configure, Configure Object
        :param vocab: data_modules.vocab, Vocab Object
        :param stage: Str, 'TRAIN'/'DEV'/'TEST', log the corpus
        :param on_memory: Boolean, True or False
        :param corpus_lines: List[Str] or None, on-memory Data
        :param mode: TRAIN / PREDICT, for loading empty label
        """
        super(ClassificationDataset, self).__init__()
        # new
        self.corpus_files = {"TRAIN": os.path.join(config.data.data_dir, config.data.train_file),
                             "VAL": os.path.join(config.data.data_dir, config.data.val_file),
                             "TEST": os.path.join(config.data.data_dir, config.data.test_file),
                             "DESC": os.path.join(config.data.data_dir, config.data.label_desc_file)}
        self.config = config
        self.vocab = vocab
        self.on_memory = on_memory
        self.data = corpus_lines
        self.max_input_length = self.config.text_encoder.max_length
        self.corpus_file = self.corpus_files[stage]
        self.sample_position = get_sample_position(self.corpus_file, self.on_memory, corpus_lines, stage)
        self.corpus_size = len(self.sample_position)
        self.mode = mode
        self.stage = stage
        self.sample_num = config.data.positive_num
        self.negative_ratio = config.data.negative_ratio
        self.dataset = config.data.dataset

        # parent_id to child_id
        self.get_child = get_hierarchy_relations(os.path.join(config.data.data_dir,
                                                              config.data.hierarchy),
                                                 self.vocab.v2i['label'],
                                                 root=Tree(-1),
                                                 fortree=False)
        # child_id to parent_id eurlex use file
        self.get_parent = get_parent(self.get_child, config, self.vocab)
        # child_id to sibling_id
        self.get_sibling, self.first_layer = get_sibling(self.get_child, config, self.vocab)

        self.tokenizer = bert_tokenizer

    def __len__(self):
        """
        get the number of samples
        :return: self.corpus_size -> Int
        """
        return self.corpus_size

    def __getitem__(self, index):
        """
        sample from the overall corpus
        :param index: int, should be smaller in len(corpus)
        :return: sample -> Dict{'token': List[Str], 'label': List[Str], 'token_len': int}
        """
        if index >= self.__len__():
            raise IndexError
        if not self.on_memory:
            position = self.sample_position[index]
            with open(self.corpus_file) as f_in:
                f_in.seek(position)
                sample_str = f_in.readline()
        else:
            sample_str = self.data[index]
        return self._preprocess_sample(sample_str)

    def create_features(self, sentences, max_seq_len=256):
        tokens_a = self.tokenizer.tokenize(sentences)
        tokens_b = None

        if len(tokens_a) > max_seq_len - 2:
            tokens_a = tokens_a[:max_seq_len - 2]
        tokens = ['[CLS]'] + tokens_a + ['[SEP]']
        segment_ids = [0] * len(tokens)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding = [0] * (max_seq_len - len(input_ids))
        input_len = len(input_ids)

        input_ids   += padding
        input_mask  += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_len
        assert len(input_mask) == max_seq_len
        assert len(segment_ids) == max_seq_len

        feature = {'input_ids': input_ids, 'input_mask': input_mask, 'segment_ids': segment_ids, 'input_len': input_len}
        return feature

        
    def _preprocess_sample(self, sample_str):
        """
        preprocess each sample with the limitation of maximum length and pad each sample to maximum length
        :param sample_str: Str format of json data, "Dict{'token': List[Str], 'label': List[Str]}"
        :return: sample -> Dict{'token': List[int], 'label': List[int], 'token_len': int, 'positive_sample': List[int], 'negative_sample': List[int], 'margin': List[int]}
        """
        raw_sample = json.loads(sample_str)
        sample = {'token': [], 'label': [], 'positive_sample': [], 'negative_sample': [], 'margin': []}
        for k in raw_sample.keys():
            if k == 'token':
                
                sample[k] = [self.vocab.v2i[k].get(v.lower(), self.vocab.oov_index) for v in raw_sample[k]]
                
                sentences = " ".join(raw_sample[k])
                features = self.create_features(sentences, self.max_input_length)
                for (features_k, features_v) in features.items():
                    sample[features_k] = features_v
                    
            elif k == 'label':
                sample[k] = []
                if self.sample_num > len(raw_sample[k]):
                    ranking_true_sample = random.sample(raw_sample[k], len(raw_sample[k]))
                else:
                    ranking_true_sample = random.sample(raw_sample[k], self.sample_num)
                for v in raw_sample[k]:
                    if v not in self.vocab.v2i[k].keys():
                        logger.warning('Vocab not in ' + k + ' ' + v)
                    else:
                        sample[k].append(self.vocab.v2i[k][v])
                    if self.stage == "TRAIN" and v in ranking_true_sample:
                        p_int = self.vocab.v2i['label'][v]
                        sample['positive_sample'].append(p_int)
                        # its parent (no parent -> use other positive label）
                        if p_int in self.first_layer or p_int not in self.get_parent:
                            parent_list = [p_int]
                        else:
                            parent_list = self.get_parent[p_int]
                        n_int0 = random.choice(parent_list)
                        sample['negative_sample'].append(n_int0)
                        sample['margin'].append(0)

                        # its sibling
                        if p_int not in self.get_sibling:
                            n_int1 = random.randint(0, len(self.vocab.v2i['label']) - 1)
                            while self.vocab.i2v['label'][n_int1] in raw_sample['label']:
                                n_int1 = random.randint(0, len(self.vocab.v2i['label']) - 1)
                            sibling_list = [n_int1]
                        else:
                            sibling_list = self.get_sibling[p_int]
                        n_int1 = random.choice(sibling_list)
                        sample['negative_sample'].append(n_int1)
                        sample['margin'].append(1)

                        #random
                        n_int2 = random.randint(0, len(self.vocab.v2i['label']) - 1)
                        while self.vocab.i2v['label'][n_int2] in raw_sample['label']:
                            n_int2 = random.randint(0, len(self.vocab.v2i['label']) - 1)
                        sample['negative_sample'].append(n_int2)
                        sample['margin'].append(2)

                        #logger.info("p_int:" + self.vocab.i2v['label'][p_int] + " " + "parent:" + self.vocab.i2v['label'][n_int0] + " " + "sibling:" + self.vocab.i2v['label'][n_int1] + " random:" + self.vocab.i2v['label'][n_int2])

                if self.stage == "TRAIN" and self.sample_num > len(raw_sample['label']):
                    oversize = self.sample_num - len(raw_sample['label'])
                    while oversize > 0:
                        total_index = list(range(0, len(raw_sample['label']), 1))
                        select_index = random.choice(total_index)
                        sample['positive_sample'].append(sample['positive_sample'][select_index])
                        for pad in range(self.negative_ratio):
                            sample['negative_sample'].append(
                                sample['negative_sample'][self.negative_ratio * select_index + pad])
                            sample['margin'].append(sample['margin'][self.negative_ratio * select_index + pad])
                        oversize -= 1
        if not sample['token']:
            sample['token'].append(self.vocab.padding_index)
        sample['token_len'] = min(len(sample['token']), self.max_input_length)
        padding = [self.vocab.padding_index for _ in range(0, self.max_input_length - len(sample['token']))]
        sample['token'] += padding
        sample['token'] = sample['token'][:self.max_input_length]
        return sample
