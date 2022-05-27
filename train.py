#!/usr/bin/env python
# coding:utf-8

import helper.logger as logger
from models.model import HiMatch
import torch
import sys
from helper.configure import Configure
import os
from data_modules.data_loader import data_loaders
from data_modules.vocab import Vocab
from train_modules.criterions import ClassificationLoss, MarginRankingLoss
from train_modules. trainer import Trainer
from helper.utils import load_checkpoint, save_checkpoint
import time
import random
from helper.lr_schedulers import get_linear_schedule_with_warmup
from helper.adamw import AdamW

from transformers import BertTokenizer

def set_optimizer(config, model):
    """
    :param config: helper.configure, Configure Object
    :param model: computational graph
    :return: torch.optim
    """
    params = model.optimize_params_dict()
    if config.train.optimizer.type == 'Adam':
        return torch.optim.Adam(lr=config.train.optimizer.learning_rate,
                                params=params)
    else:
        raise TypeError("Recommend the Adam optimizer")


def train(config):
    """
    :param config: helper.configure, Configure Object
    """
    # loading corpus and generate vocabulary
    corpus_vocab = Vocab(config,
                         min_freq=5,
                         max_size=config.vocabulary.max_token_vocab)
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
    
    # get data
    train_loader, dev_loader, test_loader, label_desc_loader = data_loaders(config, corpus_vocab, bert_tokenizer=tokenizer)
    
    # build up model
    himatch = HiMatch(config, corpus_vocab, model_mode='TRAIN')
    himatch.to(config.train.device_setting.device)
    # define training objective & optimizer
    criterion = ClassificationLoss(os.path.join(config.data.data_dir, config.data.hierarchy),
                                   corpus_vocab.v2i['label'],
                                   recursive_penalty=config.train.loss.recursive_regularization.penalty,
                                   recursive_constraint=config.train.loss.recursive_regularization.flag, loss_type="bce")
    # define ranking loss
    criterion_ranking = MarginRankingLoss(config)
    if "bert" in config.model.type:
        t_total = int(len(train_loader) * (config.train.end_epoch-config.train.start_epoch))
    
        param_optimizer = list(himatch.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
                                        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],'weight_decay': 0.01},
                                        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                                        ]
        warmup_steps = int(t_total * 0.1)
        optimizer = AdamW(optimizer_grouped_parameters, lr=config.train.optimizer.learning_rate, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=t_total)
    else:
        optimizer = set_optimizer(config, himatch)
        scheduler = None
                                                 
    # get epoch trainer
    trainer = Trainer(model=himatch,
                      criterion=[criterion, criterion_ranking],
                      optimizer=optimizer,
                      vocab=corpus_vocab,
                      tokenizer=tokenizer,
                      scheduler=scheduler,
                      config=config,
                      label_desc_loader=label_desc_loader,
                      train_loader=train_loader,
                      dev_loader=dev_loader,
                      test_loader=test_loader)

    # set origin log
    best_epoch_dev = [-1, -1]
    best_performance_dev = [0.0, 0.0]
    best_performance_test = [0.0, 0.0]
    model_checkpoint = config.train.checkpoint.dir
    model_name = config.model.type
    wait = 0
    if not os.path.isdir(model_checkpoint):
        os.mkdir(model_checkpoint)
    else:
        # loading previous checkpoint
        dir_list = os.listdir(model_checkpoint)
        dir_list.sort(key=lambda fn: os.path.getatime(os.path.join(model_checkpoint, fn)))
        latest_model_file = 'best_micro_HiMatch'
        if os.path.isfile(os.path.join(model_checkpoint, latest_model_file)):
            logger.info('Loading Previous Checkpoint...')
            logger.info('Loading from {}'.format(os.path.join(model_checkpoint, latest_model_file)))
            best_performance_dev, config = load_checkpoint(model_file=os.path.join(model_checkpoint, latest_model_file),
                                                       model=himatch,
                                                       config=config,
                                                       optimizer=optimizer)
            logger.info('Previous Best Performance---- Micro-F1: {}%, Macro-F1: {}%'.format(
                best_performance_dev[0], best_performance_dev[1]))

    # train
    trainer.run_train()

if __name__ == "__main__":
    configs = Configure(config_json_file=sys.argv[1])

    if configs.train.device_setting.device == 'cuda':
        os.system('CUDA_VISIBLE_DEVICES=' + str(configs.train.device_setting.visible_device_list))
    else:
        os.system("CUDA_VISIBLE_DEVICES=''")
    random.seed(2021)
    torch.manual_seed(2021)
    torch.cuda.manual_seed(2021)
    logger.Logger(configs)

    if not os.path.isdir(configs.train.checkpoint.dir):
        os.mkdir(configs.train.checkpoint.dir)

    train(configs)
