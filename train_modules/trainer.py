#!/usr/bin/env python
# coding:utf-8

import helper.logger as logger
from helper.utils import load_checkpoint, save_checkpoint
from train_modules.evaluation_metrics import evaluate
from data_modules.data_loader import data_loaders
import torch
import tqdm
import random
import numpy as np
import os
#for debug
torch.autograd.set_detect_anomaly(True)


class Trainer(object):
    def __init__(self, model, criterion, optimizer, scheduler, vocab, tokenizer, config, label_desc_loader=None, train_loader=None, dev_loader=None, test_loader=None):
        super(Trainer, self).__init__()
        self.model = model
        self.vocab = vocab
        self.config = config
        self.criterion, self.criterion_ranking = criterion[0], criterion[1]
        self.optimizer = optimizer
        self.dataset = config.data.dataset
        self.scheduler = scheduler
        self.eval_steps = config.eval.eval_steps
        
        self.train_end_epoch = config.train.end_epoch
        self.train_start_epoch = config.train.start_epoch
        self.begin_eval_epoch = config.eval.begin_eval_epoch
        self.refresh_data_loader = config.train.refresh_data_loader

        self.tokenizer = tokenizer
        self.model_type = config.model.type

        self.best_epoch_dev = [-1, -1]
        self.best_performance_dev = [-0.01, -0.01]
        self.best_performance_test = [-0.01, -0.01]
        self.global_step = 0
        
        self.label_desc_loader = label_desc_loader
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader

        self.loader_map = {"TRAIN": self.train_loader, "DEV": self.dev_loader, "TEST": self.test_loader}
        
    def update_lr(self):
        logger.warning('Learning rate update {}--->{}'
                       .format(self.optimizer.param_groups[0]['lr'],
                               self.optimizer.param_groups[0]['lr'] * self.config.train.optimizer.lr_decay))
        for param in self.optimizer.param_groups:
            param['lr'] = self.config.train.optimizer.learning_rate * self.config.train.optimizer.lr_decay

    def run_eval(self, epoch, mode="EVAL"):
        predict_probs = []
        target_labels = []
        label_repre = -1

        self.model.eval()
        for batch_i, batch in enumerate(tqdm.tqdm(self.loader_map[mode])):
            logits = self.model([batch, mode, label_repre])

            predict_results = torch.sigmoid(logits).cpu().tolist()
            predict_probs.extend(predict_results)
            target_labels.extend(batch['label_list'])

        performance = evaluate(predict_probs,
                               target_labels,
                               self.vocab,
                               self.config.eval.threshold)
        if performance['micro_f1'] > self.best_performance_dev[0]:
            logger.info('DEV Improve Micro-F1 {}% --> {}%'.format(self.best_performance_dev[0], performance['micro_f1']))
            self.best_performance_dev[0] = performance['micro_f1']
            self.best_epoch_dev[0] = epoch
            logger.info('Achieve best Micro-F1 on dev set, evaluate on test set')

            test_performance = self.run_eval(epoch, "TEST")
            if test_performance['micro_f1'] > self.best_performance_test[0]:
                logger.info('TEST Improve Micro-F1 {}% --> {}%'.format(self.best_performance_test[0], test_performance['micro_f1']))
                self.best_performance_test[0] = test_performance['micro_f1']


        if performance['macro_f1'] > self.best_performance_dev[1]:
            logger.info('DEV Improve Macro-F1 {}% --> {}%'.format(self.best_performance_dev[1], performance['macro_f1']))
            self.best_performance_dev[1] = performance['macro_f1']
            self.best_epoch_dev[1] = epoch
            logger.info('Achieve best Macro-F1 on dev set, evaluate on test set')
            test_performance = self.run_eval(epoch, "TEST")

            if test_performance['macro_f1'] > self.best_performance_test[1]:
                logger.info('TEST Improve Macro-F1 {}% --> {}%'.format(self.best_performance_test[1], test_performance['macro_f1']))
                self.best_performance_test[1] = test_performance['macro_f1']
        return performance

    def run_train(self, mode='TRAIN'):
        for epoch in range(self.train_start_epoch, self.train_end_epoch):
            if self.refresh_data_loader == 1:
                self.train_loader = data_loaders(self.config, self.vocab, bert_tokenizer=self.tokenizer, only_train=True)
            predict_probs = []
            target_labels = []
            total_loss = 0.0
            label_repre = -1
            if epoch % 10 == 0:
                logger.info("========== epoch: %d ==========" % (epoch))
                if epoch == 0:
                    logger.info("begin evaluation after epoch: %d" %(self.begin_eval_epoch))

            for batch_i, batch in enumerate(tqdm.tqdm(self.loader_map[mode])):
                if batch_i == 0 or batch_i % 50 == 0:
                    with torch.no_grad():
                        for batch_label_i, batch_label in enumerate(self.label_desc_loader):
                            label_embedding = self.model.get_embedding([batch_label, mode])
                            if batch_label_i == 0:
                               label_repre = label_embedding
                            else:
                               label_repre = torch.cat([label_repre, label_embedding], 0)
                logits, text_repre, label_repre_positive, label_repre_negative = self.model([batch, mode, label_repre])
                if self.config.train.loss.recursive_regularization.flag:
                    recursive_constrained_params = self.model.hiagm.linear.weight
                else:
                    recursive_constrained_params = None
                loss = self.criterion(logits,
                                      batch['label'].to(self.config.train.device_setting.device),
                                      recursive_constrained_params)
                total_loss += loss.item()

                loss_inter, loss_intra = self.criterion_ranking(text_repre, label_repre_positive, label_repre_negative, batch['margin_mask'].to(self.config.train.device_setting.device))
                self.optimizer.zero_grad()
                loss = loss + loss_inter + loss_intra

                loss.backward(retain_graph=True)

                if "bert" in self.config.model.type:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1, norm_type=2)
                    self.scheduler.step()
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1

                if (self.global_step+1) % self.eval_steps == 0 and epoch >= self.begin_eval_epoch:
                    performance = self.run_eval(epoch, "DEV")
                    self.model.train()

                predict_results = torch.sigmoid(logits).cpu().tolist()
                predict_probs.extend(predict_results)
                target_labels.extend(batch['label_list'])
            logger.info("loss: %f" % (total_loss/len(self.loader_map[mode])))

