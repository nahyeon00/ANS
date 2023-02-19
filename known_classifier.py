from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel, BertTokenizer, AdamW, BertConfig
import pandas as pd
import torch.nn as nn
import numpy as np
import torch
import argparse
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import os
import random
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.multiclass import OneVsRestClassifier


from data_info import * 
from c_way_dataprocess import * 


class BERTfeature(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)

        # data
        self.dataset = args.dataset  # 저장 파일명 위해 필요
        self.known_cls_ratio = args.known_cls_ratio  # 저장 파일명 위해 필요

        self.num_labels = args.num_labels
        self.mode = args.mode

        # use pretrained BERT
        model_config = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.bert = BertModel.from_pretrained('bert-base-uncased', config=model_config)

        self.c_way_classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size,  self.num_labels),
            nn.ReLU()
        )
        self.rest_classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.bert.config.hidden_dropout_prob),
            nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.bert.config.hidden_dropout_prob),
            nn.Linear(self.bert.config.hidden_size, 1),
            nn.ReLU()
        )
        self.classifier_list  = [self.rest_classifier for i in range(self.num_labels)]
        # self.classifier = OneVsRestClassifier
        
        self.__build_loss()
        

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, mode=None,
                 num=None, infer=None, mid=None, mean_noise=None):
        if mid == True:
            # output [last_hidden_state, pooler_output, hidden_states]  -> last hidden layer
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            print("outputs")
            # [128, 45, 768]
            last_hidden_layer = outputs[0]
            print("last_hidden_layer", last_hidden_layer.shape)
            # mean pooling [128, 768]
            mean_pooling = last_hidden_layer.mean(dim=1)
            print("mean_pooling", mean_pooling.shape)
            # train known
            if mode == 'feature_train':

                logits = self.c_way_classifier(mean_pooling)
                # print("4", logits.shape)

                return mean_pooling, logits
            # train ANS
            else:
                if infer == True:  # inference
                    classifier = self.classifier_list[num[0]]
                    r_logits = classifier(mean_pooling)
                    c_logits = self.c_way_classifier(mean_pooling)
                    # print("5")

                    return r_logits, c_logits
                else:  # train
                    print("known train else")
                    classifier = self.classifier_list[num[0]]
                    print("classifier")
                    logits = classifier(mean_pooling)
                    print("logits")

                    return mean_pooling, logits
        else:
            classifier = self.classifier_list[num[0]]
            print("mid classifier",mean_noise)
            logits = classifier(mean_noise)
            print("mid logits")

            return mean_noise, logits
                

    
    def training_step(self, batch, batch_idx):
        print("training step")
        # batch
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']
        label_id = batch['label_id']
        
        # fwd
        _, logits = self.forward(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, 
                                    mode=self.mode, mid=True)

        # loss
        loss = self._loss(logits, label_id.long().squeeze(-1))

        
        # logs
        tensorboard_logs = {'train_loss': loss}

        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)

        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        print("start validation")
        # batch
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']
        label_id = batch['label_id']
         
        # fwd
        _, logits = self.forward(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, 
                                    mode=self.mode, mid=True)

        # loss
        loss = self._loss(logits, label_id.long().squeeze(-1))

        self.log('val_loss', loss)

        return loss
    
    
    
    def test_step(self, batch, batch_idx):
        # batch
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']
        label_id = batch['label_id']
        
        _, logits = self.forward(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, 
                                    mode=self.mode, mid=True)

        total_probs = F.softmax(logits.float().detach(), dim=1)
        total_maxprobs, total_preds = total_probs.max(dim = 1)

        y_pred = total_preds.cpu().numpy()
        y_true = label_id.cpu().numpy()
        test_acc = accuracy_score(y_true, y_pred)
        eval_score = round(test_acc * 100, 2)
        test_acc = torch.tensor(test_acc)

        
        self.log_dict({'test_acc': test_acc})
        
        return {'test_acc': test_acc}
    
    def class_count(self, labels):
        class_data_num = []
        for l in np.unique(labels):
            num = len(labels[labels == l])
            class_data_num.append(num)
        return class_data_num
    
    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        param_optimizer = list(self.bert.named_parameters())
        param_optimizer += list(self.c_way_classifier.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=2e-05)

        return optimizer
    
    def __build_loss(self):
        self._loss = nn.CrossEntropyLoss()

