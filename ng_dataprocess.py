from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
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
from pytorch_lightning.trainer.supporters import CombinedLoader

from data_info import * 

class ANSDataset(Dataset):
    def __init__(self, process_data, max_seq_len, label_list, num, pos):
        self.data = process_data
        print("data len", len(self.data))
        self.pos = pos

        self.max_seq_len = max_seq_len
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True) 
        self.label_list = label_list
        self.label_map = {}
        for i, label in enumerate(self.label_list):
            self.label_map[label] = i
        self.num = num
        print("num", self.num)

        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data = self.data[index]
        doc = data['text']

        features = self.tokenizer(str(doc), padding='max_length', max_length= self.max_seq_len, truncation=True, return_tensors='pt') 

        input_ids = features['input_ids'].squeeze(0)
        attention_mask = features['attention_mask'].squeeze(0)
        token_type_ids = features['token_type_ids'].squeeze(0)
        ori_label = data['label']
        # label_id = torch.tensor([self.label_map[ori_label]], dtype = torch.long)

        # pos 데이터일 경우 1, neg 데이터일 경우 -1
        if self.pos == 'pos':
            label_id = torch.tensor([1])  # pos
        else:
            label_id = torch.tensor([0])  # neg
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'label_id': label_id,
            'num': self.num
        }



class ANSDataModule(pl.LightningDataModule):
    def __init__(self, args):
        self.dataset = args.dataset
        self.data_path = os.path.join(args.data_path, self.dataset)
        self.train_data_path = f'{self.data_path}/train.tsv'
        self.val_data_path = f'{self.data_path}/dev.tsv'
        self.test_data_path = f'{self.data_path}/test.tsv'

        self.max_seq_len = max_seq_lengths[self.dataset]
        self.batch_size = args.batch_size
        self.known_cls_ratio = args.known_cls_ratio
        self.worker = args.num_workers
        self.labeled_ratio = args.labeled_ratio
        self.mode = args.mode

        ####### label list
        self.all_label_list = benchmark_labels[self.dataset]
        self.n_known_cls = round(len(self.all_label_list) * self.known_cls_ratio)
        self.known_label_list = np.random.choice(np.array(self.all_label_list, dtype=str), self.n_known_cls, replace=False)
        self.known_label_list = self.known_label_list.tolist()
        print("known_label_list", self.known_label_list)

        args.num_labels = self.num_labels = len(self.known_label_list)
        print("num_labels", self.num_labels)


        if self.dataset == 'oos':
            self.unseen_label = 'oos'
        else:
            self.unseen_label = '<UNK>'
        
        self.unseen_label_id = self.num_labels
        self.label_list = self.known_label_list + [self.unseen_label]

        self.label_map = {}
        for i, label in enumerate(self.label_list):
            self.label_map[label] = i
        
    def setup(self, stage):

        if stage in (None, 'fit'):
            # prepare known samples
            self.train_data = pd.read_csv(self.train_data_path, delimiter="\t")
            self.valid_data = pd.read_csv(self.val_data_path, delimiter="\t")

            self.train_examples = [[] for x in range(self.num_labels)]
            self.val_examples = [[] for x in range(self.num_labels)]

            for i in self.train_data.index:
                cur_label = self.train_data.loc[i]['label']
                if (cur_label in self.known_label_list) and (np.random.uniform(0, 1) <= self.labeled_ratio):
                    num = self.label_map[cur_label]
                    self.train_examples[num].append(self.train_data.iloc[i])  # 라벨 인덱스에 따라 데이터 저장
            for i in self.valid_data.index:
                cur_label = self.valid_data.loc[i]['label']
                if (cur_label in self.known_label_list):
                    num = self.label_map[cur_label]
                    self.val_examples[num].append(self.valid_data.iloc[i])
            
            # for binary classifier
            for i in range(self.num_labels):
                train_pos = self.train_examples[i]
                train_neg_ori = sum(self.train_examples[0:i]+self.train_examples[i+1:self.num_labels], [])
                train_neg = random.sample(train_neg_ori, len(train_pos))

                val_pos = self.val_examples[i]
                val_neg_ori = sum(self.val_examples[0:i]+self.val_examples[i+1:self.num_labels], [])
                val_neg = random.sample(val_neg_ori, len(val_pos))
            

                print("train pos", len(train_pos))
                print("train neg", len(train_neg))

                print("val_pos", len(val_pos))
                print("val_neg", len(val_neg))

                self.train_pos = ANSDataset(train_pos, self.max_seq_len, self.label_list, i, 'pos')
                self.train_neg = ANSDataset(train_neg, self.max_seq_len, self.label_list, i,'neg')

                self.valid_pos = ANSDataset(val_pos, self.max_seq_len, self.label_list, i, 'pos')
                self.valid_neg = ANSDataset(val_neg, self.max_seq_len, self.label_list, i, 'neg')

    

        elif stage in (None, 'test'):
            # prepare
            self.test_data = pd.read_csv(self.test_data_path, delimiter="\t")

            self.test_examples = []
            if self.mode == 'feature_train':
                for i in self.test_data.index:
                    cur_label = self.test_data.loc[i]['label']
                    if (cur_label in self.known_label_list):
                        self.test_examples.append(self.test_data.iloc[i])
            else:
                for i in self.test_data.index:
                    cur_label = self.test_data.loc[i]['label']
                    if (cur_label in self.label_list) and (cur_label is not self.unseen_label):
                        self.test_examples.append(self.test_data.iloc[i])
                    else:
                        self.test_data.loc[i]['label'] = self.unseen_label
                        self.test_examples.append(self.test_data.iloc[i])

            self.test = ANSDataset(self.test_examples, self.max_seq_len, self.label_list)
        
    def train_dataloader(self):
        # sampler = RandomSampler(self.train)
        train_pos_loader = DataLoader(self.train_pos, batch_size=self.batch_size, num_workers= self.worker)
        train_neg_loader = DataLoader(self.train_neg, batch_size=self.batch_size, num_workers= self.worker)

        loaders = {'pos': train_pos_loader, "neg": train_neg_loader}
    
        combined_loaders = CombinedLoader(loaders, mode="max_size_cycle")

        return combined_loaders
    
    def val_dataloader(self):
        # sampler = SequentialSampler(self.valid)
        val_pos_loader = DataLoader(self.valid_pos, batch_size=self.batch_size, num_workers= self.worker)
        val_neg_loader = DataLoader(self.valid_neg, batch_size=self.batch_size, num_workers= self.worker)

        loaders = {'pos': val_pos_loader, "neg": val_neg_loader}
    
        combined_loaders = CombinedLoader(loaders, mode="max_size_cycle")

        return combined_loaders
    
    def test_dataloader(self):
        sampler = SequentialSampler(self.test)
        return DataLoader(self.test, batch_size=self.batch_size, num_workers= self.worker, sampler = sampler)
    
    def predict_dataloader(self):
        sampler = RandomSampler(self.train)
        return DataLoader(self.train, batch_size=self.batch_size, num_workers= self.worker, sampler = sampler)
