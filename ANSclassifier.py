from sklearn.multiclass import OneVsRestClassifier


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
from torch.nn.parameter import Parameter

from known_classifier import *
from ng_dataprocess import *


def load_model_from_experiment(args, pre):
        """Function that loads the model from an experiment folder.
        :param experiment_folder: Path to the experiment folder.
        Return:
            - Pretrained model.
        """
        # hparams_file = experiment_folder + "/hparams.yaml"
        
        # hparams_file = "/workspace/intent/newADB/lightning_logs/version_7/hparams.yaml"

        # hparams = yaml.load(open(hparams_file).read(), Loader=yaml.FullLoader)

        checkpoints = [
            file
            for file in os.listdir("/workspace/intent/ANS/checkpoints/")
            if file.endswith(".ckpt")
        ]
        # checkpoint_path = ckpt_path + checkpoints[-1]
        # model = BERTfeature.load_from_checkpoint(
        #     checkpoint_path, hparams=Namespace(**hparams)
        # )
        print("name", checkpoints[-1])
        checkpoint_path = args.ckpt_path + "/checkpoints/" + checkpoints[-1]
        print("checkpoint", checkpoint_path)
        model = BERTfeature.load_from_checkpoint(checkpoint_path, args =args)
        return model


class ANS(pl.LightningModule):

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.args = args
        self.mode = args.mode
        self.noise_grad_step = args.noise_grad_step
        self.radius = args.radius
        self.gamma = args.gamma

        # data
        print("argas",args.dataset)
        self.dataset = args.dataset
        self.known_cls_ratio = args.known_cls_ratio

        self.num_labels = args.num_labels

        # 모델, 중심 불러오기
        checkpoints = [
            file
            for file in os.listdir("/workspace/intent/ANS/checkpoints/")
            if file.endswith(".ckpt")
        ]

        checkpoint_path = args.ckpt_path + "/checkpoints/" + checkpoints[0]

        self.model = BERTfeature.load_from_checkpoint(checkpoint_path, args=args)  # known에서 학습한 모델
        
        # model_config = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
        # self.bert = BertModel.from_pretrained('bert-base-uncased', config=model_config)
        self.rest_classifier= self.model.classifier_list
        
        # freeze
        # for name, param in self.model.named_parameters():  
        #     param.requires_grad = False
        
        self.__build_loss()
        


    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, mode=None,
                 num=None, infer=None, mid=None, mean_noise=None):
        mean_pooling, logits = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, 
                            mode=mode, num=num, infer=infer, mid=mid, mean_noise=mean_noise)
        return mean_pooling, logits
    
    def training_step(self, batch, batch_idx):
        print("training step")
        # batch
        pos_batch = batch['pos']
        neg_batch = batch['neg']

        # pos batch
        pos_input_ids = pos_batch['input_ids']
        pos_attention_mask = pos_batch['attention_mask']
        pos_token_type_ids = pos_batch['token_type_ids']
        pos_label_id = pos_batch['label_id']
        pos_num = pos_batch['num']

        # neg batch
        neg_input_ids = neg_batch['input_ids']
        neg_attention_mask = neg_batch['attention_mask']
        neg_token_type_ids = neg_batch['token_type_ids']
        neg_label_id = neg_batch['label_id']
        neg_num = neg_batch['num']
        
        # fwd
        pos_feature, pos_logits = self.forward(input_ids=pos_input_ids, attention_mask=pos_attention_mask, 
                                                token_type_ids=pos_token_type_ids, 
                                                mode=self.mode, num=pos_num, infer=False, mid=True)
        print("1", pos_logits.size())
        _, neg_logits = self.forward(input_ids=neg_input_ids, attention_mask=neg_attention_mask, 
                                    token_type_ids=neg_token_type_ids, 
                                    mode=self.mode, num=pos_num, infer=False, mid=True)
        print("2")
    
        # loss
        pos_loss = self._lossBC((-pos_logits), pos_label_id.float())
        print("3")

        # neg_total_maxprobs, neg_total_preds = neg_logits.max(dim=1)
        neg_loss = self._lossBC(neg_logits, neg_label_id.float())
        print("4")
        real_loss = pos_loss + neg_loss


        # ANS
        noise =torch.empty(pos_feature.size()).normal_(mean=0, std=4*abs(torch.sum(pos_feature).item()))
        noise_pp = Parameter(noise.to(self.device),requires_grad=True)
        op_noise = torch.optim.Adam([noise_pp], lr=3e-4)
        print("noise_pp before", noise_pp)

        #ANS classifier
        print("ans")
        _, ans_logits = self.forward(num=pos_num,mid=False, mean_noise=(pos_feature+noise_pp))

        for i in range(self.noise_grad_step):
            print("ans for")
            ascend_loss = self._lossBC(ans_logits, neg_label_id.float())
            ascend_loss.requires_grad_(True)
            ascend_loss.backward(retain_graph=True)
            op_noise.step()
        
        # projection
        alpha = torch.empty(pos_feature.size())
        print("alpha")
        alpha = alpha_cal(alpha, self.radius, (pos_logits+noise_pp).to(self.device), pos_logits, self.gamma)
        noise_pp = (alpha.to(self.device)/torch.abs(noise).to(self.device))*noise_pp  # alpha

        print("pos feature", pos_feature)
        print("noise_pp", noise_pp)
        _, ans_logits = self.forward(num=pos_num,mid=False, mean_noise=(pos_feature+noise_pp))
        syn_loss = self._lossBC(ans_logits, neg_label_id.float())
        print("5")
        final_loss = real_loss + syn_loss

        # logs
        tensorboard_logs = {'train_loss': final_loss}

        self.log("train_loss", final_loss, on_epoch=True, prog_bar=True, logger=True)

        return {'loss': final_loss, 'log': tensorboard_logs}
    
    def validation_step(self, batch, batch_idx):
        print("start validation")
        # batch
        pos_batch = batch['pos']
        neg_batch = batch['neg']

        # pos batch
        pos_input_ids = pos_batch['input_ids']
        pos_attention_mask = pos_batch['attention_mask']
        pos_token_type_ids = pos_batch['token_type_ids']
        pos_label_id = pos_batch['label_id']
        pos_num = pos_batch['num']

        # neg batch
        neg_input_ids = neg_batch['input_ids']
        neg_attention_mask = neg_batch['attention_mask']
        neg_token_type_ids = neg_batch['token_type_ids']
        neg_label_id = neg_batch['label_id']
        neg_num = neg_batch['num']
        
        # fwd
        pos_feature, pos_logits = self.forward(input_ids=pos_input_ids, attention_mask=pos_attention_mask, 
                                                token_type_ids=pos_token_type_ids, 
                                                mode=self.mode, num=pos_num, infer=False, mid=True)
        print("1", pos_feature, pos_feature.size())

        _, neg_logits = self.forward(input_ids=neg_input_ids, attention_mask=neg_attention_mask, 
                                    token_type_ids=neg_token_type_ids, 
                                    mode=self.mode, num=pos_num, infer=False, mid=True)
        print("2", neg_logits)


    
        # loss
        pos_loss = self._lossBC((-pos_logits), pos_label_id.float())
        print("3")

        # neg_total_maxprobs, neg_total_preds = neg_logits.max(dim=1)
        neg_loss = self._lossBC(neg_logits, neg_label_id.float())
        print("4")
        real_loss = pos_loss + neg_loss


        # ANS
        noise =torch.empty(pos_feature.size()).normal_(mean=0, std=4*abs(torch.sum(pos_feature).item()))
        noise_pp = Parameter(noise.to(self.device),requires_grad=True)
        op_noise = torch.optim.Adam([noise_pp], lr=3e-4)

        #ANS classifier
        print("ans")
        _, ans_logits = self.forward(num=pos_num,mid=False, mean_noise=(pos_feature+noise_pp))

        for i in range(self.noise_grad_step):
            print("ans for")
            ascend_loss = self._lossBC(ans_logits, neg_label_id.float())
            ascend_loss.requires_grad_(True)
            ascend_loss.backward(retain_graph=True)
            op_noise.step()

        # projection
        alpha = torch.empty(pos_feature.size())
        print("alpha")
        alpha = alpha_cal(alpha, self.radius, (pos_logits+noise_pp).to(self.device), pos_logits, self.gamma)
        noise_pp = (alpha.to(self.device)/torch.abs(noise).to(self.device))*noise_pp  # alpha

        _, ans_logits = self.forward(num=pos_num,mid=False, mean_noise=(pos_feature+noise_pp))
        syn_loss = self._lossBC(ans_logits, neg_label_id.float())
        print("5")
        val_final_loss = real_loss + syn_loss
        
        # y_pred = total_preds.cpu().numpy()
        # y_true = label_id.cpu().numpy()

        # val_acc = accuracy_score(y_true, y_pred)
        # eval_score = round(val_acc * 100, 2)

        self.log('val_loss', val_final_loss)
        
        return val_final_loss

    def validation_epoch_end(self, outputs):
        losses = []
        for loss in outputs:
            losses.append(loss)
        val_loss_mean = torch.stack(losses).mean()
        self.log('val_loss', val_loss_mean, prog_bar=True)

    def test_step(self, batch, batch_idx):
        # batch
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']
        label_id = batch['label_id']

        # fwd
        for i in range(self.num_labels):
            r_logits, c_logits = self.forward(input_ids, attention_mask, token_type_ids, self.mode, pos_num, True)
            if r_logits > 0 : 
                preds = c_logits
                
            if i == self.num_labels-1:
                preds = self.self.unseen_label_id




    def __build_loss(self):
        """Initializes the loss function/s."""
        self._lossBC = nn.BCEWithLogitsLoss()
    
    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        param_optimizer = []
        for i in range(len(self.rest_classifier)):
            param_optimizer += list(self.rest_classifier[i].named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        if self.dataset == 'stackoverflow':
            learning_rate = 3e-4
        else:
            learning_rate = 1e-3
        
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=learning_rate)

        return optimizer


def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    dis = -((a - b)**2).sum(dim=2)
    return dis


def alpha_cal(alpha, radius, ans, feature, gamma):
    dis = torch.norm((ans-feature).float(),2,1).unsqueeze(-1)
    k = gamma*radius
    for i in range(len(dis)):
        for j in range(len(dis[i])):
            if radius<=dis[i][j] and dis[i][j]<=k:
                alpha[i][j] = 1
            elif k<=dis[i][j]:
                alpha[i][j] = k/dis[i][j]
            elif dis[i][j]<radius:
                alpha[i][j] = radius/dis[i][j]

    return alpha