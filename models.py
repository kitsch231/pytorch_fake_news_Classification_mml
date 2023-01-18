import torch.nn as nn
from transformers import BertModel
import torch
import torch.nn.functional as F
import numpy as np
import math
import torch.utils.model_zoo as model_zoo
from resnet_models import *



class Mynet(nn.Module):
    def __init__(self,config):
        super(Mynet, self).__init__()
        self.config=config
        resnet_name=self.config.resnet_name#选取resnet种类
        if resnet_name=='resnet18':
            self.resnet=resnet18(self.config.resnet_fc)
        elif resnet_name=='resnet34':
            self.resnet=resnet34(self.config.resnet_fc)
        elif resnet_name=='resnet50':
            self.resnet=resnet50(self.config.resnet_fc)
        elif resnet_name=='resnet101':
            self.resnet=resnet101(self.config.resnet_fc)
        elif resnet_name=='resnet152':
            self.resnet=resnet152(self.config.resnet_fc)

        self.bert= BertModel.from_pretrained(self.config.bert_name)#bert的种类

        self.fc_1 = nn.Linear(self.config.bert_fc+self.config.resnet_fc, self.config.num_classes)
        self.drop=nn.Dropout(self.config.dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self,inx):
        # BERT
        img,tokens,mask=inx

        # attention_mask=mask
        img=self.resnet(img)

        outputs = self.bert(tokens,attention_mask=mask)
        #emb (32,128)-(32,768)
        pooled_output = outputs[1]
        pooled_output=self.drop(pooled_output)
        logits=torch.cat([img,pooled_output],1)
        logits = self.fc_1(logits) (32,768)-(32,2)
        logits=self.softmax(logits)

        return logits
