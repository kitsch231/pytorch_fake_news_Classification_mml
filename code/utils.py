import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms as transforms
import cv2
from transformers import BertTokenizer
from Config import Config
# a.通过词典导入分词器
#"bert-base-chinese"


#bert_model/chinese-bert-wwm-ext

#tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
class My_Dataset(Dataset):
    def __init__(self,path,config,iftrain):#### 读取数据集
        self.config=config
        #启用训练模式，加载数据和标签
        #D:\A_sell_project\cv\多模态虚假新闻分类\data\train.csv
        self.iftrain=iftrain
        df = pd.read_csv(path).sample(frac=self.config.frac)
        self.img_path = df['path'].to_list() #[img]
        self.text = df['text'].to_list()
        self.tokenizer = BertTokenizer.from_pretrained(self.config.bert_name)

        #启用训练模式，加载数据和标签
        if self.iftrain==1:
            self.labels=df['label'].to_list()#[label]


    def __getitem__(self, idx):
        img=Image.open(self.img_path[idx])
        img=img.convert("RGB")
        img=np.array(img)
        img=cv2.resize(img,(224,224))#
        img = img / 255.
        img=np.transpose(img,(2,0,1))
        img = torch.tensor(img, dtype=torch.float32)

        text=self.text[idx]
        try:
            len(text)#部分文本是nan
        except:
            text=''


        text=self.tokenizer(text=text, add_special_tokens=True,
                  max_length=self.config.pad_size,  # 最大句子长度
                  padding='max_length',  # 补零到最大长度
                  truncation=True)
        #print(text)
        # 中文-英文  （t1[我 吃 饭],t2[i eat food]）  [[0,0,0,0,0],[1,1,1,1,1]]
        #text 三个部分  token_type_ids(句子对 中文句子 英文句子)
        input_id= torch.tensor(text['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(text['attention_mask'], dtype=torch.long)#可用可不用
        #

        if self.iftrain==1:
            label=int(self.labels[idx])
            label = torch.tensor(label, dtype=torch.long)
            return (img.to(self.config.device),input_id.to(self.config.device),attention_mask.to(self.config.device)),label.to(self.config.device)

        else:
            return (img.to(self.config.device),input_id.to(self.config.device),attention_mask.to(self.config.device))

    def __len__(self):
        return len(self.img_path)#总数据长度

def get_time_dif(start_time):

    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

if __name__=='__main__':
    config=Config()
    train_data=My_Dataset('./data/train.csv',config,1)
    train_iter = DataLoader(train_data, batch_size=32)
    n=0
    for a,b in train_iter:
        n=n+1

        print(n,b.shape)
        #print(y)
        print('************')