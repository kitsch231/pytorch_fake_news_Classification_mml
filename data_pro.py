import pandas as pd
import os
import csv
import numpy as np
#原始数据重新整理
imgs=os.listdir('./data/images')
# print(imgs)
def new_data(path,label,newpath):
    len_list=[]
    with open(path,'r',encoding='utf-8')as t1:
        t1=t1.readlines()

        if len(t1)%3==0:
            num=int(len(t1)/3)
            print('数据列数:',len(t1))
            print('数据条数(除以3):', num)
            for n in range(num):
                #a2为图片名，a3为文本
                a1,a2,a3=t1[n*3],t1[n*3+1],t1[n*3+2]
                a2=a2.strip()#取出换行符
                a3=a3.strip()
                text_len=len(a3)
                len_list.append(text_len)
                a2=a2.split('|')#分割图片
                a2=[x.split('/')[-1] for x in a2 if x!='null']#去除空数据并分割出图片路径

                a2 = ['./data/images/' + x for x in a2 if x in imgs]
                for m in a2:
                    all_info=m,a3,label
                    # print(all_info)
                    with open(newpath,'a',encoding='utf-8',newline='')as f:
                        writer=csv.writer(f)
                        writer.writerow(all_info)

        else:
            print('数据长度不合理')
    print('平均句子长度:',np.mean(len_list))

if os.path.exists('./data/train.csv'):
    os.remove('./data/train.csv')#如果存在就删除以免重复写入
with open('./data/train.csv', 'a', encoding='utf-8', newline='') as f:#写入列名
    writer = csv.writer(f)
    writer.writerow(('path','text','label'))
new_data('./data/tweets/train_rumor.txt',1,'./data/train.csv')#训练谣言数据，1表示给谣言数据添加标签1
new_data('./data/tweets/train_nonrumor.txt',0,'./data/train.csv')#训练非谣言数据，0表示给谣言数据添加标签0


if os.path.exists('./data/test.csv'):
    os.remove('./data/test.csv')
with open('./data/test.csv', 'a', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(('path', 'text', 'label'))
new_data('./data/tweets/test_rumor.txt',1,'./data/test.csv')#测试谣言数据
new_data('./data/tweets/test_nonrumor.txt',0,'./data/test.csv')#测试非谣言数据

df=pd.read_csv('./data/train.csv',encoding='utf-8')
val_df=df.sample(frac=0.1)#划分验证集
train_df=df.drop(index=val_df.index.to_list())#划分训练集
print('训练集长度:',len(train_df))
print('测试集长度:',len(val_df))
val_df.to_csv('./data/val.csv',encoding='utf-8',index=None)
train_df.to_csv('./data/train.csv',encoding='utf-8',index=None)