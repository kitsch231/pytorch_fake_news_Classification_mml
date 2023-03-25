# pytorch_bert_resnet_mml
使用pytorch完成的一个多模态虚假新闻分类任务，文本和图像部分分别使用了bert和resnet提取特征（在config里可以组合多种模型）,在weibo谣言数据集上取得了良好的性能（测试集acc89%左右）。code文件夹下是引入了对比学习的版本，测试最终使得模型的准确率提高了一个百分点。


torch1.10以上版本最好，在Config.py种修改训练参数后运行train.py即可，模型保存在model文件夹下，
对应的tensorboard保存在log文件夹下，最终输入命令即可查看，如果有多个事件则查看最后一个
如tensorboard --logdir=log/minirbt-h256_resnet18 --port=6006

采用的三个bert模型请自行从huggfacing git到本地目录，如果需要数据集进行演示请与我联系

数据集下载：https://drive.google.com/drive/folders/1SYHLEMwC8kCibzUYjOqiLUeFmP2a19CH?usp=sharing
