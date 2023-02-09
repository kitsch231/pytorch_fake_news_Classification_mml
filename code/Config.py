import os.path
import torch

#通过修改self.bert_name和self.resnet_name来决定模型的具体结构
class Config(object):
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        self.dropout = 0.3     # 随机失活
        self.require_improvement = 2000  # 若超过2000batch效果还没提升，则提前结束训练
        self.num_classes = 2  # 类别数无需修改
        self.num_epochs = 20   # epoch数
        self.batch_size =32   # mini-batch大小，看显存决定
        self.pad_size = 128  # 每句话处理成的长度(短填长切)
        self.bert_learning_rate = 1e-5   # bert的学习率，minirbt-h256需要用更大的学习率例如1e-4,其他bert模型设置为1e-5较好
        self.resnet_learning_rate = 2e-5 #resnet的学习率,最好比bert的学习率略高
        self.other_learning_rate = 2e-5#其他层的学习率
        self.frac=1#使用数据的比例，因为训练时间长，方便调参使用,1为全部数据，0.1代表十分之一的数据

        #['bert_model/bert-base-chinese','bert_model/chinese-bert-wwm-ext','bert_model/minirbt-h256']
        self.bert_name='bert_model/minirbt-h256'#bert类型，三种选择
        self.bert_fc=256 #bert全连接的输出维度 bert-base-chinese和chinese-bert-wwm-ext为768，minirbt-h256为256

        #['resnet18', 'resnet34', 'resnet50', 'resnet101','resnet152']
        self.resnet_name='resnet18'#resnet的种类，5种可选
        self.resnet_fc=self.bert_fc#resnet全连接的输出维度，跟bert需要保持一致
        self.usesloss=True#是否使用对比学习

        if not os.path.exists('model'):
            os.makedirs('model')
        if self.usesloss:
            self.save_path = 'model/'+'S_'+self.bert_name.replace('bert_model/','')+'_'+self.resnet_name+'.ckpt'#保存模型的路径
            self.log_dir= './log/'+'S_'+self.bert_name.replace('bert_model/','')+'_'+self.resnet_name#tensorboard日志的路径

        else:
            self.save_path = 'model/'+self.bert_name.replace('bert_model/','')+'_'+self.resnet_name+'.ckpt'#保存模型的路径
            self.log_dir= './log/'+self.bert_name.replace('bert_model/','')+'_'+self.resnet_name#tensorboard日志的路径

