from data_load.data_loader import BertRecommendDataset as Data
from data_load.data_loader import bert_batch_preprocessing
import torch
from model_py.model import PromptLearning

# 本文件主要设计整个实验的大体代码架构，具体的实现函数放在其他文件中实现

# 数据读取
# 数据分为train, test 两类
def DataLoader(path, ):

    train_data = Data(file_src=path + "train.csv", is_training=True)
    test_data = Data(file_src=path + "test.csv", is_training=False)
 
    return train_data, test_data

# 模型加载
# 模型分为两种：一、没有训练过的模型；二、训练过的模型

def GetModel(path, split, ):
    
    if split == True:
        model = PromptLearning.from_pretrained(path + "/trained/")
    else:
        model = PromptLearning.from_pretrained(path + "/untrained/")
    
    return model


# 训练
# 使用不同模板进行训练，分别得到相应的指标结果
def Train(data, model, template):

    # train_batches = torch.utils.data.DataLoader(dataset=train_data, batch_size=5,  shuffle=True, collate_fn=bert_batch_preprocessing)
    # test_batches = torch.utils.data.DataLoader(dataset=test_data, batch_size=5,  shuffle=True, collate_fn=bert_batch_preprocessing)

    pass

# baseline模型
# 使用不同的baseline模型分别得到指标结果
def BaselineModel(data, baseline_model, ):
    pass