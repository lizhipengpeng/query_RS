


# preprocess
unprocessed_train_file = "../dataset/train_shuffle"
filtered_file = "../dataset/preprocessed_data/filtered_train.txt"
data_divided_ratio = 0.95

# bert based model
pretrained_model_name = 'bert-base-chinese'
bert_hsz = 768
classifier_hsz = 768
categories = 4
user_feature_input_size = 256

# train
train_file_src = "./data_load/train_data.csv"
valid_file_src = "./data_load/valid_data.csv"
max_epoches = 50
eval_every = 1
model_dir = './Result'
using_pre_trained_emb = False
batch_size = 128
bert_lr = 2e-5
lr = 0.01
summary_flush_every = 10
report_every = 10

# test
test_file_src = './data_load/test_data.csv'

