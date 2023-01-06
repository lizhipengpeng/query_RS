import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer
import numpy as np
import csv
import pandas as pd
from data_load import config
from d2l import torch as d2l


class BertRecommendDataset(Dataset):
    def __init__(self, file_src, is_training):
        sequence_list = []
        user_list = []
        label_list = []
        with open(file_src, 'r', encoding='utf-8') as f:
            reader = list(csv.reader(f))
            head = reader[0]
            reader = reader[1:]
            for line in reader:
                if is_training:
                    # user: 'cityid', 'page_city_id', 'locate_city_id', 'age', 'edu_level', 'comsume_style'
                    # lable: 'clickcnt', 'landpage_clickcnt', 'landpage_ordercnt', 'landpage_paycnt'
                    # sequence: 'query'+'item_name'
                    user, lable, sequence = line[3][:-2], line[11:15], line[7:11]
                    user_list.append(user)
                    sequence_list.append([sequence[0],sequence[3]])
                    label_list.append([eval(i) for i in lable])
                else:
                    user, sequence = line[1]+line[4:7]+line[16:], line[7:11]
                    user_list.append(user)
                    sequence_list.append([sequence[0],sequence[3]])
        print("The size of dataset {} is: {}".format(file_src, len(sequence_list)))

        self.vocab = d2l.Vocab(user_list)
        corpus = [self.vocab[user] for user in user_list]
        self.user_list = torch.tensor(corpus, dtype=torch.int64).contiguous()
        self.user_embedding = nn.Embedding(len(corpus), config.user_feature_input_size)
        initrange = 0.1
        self.user_embedding.weight.data.uniform_(-initrange, initrange)
        self.sequence_list = sequence_list
        self.label_list = label_list
        self.tokenizer = BertTokenizer('./pre_trained_model/vocab.txt')
        self.is_training = is_training
    
    def __getitem__(self, item):

        sequence = '[CLS]' + self.sequence_list[item][0] + '[SEP]' + self.sequence_list[item][1]
        # sequence = " ".join(sequence)
        # print(type(sequence))
        sequence = self.tokenizer.tokenize(sequence)
        seq_ids = self.tokenizer.convert_tokens_to_ids(sequence)
        user = self.user_embedding(self.user_list[item])


        if self.is_training:
            # print(type(seq_ids[0]))
            return user, self.label_list[item], \
                   torch.tensor(seq_ids).long(), \
                   len(seq_ids), self.sequence_list[item]
        else:
            return user, None, torch.tensor(seq_ids).long(), len(seq_ids), self.sequence_list[item]
    
    def __len__(self) -> int:
        return len(self.label_list) - 1

def bert_batch_preprocessing(batch):
    user, labels, seq_ids, lens, seqs = zip(*batch)
    seq_ids = pad_sequence(seq_ids, batch_first=True, padding_value=0)

    bsz, max_len = seq_ids.size()
    masks = np.zeros([bsz, max_len], dtype=np.float)

    for index, seq_len in enumerate(lens):
        masks[index][:seq_len] = 1

    masks = torch.from_numpy(masks)

    if labels[0] is not None:
        labels = torch.tensor(labels)
    lens = torch.tensor(lens)


    return torch.stack(user), labels, seq_ids, lens, masks, seqs


def split_train_valid_test(path):
    data = pd.read_csv(path)
    train_data = data[data['dt'] < 20221123]
    valid_data = data[data['dt'] == 20221123]
    test_data = data[data['dt'] == 20221124]

    train_data.to_csv("./data_load/train_data.csv", index=False)
    valid_data.to_csv("./data_load/valid_data.csv", index=False)
    test_data.to_csv("./data_load/test_data.csv", index=False)


# train_dataset = BertRecommendDataset(file_src="./data_load/train_data.csv", is_training=True)
# train_batches = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=5,  shuffle=True, collate_fn=bert_batch_preprocessing)

# for i in train_batches:
#     print(i)