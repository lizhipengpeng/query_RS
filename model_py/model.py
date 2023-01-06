import torch
from torch import nn
from transformers import BertModel, GPT2LMHeadModel



class MLP(nn.Module):
    def __init__(self, mlp_num_input, mlp_num_hiddens, mlp_num_outputs,
                 **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.dense1 = nn.Linear(mlp_num_input, mlp_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(mlp_num_hiddens, mlp_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))

class BertClassifier(nn.Module):
    def __init__(self, user_feature_input_size, hidden_size, classifier_hsz, categories):
        super(BertClassifier, self).__init__()

        self.linear = nn.Linear(in_features=hidden_size+user_feature_input_size, out_features=classifier_hsz, bias=True)
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0.0)

        self.batch_normalization = nn.BatchNorm1d(num_features=classifier_hsz)

        self.classifier = nn.Linear(in_features=classifier_hsz, out_features=categories, bias=True)
        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0.0)

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, bert_pre, user_fea):

        features = torch.concat((bert_pre, user_fea), 1)
        features = self.dropout(features)

        dense_features = self.linear(features)
        dense_features = torch.relu(dense_features)
        normalized = self.batch_normalization(dense_features)

        probs = self.classifier(normalized)
        probs = torch.softmax(probs, dim=-1)

        return probs, dense_features


class BertBasedModel(nn.Module):

    def __init__(self, user_feature_input_size, hidden_size, classifier_hsz, categories, model_file_src=None):
        super(BertBasedModel, self).__init__()
        self.bert = BertModel.from_pretrained('./pre_trained_model/bert/')

        self.classifier = BertClassifier(user_feature_input_size, hidden_size=hidden_size, classifier_hsz=classifier_hsz, categories=categories)

        if model_file_src is not None:
            state = torch.load(model_file_src, map_location=lambda storage, location: storage)
            self.bert.load_state_dict(state_dict=state['bert_state_dict'])
            self.classifier.load_state_dict(state_dict=state['classifier_state_dict'])
            print("Loading model {} successfully".format(model_file_src))

    def forward(self, seq_ids, masks, user_fea):

        res = self.bert(seq_ids, attention_mask=masks)
        last_hidden_state, pooler_output = res['last_hidden_state'], res['pooler_output']
        pre, mixed_fea = self.classifier(pooler_output, user_fea)
        return [pre, mixed_fea]

    def _get_mixed_fea(self):

        return self.mixed_fea



class Prompt:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, freezeLM=True, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        # freeze pretrained model parameters
        if freezeLM:
            for param in model.parameters():
                param.requires_grad = False

        return model

    def forward(self, user_result_raletion, text, mask, ignore_index=-100):
        self.src_len = 1
        device = user_result_raletion.device
        batch_size = user_result_raletion.size(0)

        # embeddings
        w_src = self.transformer.wte(text)  # (batch_size, tgt_len, emsize)
        src = torch.cat([user_result_raletion.unsqueeze(1), w_src], 1)  # (batch_size, total_len, emsize)

        if mask is None:
            # auto-regressive generation
            return super().forward(inputs_embeds=src)
        else:
            # training
            # input padding
            pad_left = torch.ones((batch_size, self.src_len), dtype=torch.int64).to(device)
            pad_input = torch.cat([pad_left, mask], 1)  # (batch_size, total_len)

            # prediction for training
            pred_left = torch.full((batch_size, self.src_len), ignore_index, dtype=torch.int64).to(device)  # (batch_size, src_len)
            pred_right = torch.where(mask == 1, text, torch.tensor(ignore_index).to(device))  # replace <pad> with ignore_index
            prediction = torch.cat([pred_left, pred_right], 1)  # (batch_size, total_len)

            return super().forward(attention_mask=pad_input, inputs_embeds=src, labels=prediction)

class PromptLearning(Prompt, GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)

        
