import os
import time
import torch
from transformers.optimization import AdamW
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from model_py.model import BertBasedModel
from data_load.data_loader import BertRecommendDataset, bert_batch_preprocessing
from data_load import utils, config
from sklearn.metrics import accuracy_score

torch.manual_seed(123)
torch.cuda.manual_seed_all(123)


class Train(object):

    def __init__(self, user_feature_input_size):

        self.model_dir = os.path.join(config.model_dir, 'models')
        self.summary_dir = os.path.join(config.model_dir, 'summary')
        for path in [config.model_dir, self.model_dir, self.summary_dir]:
            if not os.path.exists(path):
                os.mkdir(path)

        # training
        self.max_epoches = config.max_epoches
        self.eval_every = config.eval_every
        self.batch_size = 5
        self.summary_flush_every = config.summary_flush_every
        self.report_every = config.report_every

        # model
        self.bert_hsz= config.bert_hsz
        self.classifier_hsz = config.classifier_hsz
        self.categories = config.categories
        self.user_feature_input_size = user_feature_input_size

    def set_train(self, model_file_path=None):

        self.model = BertBasedModel(self.user_feature_input_size, hidden_size=self.bert_hsz, classifier_hsz=self.classifier_hsz, categories=self.categories)# .cuda()

        self.summary_writer = SummaryWriter(log_dir=self.summary_dir)
        bert_parameters = list(self.model.bert.parameters())
        bert_named_parameters = list(self.model.bert.named_parameters())
        classifier_parameters = list(self.model.classifier.parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        optimizer_grouped_parameters = [
            {'params': [p for n, p in bert_named_parameters if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01,
             'lr': config.bert_lr},
            {'params': [p for n, p in bert_named_parameters if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
             'lr': config.bert_lr},
            {'params': classifier_parameters}]

        self.optimizer = AdamW(params=optimizer_grouped_parameters, lr=config.lr)

    def save_model(self, loss, itr):
        state = {
            "itrs": itr,
            'bert_state_dict': self.model.bert.state_dict(),
            'classifier_state_dict': self.model.classifier.state_dict(),
            'loss': loss
        }

        model_saved_path = os.path.join(self.model_dir, 'model_{}'.format(itr))
        # torch.save(state, model_saved_path)
        torch.save(self.model, model_saved_path)

    def train_one_batch(self, user, labels, seq_ids, masks):

        probs = self.model(seq_ids, masks, user)[0]
        loss_fun = torch.nn.CrossEntropyLoss()
        loss = loss_fun(probs, labels)

        return loss, probs

    def train_itrs(self, train_file_src, valid_file_src):
        report_print = "At: epoch:{}: {}, loss: {}, total_time: {}"
        eval_report_print = 'At: epoch{}: {}, the eval result is {}, best result is {}, at {}: {}'

        train_dataset = BertRecommendDataset(file_src=train_file_src, is_training=True)
        validation = BertRecommendDataset(file_src=valid_file_src, is_training=True)

        valid_data_size = validation.__len__() // 10

        self.set_train()

        best_eval_result = 0.0
        best_result_itr = 0
        best_epoch = 0

        total_times = time.time()
        itrs = 0

        train_batches = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=bert_batch_preprocessing, drop_last=True)

        validation_batches = torch.utils.data.DataLoader(dataset=validation,
                                                         batch_size=valid_data_size, shuffle=False,
                                                         collate_fn=bert_batch_preprocessing,drop_last=True)

        loss = 0
        for epoch in range(self.max_epoches):

            for batch in train_batches:
                user, labels, seq_ids, lens, masks, seqs = batch
                # user = user.cuda()
                # labels = labels.cuda()
                # seq_ids = seq_ids.cuda()
                # masks = masks.cuda()

                self.optimizer.zero_grad()

                loss, probs = self.train_one_batch(user, labels, seq_ids, masks)

                loss.backward()

                self.optimizer.step()

                itrs += 1
                if itrs % self.summary_flush_every == 0:
                    self.summary_writer.add_scalar(tag='train/loss',
                                                   scalar_value=loss.item(), global_step=itrs)
                    self.summary_writer.flush()

                if itrs % self.report_every == 0:
                    print(report_print.format(epoch, itrs, loss, utils.get_time(total_times)))

            if epoch % self.eval_every == 0:
                self.model.eval()

                y_probs = []
                ys = []
                for valid_batch in validation_batches:
                    user, labels, seq_ids, lens, masks, seqs = valid_batch

                    # user = user.cuda()
                    # labels = labels.cuda()
                    # seq_ids = seq_ids.cuda()
                    # masks = masks.cuda()

                    loss, probs = self.train_one_batch(user, labels, seq_ids, masks)

                    labels = labels.cpu().numpy().astype(int).tolist()
                    probs = probs.detach().numpy()
                    probs = np.around(probs,0).astype(int)
                    y_probs += probs.tolist()
                    ys += labels
                
                def assit(fun, ys, y_probs):
                    list_fun = [fun(i, j) for i in ys for j in y_probs]
                    return sum(list_fun)/len(list_fun)


                accuracy = assit(accuracy_score, ys, y_probs)
                # precision = assit(precision_score, ys, y_probs)
                # f1 = assit(f1_score, ys, y_probs)
                # auc_score = assit(roc_auc_score, ys, y_probs)

                if best_eval_result < accuracy:
                    best_eval_result = accuracy
                    best_result_itr = itrs
                    best_epoch = epoch
                    self.save_model(loss, itrs)
                self.summary_writer.add_scalar(tag='train/accuracy',
                                               scalar_value=accuracy, global_step=itrs)

                print(eval_report_print.format(epoch, itrs, [accuracy], best_eval_result, best_epoch, best_result_itr))

                self.model.train()


if __name__ == '__main__':
    train = Train(user_feature_input_size=config.user_feature_input_size)
    train.train_itrs(train_file_src=config.train_file_src, valid_file_src=config.valid_file_src)
