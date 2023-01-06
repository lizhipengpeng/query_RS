import os
import math
import torch
import argparse
from transformers import AdamW
from model_py.model import PromptLearning
from data_load import config
from data_load.utils import now_time
from data_load.data_loader import BertRecommendDataset, bert_batch_preprocessing
from model_py.model import BertBasedModel


parser = argparse.ArgumentParser(description='SUG')
parser.add_argument('--data_path', type=str, default=config.train_file_src, help='path for loading the pickle data')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--epochs', type=int, default=10, help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=10, help='batch size')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log_interval', type=int, default=200, help='report interval')
parser.add_argument('--checkpoint', type=str, default='./pepler/', help='directory to save the final model')
parser.add_argument('--outf', type=str, default='generated.txt', help='output file for generated text')
parser.add_argument('--endure_times', type=int, default=5, help='the maximum endure times of loss increasing on validation')
parser.add_argument('--words', type=int, default=10,
                    help='number of words to generate for each sample')
args = parser.parse_args()

if args.data_path is None:
    parser.error('--data_path should be provided for loading data')

print('-' * 40 + 'ARGUMENTS' + '-' * 40)
for arg in vars(args):
    print('{:40} {}'.format(arg, getattr(args, arg)))
print('-' * 40 + 'ARGUMENTS' + '-' * 40)

if torch.cuda.is_available():
    if not args.cuda:
        print('WARNING: You have a CUDA device, so you should probably run with --cuda')
# device = torch.device('cuda' if args.cuda else 'cpu')
device = 'cpu'

if not os.path.exists(args.checkpoint):
    os.makedirs(args.checkpoint)
model_path = os.path.join(args.checkpoint, 'model.pt')
prediction_path = os.path.join(args.checkpoint, args.outf)

###############################################################################
# Load data
###############################################################################
train_dataset = BertRecommendDataset(file_src=config.train_file_src, is_training=True)
validation = BertRecommendDataset(file_src=config.valid_file_src, is_training=True)
test_data = BertRecommendDataset(file_src=config.test_file_src, is_training=True)


train_batches = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=bert_batch_preprocessing, drop_last=True)
valid_data_size = validation.__len__() // 10
validation_batches = torch.utils.data.DataLoader(dataset=validation, batch_size=valid_data_size, shuffle=False, collate_fn=bert_batch_preprocessing,drop_last=True)
test_batches = torch.utils.data.DataLoader(dataset=test_data, batch_size=valid_data_size, shuffle=False, collate_fn=bert_batch_preprocessing,drop_last=True)
###############################################################################
# Build the model
###############################################################################

model = PromptLearning.from_pretrained('./pre_trained_model/gpt/')
model.to(device)
optimizer = AdamW(model.parameters(), lr=args.lr)

###############################################################################
# Training code
###############################################################################


def train(data, bert_model):
    # Turn on training mode which enables dropout.
    model.train()
    text_loss = 0.
    total_sample = 0
    for batch in data:
        user, _, seq_ids, lens, masks, seqs = batch 
        user_result_raletion = bert_model(seq_ids, masks, user)[1]
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        optimizer.zero_grad()
        outputs = model(user_result_raletion, seq_ids, masks)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        batch_size = user.size(0)
        text_loss += batch_size * loss.item()
        total_sample += batch_size
    print("text_loss:{}, average_loss:{}".format(text_loss, text_loss/total_sample))

def evaluate(data, bert_model):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    text_loss = 0.
    total_sample = 0
    with torch.no_grad():
        for batch in data:
            user, _, seq_ids, lens, masks, seqs = batch 
            user_result_raletion = bert_model(seq_ids, masks, user)[1]
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            optimizer.zero_grad()
            outputs = model(user_result_raletion, seq_ids, masks)
            loss = outputs.loss
            batch_size = user.size(0)
            text_loss += batch_size * loss.item()
            total_sample += batch_size

    return text_loss / total_sample


def generate(data):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    idss_predict = []
    with torch.no_grad():
        while True:
            user, item, _, seq, _ = data.next_batch()  # data.step += 1
            user = user.to(device)  # (batch_size,)
            item = item.to(device)
            text = seq[:, :1].to(device)  # bos, (batch_size, 1)
            for idx in range(seq.size(1)):
                # produce a word at each step
                outputs = model(user, item, text, None)
                last_token = outputs.logits[:, -1, :]  # the last token, (batch_size, ntoken)
                word_prob = torch.softmax(last_token, dim=-1)
                token = torch.argmax(word_prob, dim=1, keepdim=True)  # (batch_size, 1), pick the one with the largest probability
                text = torch.cat([text, token], 1)  # (batch_size, len++)
            ids = text[:, 1:].tolist()  # remove bos, (batch_size, seq_len)
            idss_predict.extend(ids)

            if data.step == data.total_step:
                break
    return idss_predict


print('Tuning Prompt Only')
# Loop over epochs.
best_val_loss = float('inf')
endure_count = 0
bert_model = torch.load('./Result/models/model_135')
for epoch in range(1, args.epochs + 1):
    print(now_time() + 'epoch {}'.format(epoch))
    train(train_batches, bert_model)
    val_loss = evaluate(validation_batches, bert_model)
    print(now_time() + 'text ppl {:4.4f} | valid loss {:4.4f} on validation'.format(math.exp(val_loss), val_loss))
    # Save the model if the validation loss is the best we've seen so far.
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        with open(model_path, 'wb') as f:
            torch.save(model, f)
    else:
        endure_count += 1
        print(now_time() + 'Endured {} time(s)'.format(endure_count))
        if endure_count == args.endure_times:
            print(now_time() + 'Cannot endure it anymore | Exiting from early stop')
            break

# Load the best saved model.
with open(model_path, 'rb') as f:
    model = torch.load(f).to(device)


print(now_time() + 'Tuning both Prompt and LM')
for param in model.parameters():
    param.requires_grad = True
optimizer = AdamW(model.parameters(), lr=args.lr)

# Loop over epochs.
best_val_loss = float('inf')
endure_count = 0
for epoch in range(1, args.epochs + 1):
    print(now_time() + 'epoch {}'.format(epoch))
    train(train_batches, bert_model)
    val_loss = evaluate(validation_batches, bert_model)
    print(now_time() + 'text ppl {:4.4f} | valid loss {:4.4f} on validation'.format(math.exp(val_loss), val_loss))
    # Save the model if the validation loss is the best we've seen so far.
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        with open(model_path, 'wb') as f:
            torch.save(model, f)
    else:
        endure_count += 1
        print(now_time() + 'Endured {} time(s)'.format(endure_count))
        if endure_count == args.endure_times:
            print(now_time() + 'Cannot endure it anymore | Exiting from early stop')
            break

# Load the best saved model.
with open(model_path, 'rb') as f:
    model = torch.load(f).to(device)


# Run on test data.
test_loss = evaluate(test_batches)
print('=' * 89)
print(now_time() + 'text ppl {:4.4f} on test | End of training'.format(math.exp(test_loss)))
print(now_time() + 'Generating text')
idss_predicted = generate(test_batches)
tokens_test = [ids2tokens(ids[1:], tokenizer, eos) for ids in test_data.seq.tolist()]
tokens_predict = [ids2tokens(ids, tokenizer, eos) for ids in idss_predicted]
BLEU1 = bleu_score(tokens_test, tokens_predict, n_gram=1, smooth=False)
print(now_time() + 'BLEU-1 {:7.4f}'.format(BLEU1))
BLEU4 = bleu_score(tokens_test, tokens_predict, n_gram=4, smooth=False)
print(now_time() + 'BLEU-4 {:7.4f}'.format(BLEU4))
USR, USN = unique_sentence_percent(tokens_predict)
print(now_time() + 'USR {:7.4f} | USN {:7}'.format(USR, USN))
feature_batch = feature_detect(tokens_predict, feature_set)
DIV = feature_diversity(feature_batch)  # time-consuming
print(now_time() + 'DIV {:7.4f}'.format(DIV))
FCR = feature_coverage_ratio(feature_batch, feature_set)
print(now_time() + 'FCR {:7.4f}'.format(FCR))
FMR = feature_matching_ratio(feature_batch, test_data.feature)
print(now_time() + 'FMR {:7.4f}'.format(FMR))
text_test = [' '.join(tokens) for tokens in tokens_test]
text_predict = [' '.join(tokens) for tokens in tokens_predict]
ROUGE = rouge_score(text_test, text_predict)  # a dictionary
for (k, v) in ROUGE.items():
    print(now_time() + '{} {:7.4f}'.format(k, v))
text_out = ''
for (real, fake) in zip(text_test, text_predict):
    text_out += '{}\n{}\n\n'.format(real, fake)
with open(prediction_path, 'w', encoding='utf-8') as f:
    f.write(text_out)
print(now_time() + 'Generated text saved to ({})'.format(prediction_path))
