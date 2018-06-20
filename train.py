import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import  DataLoader

from torch.autograd import Variable
from process_data import *
import config
from utils.dataset import MyDataSet
from model.RNN import MyModel
import time
from sklearn import metrics

epochs = 5
folds = 10
rnn_config = config.rnn_config
log_interval = 100
threshold = 0.1

def cal_f1(pred, label):
    # TP    predict 和 label 同时为1
    TP = ((pred == 1) & (label == 1)).sum()
    # TN    predict 和 label 同时为0
    TN = ((pred == 0) & (label == 0)).sum()
    # FN    predict 0 label 1
    FN = ((pred == 0) & (label == 1)).sum()
    # FP    predict 1 label 0
    FP = ((pred == 1) & (label == 0)).sum()

    p = TP / (TP + FP)
    r = TP / (TP + FN)
    F1 = 2 * r * p / (r + p)
    acc = (TP + TN) / (TP + TN + FP + FN)
    return acc, p, r, F1

def cal_auc(label, pred_prob):
    auc = metrics.roc_auc_score(label, pred_prob)
    return auc

def split_data(user_ids, data):
    train_data = {}
    for user_id in user_ids[:70000]:
        train_data[user_id] = data[user_id]
    valid_data = {}
    for user_id in user_ids[70000:]:
        valid_data[user_id] = data[user_id]
    return train_data,valid_data


def train(dataloader):
    model.train()
    total_loss = 0
    total_items = 0
    total_acc_count = 0
    start_time = time.time()
    pred_choice = np.zeros(len(dataloader)*rnn_config['batch_size'],int)
    label_choice = np.zeros(len(dataloader)*rnn_config['batch_size'],int)
    pred_prob = np.zeros(len(dataloader)*rnn_config['batch_size'],float)
    for i_batch, batch in enumerate(dataloader):
        label_seq = Variable(batch['label'])
        del (batch['label'])
        for k in batch:
            batch[k] = Variable(batch[k])
        if rnn_config['use_CUDA']:
            label_seq = label_seq.cuda()
            for k in batch:
                batch[k] = batch[k].cuda()
        model.zero_grad()
        pred = model.forward(**batch)
        pred = pred.view(-1, pred.size(-1))
        label_seq = label_seq.view(-1)
        loss = criteria(pred, label_seq)
        loss.backward()

        pred = F.softmax(pred, dim=-1)
        for i in range(len(label_seq)):
            if pred[i][1].item()>threshold:
                pred_choice[i+i_batch*rnn_config['batch_size']] = 1
            label_choice[i+i_batch*rnn_config['batch_size']] = label_seq[i].item()
            pred_prob[i+i_batch*rnn_config['batch_size']] = pred[i][1].item()

        num_items = pred.size(0)
        total_loss += num_items * loss.data
        total_items += num_items
        optimizer.step()

        if i_batch % log_interval == 0 and i_batch > 0:

            cur_loss = total_loss[0] / total_items

            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:04.4f} | ms/batch {:5.2f} | '
                  'loss {:5.6f}| '.format(
                epoch, i_batch, len(dataloader.dataset) // dataloader.batch_size, optimizer.param_groups[0]['lr'],
                                elapsed * 1000 / log_interval, cur_loss))
            total_loss = 0
            total_items = 0
            total_acc_count = 0
            start_time = time.time()

    cur_acc, cur_p, cur_r, cur_f1 = cal_f1(pred_choice, label_choice)
    auc = cal_auc(label_choice, pred_prob)
    print('acc {:04.4f} | p {:04.4f} | r {:04.4f} | f1 {:04.4f}| auc {:04.4f}'.format(cur_acc, cur_p, cur_r, cur_f1, auc))


def evaluate( user_ids, dataloader, epoch, fold, is_test=False):
    total_loss = 0
    total_items = 0
    total_acc_count = 0
    model.eval()
    if is_test:
        output = open('output_' + str(epoch)+ '_' + str(fold) + '.txt', 'w')
    pred_choice = np.zeros(len(dataloader) * rnn_config['batch_size'], int)
    label_choice = np.zeros(len(dataloader) * rnn_config['batch_size'], int)
    pred_prob = np.zeros(len(dataloader)*rnn_config['batch_size'],float)

    for i_batch, batch in enumerate(dataloader):
        if not is_test:
            label_seq = Variable(batch['label'])
            del (batch['label'])
        for k in batch:
            batch[k] = Variable(batch[k])
        if rnn_config['use_CUDA']:
            for k in batch:
                batch[k] = batch[k].cuda()
        pred = model.forward(**batch)

        num_items = pred.size(0)
        pred = F.softmax(pred, dim=-1)

        if not is_test:
            for i in range(len(label_seq)):
                if pred[i][1].item()>threshold:
                    pred_choice[i+i_batch*rnn_config['batch_size']] = 1
                label_choice[i+i_batch*rnn_config['batch_size']] = label_seq[i].item()
                pred_prob[i + i_batch * rnn_config['batch_size']] = pred[i][1].item()
        if is_test:
            for i in range(num_items):
                output.write(user_ids[i+i_batch*rnn_config['batch_size']]+'\t'+str(float(pred[i][1])) + '\n')
        total_items += num_items
    if is_test:
        output.close()
        return 0

    cur_acc, cur_p, cur_r, cur_f1 = cal_f1(pred_choice, label_choice)

    auc = cal_auc(label_choice, pred_prob)
    print('acc {:04.4f} | p {:04.4f} | r {:04.4f} | f1 {:04.4f}| auc {:04.4f}'.format(cur_acc, cur_p, cur_r, cur_f1, auc))
    return auc


if __name__ == '__main__':

    log_res = deal_log_data(train_dir + 'train_log.csv')
    agg_res = read_agg_data(train_dir + 'train_agg.csv')
    flg_res = read_flg_data(train_dir + 'train_flg.csv')
    print('load data end')

    test_log = deal_log_data(test_dir + 'test_log.csv')
    test_agg = read_agg_data(test_dir + 'test_agg.csv')

    user_ids = list(agg_res.keys())
    test_user_ids = list(test_agg.keys())

    for fold in range(folds):

        np.random.shuffle(user_ids)
        train_log, valid_log = split_data(user_ids,log_res)
        train_agg, valid_agg = split_data(user_ids,agg_res)
        train_flg, valid_flg = split_data(user_ids,flg_res)
        event_vocab = build_log_vocab(train_log)
        event_vocab_size =len(event_vocab)

        rnn_config['event_vocab_size'] = event_vocab_size
        print('build vocab end')
        train_dataset = MyDataSet(train_agg, train_log, event_vocab, train_flg)
        valid_dataset = MyDataSet(valid_agg, valid_log, event_vocab, valid_flg)
        test_dataset = MyDataSet(test_agg, test_log, event_vocab)

        train_loader = DataLoader(dataset=train_dataset, batch_size=rnn_config['batch_size'], shuffle=True)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=rnn_config['batch_size'], shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=rnn_config['batch_size'])

        model = MyModel(rnn_config)
        model.init_weights()

        model = nn.DataParallel(model)
        if rnn_config['use_CUDA']:
            model = model.cuda()

        optimizer = torch.optim.Adam(model.parameters())
        criteria = nn.CrossEntropyLoss()

        best_val_acc = 0

        try:
            print(model)

            for epoch in range(1, epochs + 1):
                # scheduler.step()
                epoch_start_time = time.time()
                train(train_loader)
                val_acc = evaluate(user_ids[:70000], valid_loader, epoch, fold, False)
                print('-' * 89)
                print( '| end of epoch {:3d} | time: {:5.2f}s | valid acc {:5.6f}'.format(epoch,
                                                                                        (time.time() - epoch_start_time),
                                                                                        val_acc))
                print('-' * 89)

                _ = evaluate(test_user_ids, test_loader, epoch, fold, True)
                # Save the model if the validation loss is the best we've seen so far.
                if not best_val_acc or val_acc > best_val_acc:
                    print('new best val loss, saving model')
                    with open('model.pkl', 'wb') as f:
                        torch.save(model, f)
                    best_val_acc = val_acc
                else:
                    # Anneal the learning rate if no improvement has been seen in the validation dataset.
                    pass
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')

