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

epochs = 20
use_CUDA = 1
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

def split_data(user_ids, data):
    train_data = {}
    for user_id in user_ids[:60000]:
        train_data[user_id] = data[user_id]
    valid_data = {}
    for user_id in user_ids[60000:]:
        valid_data[user_id] = data[user_id]
    return train_data,valid_data


def train(model, dataloader):
    model.train()
    total_loss = 0
    total_items = 0
    total_acc_count = 0
    start_time = time.time()
    pred_choice = np.zeros(len(dataloader)*rnn_config['batch_size'],int)
    label_choice = np.zeros(len(dataloader)*rnn_config['batch_size'],int)
    for i_batch, batch in enumerate(dataloader):
        label_seq = Variable(batch['label'])
        del (batch['label'])
        for k in batch:
            batch[k] = Variable(batch[k])
        if use_CUDA:
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
    print('acc {:04.4f} | p {:04.4f} | r {:04.4f} | f1 {:04.4f} '.format(cur_acc, cur_p, cur_r, cur_f1))
    return model

def evaluate(model, dataloader, epoch=0, is_test=False):
    total_loss = 0
    total_items = 0

    total_acc_count = 0
    model.eval()
    output = open('test_final_output_' + str(epoch) + '.txt', 'w')
    pred_choice = np.zeros(len(dataloader) * rnn_config['batch_size'], int)
    label_choice = np.zeros(len(dataloader) * rnn_config['batch_size'], int)

    for i_batch, batch in enumerate(dataloader):
        if not is_test:
            label_seq = Variable(batch['label'])
            del (batch['label'])
        for k in batch:
            batch[k] = Variable(batch[k])
        if use_CUDA:
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

        for i in range(num_items):
            output.write(str(float(pred[i][1])) + '\r\n')
        total_items += num_items
    output.close()
    if is_test:
        return 0

    cur_acc, cur_p, cur_r, cur_f1 = cal_f1(pred_choice, label_choice)
    print('acc {:04.4f} | p {:04.4f} | r {:04.4f} | f1 {:04.4f} '.format(cur_acc, cur_p, cur_r, cur_f1))
    return cur_f1


if __name__ == '__main__':

    log_res = deal_log_data(train_dir + 'train_log.csv')
    agg_res = read_agg_data(train_dir + 'train_agg.csv')
    flg_res = read_flg_data(train_dir + 'train_flg.csv')
    print('load data end')
    user_ids = list(agg_res.keys())
    np.random.shuffle(user_ids)
    train_log, valid_log = split_data(user_ids,log_res)
    train_agg, valid_agg = split_data(user_ids,agg_res)
    train_flg, valid_flg = split_data(user_ids,flg_res)
    event_vocab = build_log_vocab(train_log)
    event_vocab_size =len(event_vocab)
    rnn_config = config.rnn_config
    rnn_config['event_vocab_size'] = event_vocab_size
    print('build vocab end')
    train_dataset = MyDataSet(train_agg, train_log, event_vocab, train_flg)
    valid_dataset = MyDataSet(valid_agg, valid_log, event_vocab, valid_flg)
    train_loader = DataLoader(dataset=train_dataset, batch_size=rnn_config['batch_size'])
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=rnn_config['batch_size'])
    rnn_model = MyModel(rnn_config)
    rnn_model.init_weights()

    rnn_model = nn.DataParallel(rnn_model)
    if use_CUDA:
        rnn_model = rnn_model.cuda()

    optimizer = torch.optim.Adam(rnn_model.parameters())
    criteria = nn.CrossEntropyLoss()

    best_val_acc = 0

    try:
        print(rnn_model)

        for epoch in range(1, epochs + 1):
            # scheduler.step()
            epoch_start_time = time.time()
            rnn_model = train(rnn_model, train_loader)
            val_acc = evaluate(rnn_model,valid_loader, epoch, False)
            print('-' * 89)
            print( '| end of epoch {:3d} | time: {:5.2f}s | valid acc {:5.6f}'.format(epoch,
                                                                                    (time.time() - epoch_start_time),
                                                                                    val_acc))
            print('-' * 89)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_acc or val_acc > best_val_acc:
                print('new best val loss, saving model')
                with open('model.pkl', 'wb') as f:
                    torch.save(rnn_model, f)
                best_val_acc = val_acc
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                pass
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

