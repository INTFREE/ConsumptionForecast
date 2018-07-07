from process_data import *
import config
import time
from sklearn import metrics
from xgboost import XGBClassifier
import xgboost
from matplotlib import pyplot as plt

epochs = 5
folds = 5
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
    return train_data, valid_data


def evaluate(user_ids, dataloader, epoch, fold, is_test=False):
    total_loss = 0
    total_items = 0
    total_acc_count = 0
    model.eval()
    if is_test:
        output = open('output_' + str(epoch) + '_' + str(fold) + '.txt', 'w')
    pred_choice = np.zeros(len(dataloader) * rnn_config['batch_size'], int)
    label_choice = np.zeros(len(dataloader) * rnn_config['batch_size'], int)
    pred_prob = np.zeros(len(dataloader) * rnn_config['batch_size'], float)

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
                if pred[i][1].item() > threshold:
                    pred_choice[i + i_batch * rnn_config['batch_size']] = 1
                label_choice[i + i_batch * rnn_config['batch_size']] = label_seq[i].item()
                pred_prob[i + i_batch * rnn_config['batch_size']] = pred[i][1].item()
        if is_test:
            for i in range(num_items):
                output.write(user_ids[i + i_batch * rnn_config['batch_size']] + '\t' + str(float(pred[i][1])) + '\n')
        total_items += num_items
    if is_test:
        output.close()
        return 0

    cur_acc, cur_p, cur_r, cur_f1 = cal_f1(pred_choice, label_choice)

    auc = cal_auc(label_choice, pred_prob)
    print(
        'acc {:04.4f} | p {:04.4f} | r {:04.4f} | f1 {:04.4f}| auc {:04.4f}'.format(cur_acc, cur_p, cur_r, cur_f1, auc))
    return auc


if __name__ == '__main__':

    agg_res = read_agg_data(
        train_dir + 'train_agg.csv',
        train_dir + 'train_log.csv'
    )
    flg_res = read_flg_data(train_dir + 'train_flg.csv')
    test_agg = read_agg_data(
        test_dir + 'test_agg.csv',
        test_dir + 'test_log.csv'
    )
    print('load data end')
    user_ids = list(agg_res.keys())
    test_user_ids = list(test_agg.keys())

    for fold in range(folds):
        np.random.shuffle(user_ids)

        train_agg, valid_agg = split_data(user_ids, agg_res)
        train_flg, valid_flg = split_data(user_ids, flg_res)

        train_X = np.array([train_agg[key] for key in train_agg])
        train_Y = np.array([train_flg[key] for key in train_agg])

        valid_X = np.array([valid_agg[key] for key in valid_agg])
        valid_Y = np.array([valid_flg[key] for key in valid_agg])

        test_X = np.array([test_agg[key] for key in test_agg])

        print('train start')
        classifier = XGBClassifier(learning_rate=0.1,
                                   n_estimators=100,
                                   max_depth=5,
                                   min_child_weight=2,
                                   gamma=0,
                                   subsample=0.8,
                                   colsample_bytree=0.8,
                                   objective='binary:logistic',
                                   nthread=4,
                                   scale_pos_weight=1,
                                   seed=27)

        classifier.fit(train_X, train_Y)
        print('train end')

        train_prob = classifier.predict_proba(train_X)[:, 1]
        train_auc = cal_auc(train_Y, train_prob)

        valid_prob = classifier.predict_proba(valid_X)[:, 1]
        valid_auc = cal_auc(valid_Y, valid_prob)
        print('Train auc %f, Valid auc %f' % (train_auc, valid_auc))

        #xgboost.plot_tree(classifier, num_trees=0)
        test_prob = classifier.predict_proba(test_X)[:, 1]
        with open('xgboost_result/%d.txt' % fold, 'w') as output:
            for i, key in enumerate(test_agg):
                output.write(str(key) + '\t' + str(test_prob[i]) + '\n')

        #plt.savefig('800.png', dpi=800)
