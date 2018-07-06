from sklearn import svm
from sklearn import preprocessing
from collections import defaultdict
import numpy as np
from xgboost import XGBClassifier


def read_test_data():
    test_dir = './data/test/'
    train_res = defaultdict(list)
    temp_res = []
    temp_ids = []
    count = 0
    with open(test_dir + 'test_agg.csv', 'r') as f:
        title = f.readline()
        for line in f.readlines():
            line = line.strip().split('\t')
            user_id = line[-1]
            data = line[:-1]
            for i, item in enumerate(data):
                data[i] = float(item)
            temp_ids.append(user_id)
            temp_res.append(data)
            count += 1
    res_normal = preprocessing.normalize(temp_res)
    for i in range(0, len(temp_ids)):
        train_res[temp_ids[i]] = list(res_normal[i])
    return train_res


def read_agg_data():
    train_dir = './data/train/'
    train_res = defaultdict(list)
    temp_res = []
    temp_ids = []
    count = 0
    with open(train_dir + 'train_agg.csv', 'r') as f:
        title = f.readline()
        for line in f.readlines():
            line = line.strip().split('\t')
            user_id = line[-1]
            data = line[:-1]
            for i, item in enumerate(data):
                data[i] = float(item)
            temp_ids.append(user_id)
            temp_res.append(data)
            count += 1
    res_normal = preprocessing.normalize(temp_res)
    for i in range(0, len(temp_ids)):
        train_res[temp_ids[i]] = list(res_normal[i])

    with open(train_dir + 'train_flg.csv', 'r') as f:
        title = f.readline()
        for line in f.readlines():
            line = line.strip().split('\t')
            user_id = line[0]
            flag = int(line[1])
            train_res[user_id].append(flag)
    return train_res


if __name__ == '__main__':
    train_data = read_agg_data()
    test_data = read_test_data()
    print('read finish, train start')
    train_X = []
    train_Y = []
    test_X = []
    for key in train_data.keys():
        data = train_data[key]
        temp_data = data[:-1]
        train_X.append(temp_data)
        train_Y.append(data[-1])
    for key in test_data.keys():
        test_X.append(test_data[key])
    train_X = np.array(train_X)
    train_Y = np.array(train_Y)
    test_X = np.array(test_X)
    classifier = XGBClassifier(learning_rate=0.1,
         n_estimators=1000,
         max_depth=5,
         min_child_weight=1,
         gamma=0,
         subsample=0.8,
         colsample_bytree=0.8,
         objective= 'binary:logistic',
         nthread=4,
         scale_pos_weight=1,
         seed=27)
    classifier.fit(train_X, train_Y)
    result = classifier.predict_proba(test_X)

    # classifier = svm.SVC(kernel='rbf', class_weight={0: 1, 1: 24}, probability=True)
    # classifier.fit(train_X, train_Y)
    # result = classifier.score(test_X)
    print('train finish')
    print(result[0])
    ids = list(test_data.keys())
    with open('test_result.csv', 'w') as f:
        f.write('USERID' + '\t' + 'RST' + '\n')
        for i in range(0, len(ids)):
            f.write(str(ids[i]) + '\t' + str(result[i][1]).strip() + '\n')
    print(np.sum(train_Y))
