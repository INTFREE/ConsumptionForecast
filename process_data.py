import numpy as np
from collections import defaultdict
from sklearn import preprocessing

train_dir = './data/train/'
test_dir = './data/test/'
time = ['00', '04', '08', '12', '16', '20', '24']


def merge_features(agg_feature, **other_features):
    for other_feature in other_features.values():
        other_feature_keys = other_feature.keys()
        temp_feature_key = list(other_feature_keys)[0]
        temp_value = other_feature[temp_feature_key]
        dimension = len(temp_value)
        initial_value = [0 for i in range(dimension)]
        for key in agg_feature.keys():
            value = agg_feature[key]
            if key in other_feature_keys:
                value = np.append(value, other_feature[key])
            else:
                value = np.append(value, initial_value)
            agg_feature[key] = value
    return agg_feature


def deal_log_data(file):
    res = defaultdict(list)
    count = 0
    user_ids = set()
    with open(file, 'r') as f:
        lines = f.readlines()
        pre_id = lines[1].split('\t')[0]
        features = []
        for line in lines[1:]:
            paras = line.strip().split('\t')
            user_id = paras[0]
            user_ids.add(user_id)
            if user_id != pre_id:
                count += 1
                if pre_id in res.keys():
                    for feature in features:
                        res[pre_id].append(feature)
                else:
                    res[pre_id] = features
                pre_id = user_id
                features = []
            temp_feature = []
            labels = paras[1].split('-')
            for label in labels:
                temp_feature.append(label)
            temp_feature.append(paras[2].split()[0])
            temp_feature.append(paras[2].split()[1])
            temp_feature.append(paras[3])
            features.append(temp_feature)

        if pre_id in res.keys():
            for feature in features:
                res[pre_id].append(feature)
        else:
            res[pre_id] = features

        for key in res.keys():
            temp_res = np.array(res[key])
            res[key] = temp_res[np.argsort(temp_res[:, 4])]
            for temp_feature in res[key]:
                temp_time = temp_feature[4]
                day = int(temp_feature[3].split('-')[2])
                week_day = day%7
                temp_feature[3] = week_day
                hour = temp_time.split(':')[0]
                for i in range(0, len(time) - 1):
                    if hour >= time[i] and hour < time[i + 1]:
                        temp_feature[4] = i

        return res


def extract_time_interval_feature(file):
    res = deal_log_data(file)

    return_res = {}
    for key in res.keys():
        values = res[key]
        initial = [0 for i in range(6)]
        for value in values:
            initial[int(value[4])] += 1
        return_res[key] = initial
    return return_res

def extract_day_interval_feature(file):
    res = deal_log_data(file)

    return_res = {}
    for key in res.keys():
        values = res[key]
        initial = [0 for i in range(7)]
        for value in values:
            initial[int(value[3])] += 1
        return_res[key] = initial
    return return_res

def read_agg_data(file, log_file=None):
    res = defaultdict(list)
    temp_res = []
    temp_ids = []
    id_index = {}
    count = 0
    with open(file, 'r') as f:
        title = f.readline()
        for line in f.readlines():
            line = line.strip().split('\t')
            user_id = line[-1]
            data = line[:-1]
            if log_file:
                data.append(0)
            for i, item in enumerate(data):
                data[i] = float(item)
            temp_ids.append(user_id)
            temp_res.append(data)
            id_index[user_id] = count
            count += 1
    res_normal = preprocessing.normalize(temp_res)
    for i in range(0, len(temp_ids)):
        res[temp_ids[i]] = res_normal[i]
    return res


def read_log_data(file,feature_vocab):
    res = defaultdict(list)
    user_logs = {}
    vocab_size = len(feature_vocab)
    with open(file, 'r') as f:
        title = f.readline()
        for line in f.readlines():
            paras = line.strip().split('\t')
            user_id = paras[0]
            if not user_id in user_logs:
                user_logs[user_id] = defaultdict(int)
            labels = paras[1].split('-')
            for label in labels:
                user_logs[user_id][int(label)] += 1

        for user in user_logs:
            res[user] = np.zeros(vocab_size + 1)
            for feature in user_logs[user]:
                if feature in feature_vocab:
                    feature_index = feature_vocab[feature]
                    res[user][feature_index] = user_logs[user][feature]
                res[user][-1] += user_logs[user][feature]
    return res


def read_flg_data(file):
    res = defaultdict(int)
    count = 0
    with open(file, 'r') as f:
        title = f.readline()
        for line in f.readlines():
            line = line.strip().split('\t')
            user_id = line[0]
            flag = line[1]
            res[user_id] = int(flag)
            if flag == '1':
                count += 1
    print(count)
    return res


def build_log_vocab(logs):
    event_count = defaultdict(int)
    for user in logs:
        log = logs[user]
        for features in log:
            for feature in features[:3]:
                event_count[feature] += 1
    event_vocab = {'pad': 0, 'unk': 1}
    count = 2
    for event in event_count:
        if event_count[event] > 5:
            event_vocab[event] = count
            count += 1

    return event_vocab


def build_log_vocab_from_file(file):
    event_vocab = defaultdict(int)
    count = 0
    with open(file, 'r') as f:
        title = f.readline()
        for line in f.readlines():
            paras = line.strip().split('\t')
            labels = paras[1].split('-')
            for label in labels:
                label = int(label)
                if not label in event_vocab:
                    event_vocab[label] = count
                    count += 1
    return event_vocab


def extract_features(agg_file, log_file):
    res = defaultdict(dict)
    temp_res = []
    temp_ids = []
    id_index = {}
    count = 0
    with open(agg_file, 'r') as f:
        title = f.readline()
        for line in f.readlines():
            line = line.strip().split('\t')
            user_id = line[-1]
            data = line[:-1]
            if log_file:
                data.append(0)
            for i, item in enumerate(data):
                data[i] = float(item)
            temp_ids.append(user_id)
            temp_res.append(data)
            id_index[user_id] = count
            count += 1

    if log_file:
        with open(log_file, 'r') as f:
            title = f.readline()
            for line in f.readlines():
                paras = line.strip().split('\t')
                user_id = paras[0]
                index = id_index[user_id]
                temp_res[index][-1] += 1
    # res_normal = preprocessing.normalize(temp_res)
    for i in range(0, len(temp_ids)):
        res[temp_ids[i]] = res_normal[i]


if __name__ == '__main__':
    train_dir = './data/train/'
    test_dir = './data/test/'
    user_profile = []
    # vocab = build_log_vocab_from_file(train_dir+'train_log.csv')
    # print(vocab)
    extract_time_interval_feature('data/train/train_log.csv')
    # with open(train_dir + 'train_agg.csv', 'r') as f:
    #     lines = f.readlines()
    #     for line in lines[1:]:
    #         paras = line.strip().split('\t')
    #         temp_re = [int(paras[-1])]
    #         for para in paras[:-1]:
    #             temp_re.append(float(para))
    #         user_profile.append(temp_re)
    #         break
    # log_res = deal_log_data(train_dir + 'train_log.csv')
    # print(log_res['10002'])
    # lens = []
    # for item in log_res.values():
    #     lens.append(len(item))
    # plt.hist(lens)
    # plt.show()
    # agg_res = read_agg_data(test_dir + 'test_agg.csv')
    # print(agg_res.keys())
    # flg_res = read_flg_data(train_dir + 'train_flg.csv')
