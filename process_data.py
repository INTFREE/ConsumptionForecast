import numpy as np
time = ['00', '04', '08', '12', '16', '20', '24']
def deal_log_data(train_file):
    res = {}
    count = 0
    user_ids = set()
    with open(train_file, 'r') as f:
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
            temp_feature.append(paras[2])
            temp_feature.append(paras[3])
            features.append(temp_feature)
        if pre_id in res.keys():
            for feature in features:
                res[pre_id].append(feature)
        else:
            res[pre_id] = features
        for key in res.keys():
            temp_res = np.array(res[key])
            res[key] = temp_res[np.argsort(temp_res[:,3])]
            for temp_feature in res[key]:
                temp_time = temp_feature[3]
                hour = temp_time.split(' ')[1].split(':')[0]
                for i in range(0, len(time) - 1):
                    if hour >= time[i] and hour < time[i + 1]:
                        temp_feature[3] = i
        return res

if __name__ == '__main__':
    train_dir = './data/train/'
    test_dir = './data/test/'
    user_profile = []
    # with open(train_dir + 'train_agg.csv', 'r') as f:
    #     lines = f.readlines()
    #     for line in lines[1:]:
    #         paras = line.strip().split('\t')
    #         temp_re = [int(paras[-1])]
    #         for para in paras[:-1]:
    #             temp_re.append(float(para))
    #         user_profile.append(temp_re)
    #         break
    deal_log_data(train_dir + 'train_log.csv')
