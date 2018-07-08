from process_data import *
from sklearn import metrics
from xgboost import XGBClassifier
from sklearn.grid_search import GridSearchCV


folds = 1


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
                                   max_depth=3,
                                   min_child_weight=1,
                                   gamma=0.5,
                                   subsample=0.8,
                                   colsample_bytree=0.8,
                                   objective='binary:logistic',
                                   nthread=4,
                                   scale_pos_weight=1,
                                   seed=27)
        tuned_parameters = {'max_depth':[3,4,5,6,7,8]}
        clf = GridSearchCV(classifier, param_grid=tuned_parameters, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
        clf.fit(train_X, train_Y)
        print('best paras')
        print(clf.best_params_)
        #classifier.fit(train_X, train_Y)
        print('train end')

        train_prob = clf.predict_proba(train_X)[:, 1]
        train_auc = cal_auc(train_Y, train_prob)

        valid_prob = clf.predict_proba(valid_X)[:, 1]
        valid_auc = cal_auc(valid_Y, valid_prob)
        print('Train auc %f, Valid auc %f' % (train_auc, valid_auc))

        #xgboost.plot_tree(classifier, num_trees=0)
        test_prob = clf.predict_proba(test_X)[:, 1]
        with open('xgboost_result/%d.txt' % fold, 'w') as output:
            for i, key in enumerate(test_agg):
                output.write(str(key) + '\t' + str(test_prob[i]) + '\n')
        break
        #plt.savefig('800.png', dpi=800)
