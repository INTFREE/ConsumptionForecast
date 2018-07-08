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

    train_agg = read_agg_data(
        train_dir + 'train_agg.csv',
        train_dir + 'train_log.csv'
    )
    train_flg = read_flg_data(train_dir + 'train_flg.csv')
    test_agg = read_agg_data(
        test_dir + 'test_agg.csv',
        test_dir + 'test_log.csv'
    )
    print('load data end')
    user_ids = list(train_agg.keys())
    test_user_ids = list(test_agg.keys())



    #    train_agg, valid_agg = split_data(user_ids, agg_res)
    #    train_flg, valid_flg = split_data(user_ids, flg_res)

    train_X = np.array([train_agg[key] for key in train_agg])
    train_Y = np.array([train_flg[key] for key in train_agg])

    test_X = np.array([test_agg[key] for key in test_agg])

    print('train start')
    cv_params = {'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800]}
    other_params = {'learning_rate': 0.1, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}

    classifier = XGBClassifier(**other_params)

    clf = GridSearchCV(classifier, param_grid=cv_params, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
    clf.fit(train_X, train_Y)
    print('best paras')
    print('grid_scores:{0}'.format(clf.grid_scores_))
    print('best paras{0}'.format(clf.best_params_))
    print('best score:{0}'.format(clf.best_score_))
    #classifier.fit(train_X, train_Y)
    print('train end')

    train_prob = clf.predict_proba(train_X)[:, 1]
    train_auc = cal_auc(train_Y, train_prob)

    print('Train auc %f' % (train_auc))

    #xgboost.plot_tree(classifier, num_trees=0)
    #test_prob = clf.predict_proba(test_X)[:, 1]
    #with open('xgboost_result.txt', 'w') as output:
    #    for i, key in enumerate(test_agg):
    #        output.write(str(key) + '\t' + str(test_prob[i]) + '\n')
    #plt.savefig('800.png', dpi=800)
