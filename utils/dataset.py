
from torch.utils.data import Dataset
import numpy as np
from config import rnn_config

class MyDataSet(Dataset):
    def __init__(self, agg, log, event_vocab, flg = None):

        self.user_ids = agg.keys()
        self.agg = []
        self.log = []
        self.len = []
        if flg:
            self.flg = []
            self.is_test = False
        else:
            self.is_test = True
        for user_id in self.user_ids:
            self.agg.append(np.array(agg[user_id],dtype=float))
            self.len.append(1)
            if log[user_id] == []:
                self.log.append(np.zeros((rnn_config['max_len'],5),dtype=int))
            else:
                tmp_log = []
                for one_log in log[user_id]:
                    assert len(one_log)==5
                    tmp_feature = []
                    for event in one_log[:3]:
                        if not event in event_vocab:
                            event = 'unk'
                        tmp_feature.append(event_vocab[event])

                    tmp_feature.append(int(one_log[3])+1)
                    tmp_feature.append(int(one_log[4])+1)

                    tmp_log.append(tmp_feature)
                tmp_log = tmp_log[:rnn_config['max_len']]
                self.len.append(len(tmp_log))
                tmp_log = np.array(tmp_log, dtype=int)
                tmp_log = np.pad(tmp_log, ((rnn_config['max_len']-tmp_log.shape[0], 0), (0, 0)), 'constant')

                assert tmp_log.shape == (rnn_config['max_len'], 5)
                self.log.append(tmp_log)
            if flg:
                self.flg.append(flg[user_id])

    def __len__(self):
        assert len(self.agg) == len(self.log)
        return len(self.agg)

    def __getitem__(self, item):
        agg =  self.agg[item]

        event_0 = self.log[item][:, 0]
        event_1 = self.log[item][:, 1]
        event_2 = self.log[item][:, 2]
        hour = self.log[item][:, 3]
        type = self.log[item][:, 4]

        ret = {"agg_input": agg, "event_0": event_0, "event_1": event_1, "event_2": event_2,
               "hour": hour, "type": type, "len":self.len[item]}
        if not self.is_test:
            ret['label'] = self.flg[item]

        return ret
