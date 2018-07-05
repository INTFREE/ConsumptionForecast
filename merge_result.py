from collections import defaultdict
score = defaultdict(float)
output_f = open('merge_rnn_rbf_svm_result.csv','w')
sample_f = open('./data/submit_sample.csv','r')
result_list = ['6.26_rnn_result.csv','test_result_rbf_svm.csv']
weights = [1,0.1]
for i, result_name in enumerate(result_list):
    with open(result_name,'r') as f:
        f.readline()
        for line in f:
            line = line.strip().split('\t')
            user_id = line[0]
            user_score = float(line[1])
            score[user_id] += user_score*weights[i]

title = sample_f.readline()
output_f.write(title)
for line in sample_f.readlines():
    user_id = line.strip().split()[0]
    output_f.write(user_id+'\t'+str(score[user_id])+'\n')

output_f.close()
sample_f.close()