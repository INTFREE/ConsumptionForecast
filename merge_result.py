from collections import defaultdict
score = defaultdict(float)
output_f = open('rnn_result.csv','w')
sample_f = open('./data/submit_sample.csv','r')
for i in range(10):

    with open('output_5_'+str(i)+'.txt','r') as f:
        for line in f:
            line = line.strip().split('\t')
            user_id = line[0]
            user_score = float(line[1])
            score[user_id] += user_score

title = sample_f.readline()
output_f.write(title)
for line in sample_f.readlines():
    user_id = line.strip().split()[0]
    output_f.write(user_id+'\t'+str(score[user_id])+'\n')

output_f.close()
sample_f.close()