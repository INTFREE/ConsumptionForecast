res = []
f_write = open('test_result.csv', 'w')
with open('test_result_pro.csv', 'r') as f:
    title = f.readline().strip()
    f_write.write(title + '\n')
    for line in f.readlines():
        paras = line.strip().split('\t')
        user_id = paras[0]
        temp_res = paras[1].split(' ')[1][:-1]
        f_write.write(user_id + '\t' + temp_res + '\n')

f_write.close()