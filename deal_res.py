res = []
f_write = open('xgboost_test_result.csv', 'w')
with open('test_result.csv', 'r') as f:
    title = f.readline().strip()
    f_write.write(title + '\n')
    for line in f.readlines():
        paras = line.strip().split('\t')
        user_id = paras[0]
        temp_res = paras[1].split(' ')
        if user_id == '6711':
            print(temp_res)
        if temp_res[-1] != ']':
            temp_res = temp_res[-1].strip()[:-1]
        else:
            temp_res = temp_res[-2].strip()

        f_write.write(user_id + '\t' + temp_res + '\n')

f_write.close()