if __name__ == '__main__':
    train_dir = './data/train/'
    test_dir = './data/test/'
    user_profile = []
    with open(train_dir + 'train_agg.csv', 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            paras = line.strip().split('\t')
            temp_re = [int(paras[-1])]
            for para in paras[:-1]:
                temp_re.append(float(para))
            user_profile.append(temp_re)
