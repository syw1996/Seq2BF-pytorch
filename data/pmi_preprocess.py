import math
import json
import torch
import numpy as np
import jieba
import jieba.posseg as pseg

def post_resp_dict(input_file, output_file, vocab_file):
    with open(input_file, 'r') as f:
        post_sents = f.readlines()
    
    vocabs = torch.load(vocab_file)
    for vocab in vocabs:
        if vocab[0] in input_file:
            print(vocab)
            vocab = vocab[1]
            break
    
    word_sentid_dict = dict()
    '''
    if 'src' in input_file:
        start_id = 2
    else:
        start_id = 4
    '''
    for i in range(len(vocab)):
        word_sentid_dict[vocab.itos[i]] = []

    for i, sent in enumerate(post_sents):
        sent_lst = sent.strip().split()
        try:
            sent_lst = [w.split('￨')[0] for w in sent_lst]
        except:
            continue
        # 除去句子中重复词语
        sent_lst = list(set(sent_lst))
        for w in sent_lst:
            if w in word_sentid_dict:
                word_sentid_dict[w].append(i)
        '''
        if i % 100000 == 0 and i != 0:
            print(i)
        '''
    print('total words num: ', len(word_sentid_dict))
    
    with open(output_file, 'w') as f:
        json.dump(word_sentid_dict, f)

def post_resp_array(post_dict_file, resp_dict_file, array_file):
    with open(post_dict_file, 'r') as f:
        post_dict = json.load(f)
    with open(resp_dict_file, 'r') as f:
        resp_file = json.load(f)
    w2w_array = np.zeros([len(post_dict), len(resp_file)])
    for i, post_w in enumerate(post_dict):
        post_w_ids = set(post_dict[post_w])
        if i % 100 == 0 and i:
            print(i)
        for j, resp_w in enumerate(resp_file):
            resp_w_ids = set(resp_file[resp_w])
            w2w_array[i, j] = len(post_w_ids & resp_w_ids)
    w2w_array.tofile(array_file)

def calculate_pmi(post_sent, resp_sent, post_dict, resp_dict, p2r_array, p2r_sum):
    # 除去不在post_dict中的token
    post_sent = [w for w in post_sent if w in post_dict]
    resp_pmi = dict()
    for wr in resp_sent:
        # if wr in resp_dict:
        wr_id = resp_dict[wr][0]
        wr_cnt = resp_dict[wr][1]
        wr_pmi = 0
        for wp in post_sent:
            # if wp in post_dict:
            wp_id = post_dict[wp][0]
            wp_cnt = post_dict[wp][1]
            wp_wr_cnt = p2r_array[wp_id, wr_id]
            p_pw = float(wp_wr_cnt)/p2r_sum
            p_p = float(wp_cnt)/p2r_sum
            p_r = float(wr_cnt)/p2r_sum
            if p_pw == 0:
                wr_pmi = 0
            else:
                wr_pmi += math.log(p_pw/(p_p * p_r))
            # wr_pmi = max(0, wr_pmi)
        resp_pmi[wr] = wr_pmi
    resp_pmi = sorted(resp_pmi.items(), key=lambda item:item[1], reverse=True)
    # print(resp_pmi)
    
    return resp_pmi

def word_part_of_speech(vocab_file):
    vocab = torch.load(vocab_file)
    post_vocab = vocab[0][1]
    resp_vocab = vocab[1][1]
    post_cnt = resp_cnt = 0 
    # match_lst = ['a','ad','an','i','n','nr','ns','nt','nz','r','tg','t','v','vn']
    match_lst = ['a', 'v', 'n','nr','ns','nt','nz']
    for i in range(2, len(post_vocab)):
        w = post_vocab.itos[i]
        ws = pseg.cut(w)
        for w in ws:
            if w.flag in match_lst:
                post_cnt += 1
    print(post_cnt)
    for i in range(2, len(resp_vocab)):
        w = resp_vocab.itos[i]
        ws = pseg.cut(w)
        for w in ws:
            if w.flag in match_lst:
                resp_cnt += 1
    print(resp_cnt)

def extract_post_keyword(post, post_cnt_dict, resp_cnt_dict, p2r_array, p2r_sum):
    resp_word_scores = dict()
    words = jieba.cut(post)
    post_sent = [w for w in words]
    for i, w in enumerate(resp_cnt_dict):
        if i < 4:
            continue
        resp_sent = [w]
        w_pmi = calculate_pmi(post_sent, resp_sent, post_cnt_dict, resp_cnt_dict, p2r_array, p2r_sum)
        resp_word_scores[w] = w_pmi[0][1]
    resp_word_scores = sorted(resp_word_scores.items(), key=lambda item:item[1], reverse=True)
    print(resp_word_scores[0])  


if __name__ == '__main__':
    # word_part_of_speech('data.vocab.pt')
    # post_resp_dict('tgt-train.txt', 'resp_dict', 'data.vocab.pt')
    # post_resp_array('post_dict', 'resp_dict', 'p2r_array.bin')
    # get_resp_keyword()
    
    with open('post_dict', 'r') as f:
        post_dict = json.load(f)
    with open('resp_dict', 'r') as f:
        resp_dict = json.load(f)
    p2r_array = np.fromfile('p2r_array.bin')
    p2r_sum = np.sum(p2r_array)
    p2r_array.shape = len(post_dict), len(resp_dict)
    '''
    post_cnt_dict = dict()
    resp_cnt_dict = dict()
    post_sum = np.sum(p2r_array, axis=1)
    for i, w in enumerate(post_dict):
        if i > 1:
            post_cnt_dict[w] = [i, post_sum[i]]
    with open('post_cnt.json', 'w') as f:
        json.dump(post_cnt_dict, f)
    resp_sum = np.sum(p2r_array, axis=0)
    for i, w in enumerate(resp_dict):
        if i > 3:
            resp_cnt_dict[w] = [i, resp_sum[i]]
    with open('resp_cnt.json', 'w') as f:
        json.dump(resp_cnt_dict, f)
    '''
    with open('post_cnt.json', 'r') as f:
        post_cnt_dict = json.load(f)
    with open('resp_cnt.json', 'r') as f:
        resp_cnt_dict = json.load(f)
    resp_cnt = dict()
    # resp词表截取高频词
    for w in resp_cnt_dict:
        if resp_cnt_dict[w][1] > 1000:
            resp_cnt[w] = resp_cnt_dict[w]
    resp_cnt_dict = resp_cnt
    posts = ['南京路上的房子很漂亮！', '我要换头像了！', '李友男友公开过了', '人大复试飘过', '挺漂亮的 祝福祝福', '承诺，就是一个骗子说给一个傻子听的。', '最近经济太不景气了。',
             '知道真相的我眼泪掉下来。', '今天温度多少？']
    for post in posts:
        print(post)
        extract_post_keyword(post, post_cnt_dict, resp_cnt_dict, p2r_array, p2r_sum)
    
    
    
    