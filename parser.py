#!/usr/bin/python

__author__="Jiada Chen <jc4730@columbia.edu>"

import os
import sys
import json
import numpy as np
import pandas as pd


### Q4 - count words
def parseTree(tree,cnt_dic):
    if isinstance(tree,basestring):
        cnt_dic.setdefault(tree,0)
        cnt_dic[tree]+=1
        
        return cnt_dic
    
    if len(tree) == 3:
        cnt_dic = parseTree(tree[1],cnt_dic)
        cnt_dic = parseTree(tree[2],cnt_dic)
    elif len(tree) == 2:
        cnt_dic = parseTree(tree[1],cnt_dic)
    
    return cnt_dic


### Q4 -  replace _RARE_
def rareTree(tree,cnt_dic):
    # if chile node
    if isinstance(tree,basestring):
        # if count < 5
        if cnt_dic[tree] < 5:
            tree = '_RARE_'
        return tree
    
    if len(tree) == 3:
        tree[1] = rareTree(tree[1],cnt_dic)
        tree[2] = rareTree(tree[2],cnt_dic)
    elif len(tree) == 2:
        tree[1] = rareTree(tree[1],cnt_dic)
    
    return tree


### Q5 & Q6 - recover String from backpointer
def bp_2_str(i,j,x,bp,words):
    dqote='"'
    if i == j:
        return "[" + dqote + x + dqote +", " + dqote + words[i] + dqote + "]"
    
    s = bp[(i,j,x)][0]
    y = bp[(i,j,x)][1]
    z = bp[(i,j,x)][2]

    return "[" + dqote +x +dqote+", " + bp_2_str(i,s,y,bp,words) + ", " + bp_2_str(s+1,j,z,bp,words) + "]"


def q4(in_file, out_file):
    cnt_dic = {}
    f = open(in_file, 'r')
    for line in f:
        tree = json.loads(line)
        cnt_dic = parseTree(tree,cnt_dic)

    f = open(in_file, 'r')
    with open(out_file, 'w') as output:
        for line in f:
            tree = json.loads(line)
            tree2 = rareTree(tree,cnt_dic)
            output.write(json.dumps(tree2) + '\n')


def q5(train_file,dev_file,out_file):
    # Train
    count_file = "cfg.counts"
    os.system("python count_cfg_freq.py "+train_file +" > " + count_file)
    
    data = pd.read_csv(count_file,sep =' ',names=['a','b','c','d','e'],quoting=3)

    uni_gram = data[data.b == 'NONTERMINAL'][['a','c']].rename(columns={'a':'Tag_Count','c':'X'})

    binary = data[data.b == 'BINARYRULE'][['a','c','d','e']].rename(columns={'a':'Count','c':'X','d':'Y_1','e':'Y_2'})
    binary = binary.merge(uni_gram,how='left')
    binary['q'] = np.log(binary['Count']/binary['Tag_Count'])/np.log(2)

    unary = data[data.b == 'UNARYRULE'][['a','c','d']].rename(columns={'a':'Count','c':'X','d':'Y'})
    unary = unary.merge(uni_gram,how='left')
    unary['q'] = np.log(unary['Count']/unary['Tag_Count'])/np.log(2)

    bi_dic = {}
    for index, row in binary.iterrows():
        bi_dic[(row['X'],row['Y_1'],row['Y_2'])]=row['q']

    un_dic = {}
    for index, row in unary.iterrows():
        un_dic[(row['X'],row['Y'])] = row['q']

    non_term = list(set([item for sublist in bi_dic.keys() for item in sublist]))
    term = set([i[1] for i in un_dic.keys()])

    bi_key = bi_dic.keys()
    bi_rules = {}
    for key in bi_key:
        bi_rules.setdefault(key[0],[])
        bi_rules[key[0]].append((key[1],key[2]))

    # Test
    dev_file = open(dev_file, 'r')
    sentences = dev_file.readlines()

    valid_x = set(bi_rules.keys())

    with open(out_file, 'w') as output:
        # CKY algorithm
        for sentence in sentences:
            sentence = sentence.rstrip('\n')
            words = sentence.split(' ')
            n = len(words)
            
            pi = {}
            bp = {}
            for i in range(n):
                for x in non_term:
                    word = words[i]
                    if word not in term:
                        word = '_RARE_'
                    try:
                        pi[(i,i,x)] = un_dic[(x,word)]
                    except KeyError:
                        pass
        
            for l in range(1,n):
                for i in range(n-l):
                    j = i+l
                    for x in non_term:
                        tmp_max = float("-Inf")
                        tmp_bp = None
                        if x in valid_x:
                            for rule in bi_rules[x]:
                                y = rule[0]
                                z = rule[1]
                                for s in range(i,j):
                                    try:
                                        tmp_pi = bi_dic[(x,y,z)]+pi[(i,s,y)]+pi[(s+1,j,z)]
                                        if tmp_pi > tmp_max:
                                            tmp_max = tmp_pi
                                            tmp_bp = (s,y,z)
                                    except KeyError:
                                        pass
                        if tmp_bp is not None:
                            pi[(i,j,x)] = tmp_max
                            bp[(i,j,x)] = tmp_bp

            bp_key_set = set(bp.keys())
            if (0,n-1,'S') in bp_key_set:
                key_str = bp_2_str(0,n-1,'S',bp,words)
                output.write(key_str + '\n')
            else:
                key_max = float("-Inf")
                max_bp_key = None
                for bp_key in bp_key_set:
                    if (bp_key[0] == 0) & (bp_key[1] == n-1):
                        if pi[bp_key] > key_max:
                            key_max = pi[bp_key]
                            max_bp_key = bp_key
                key_str = bp_2_str(0,n-1,max_bp_key[2],bp,words)
                output.write(key_str + '\n')



def main():
    question = sys.argv[1]

    if question == 'q4':
        in_file = sys.argv[2]
        out_file =sys.argv[3]
        q4(in_file, out_file)
    elif question == 'q5':
        train_file = sys.argv[2]
        dev_file = sys.argv[3]
        out_file = sys.argv[4]
        q5(train_file,dev_file,out_file)
    elif question == 'q6':
        train_file = sys.argv[2]
        dev_file = sys.argv[3]
        out_file = sys.argv[4]
        q5(train_file,dev_file,out_file)



if __name__ == "__main__":
    main()
