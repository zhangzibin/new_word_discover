#!/usr/bin/env python
#encoding=utf-8

from pyspark import SparkContext
import re
import sys
import math


# return: {候选集序列(长度1-5):频率}
def candidate_freq(line):
    re_chinese = re.compile(u'[^a-zA-Z0-9\u4e00-\u9fa5]+')
    # sentence = re_chinese.sub('', line.decode('utf-8').rstrip())
    sentence = re_chinese.sub('', line.rstrip())
    sen_len = len(sentence)
    result = []
    for j in xrange(1,6):
        for i in xrange(sen_len-j, -1, -1):
            result.append((sentence[i:i+j], 1))
    return result

# return: {word: 凝固度}
def solidificaton(line):
    word, w_freq = line[0], float(line[1])
    length = len(word)
    ninggudu = 0.
    if length == 2:
        word1, word2 = word
        ninggudu = w_freq / (freqs_share.value[word1]*freqs_share.value[word2])
    elif length == 3:
        word1, word2 = word[:2], word[2:3]
        ninggudu1 = w_freq / (freqs_share.value[word1]*freqs_share.value[word2])
        word1, word2 = word[:1], word[1:3]
        ninggudu2 = w_freq / (freqs_share.value[word1]*freqs_share.value[word2])
        ninggudu = min(ninggudu1, ninggudu2)
    elif length == 4:
        word1, word2 = word[:1], word[1:4]
        ninggudu1 = w_freq / (freqs_share.value[word1]*freqs_share.value[word2])
        word1, word2 = word[:2], word[2:4]
        ninggudu2 = w_freq / (freqs_share.value[word1]*freqs_share.value[word2])
        word1, word2 = word[:3], word[3:4]
        ninggudu3 = w_freq / (freqs_share.value[word1]*freqs_share.value[word2])
        ninggudu = min(ninggudu1, ninggudu2, ninggudu3)
    return (word, math.log(ninggudu) + math.log(freqs_sum.value))

# input: {word: list of right neighbours freq}
# return: {word: entropy}
def entropy(line):
    w, neighbours = line
    right_sum = sum(neighbours)
    right_prob = map(lambda x:float(x)/right_sum, neighbours)
    entropy = sum(map(lambda x:-(x)*math.log(x), right_prob))
    return (w, entropy)

if __name__ == '__main__':
    sc = SparkContext(appName='new word')

    filename = sys.argv[1]
    word_min_freq = int(sys.argv[2])

    print 'Loading data...'
    # text = sc.parallelize(sc.textFile(filename).take(10000))
    text = sc.textFile(filename)
    text_r = text.map(lambda sentence:sentence[::-1])

    # 计算所有n-gram的词频，包括正反
    print 'Extract all n-grams...'
    freqs = text.flatMap(candidate_freq).reduceByKey(lambda a,b: a+b).filter(lambda line: line[1]>=word_min_freq)
    freqs_r = text_r.flatMap(candidate_freq).reduceByKey(lambda a,b: a+b).filter(lambda line: line[1]>=word_min_freq)
    freqs_sum = sc.broadcast(freqs.values().sum())
    '''
    freqs = freqs.map(lambda line: (line[0], float(line[1])/freqs_sum.value))
    freqs_r = freqs_r.map(lambda line: (line[0], float(line[1])/freqs_sum.value))
    '''
    # print 'Number of n-grams: %d' % freqs.count()

    # 构造候选集，长度为2-4的词
    print 'Extract all candidates...'
    candidates = freqs.filter(lambda line: 2<=len(line[0])<=4)
    candidates_r = freqs_r.filter(lambda line: 2<=len(line[0])<=4)
    # print 'Number of candidates: %d' % candidates.count()

    # 计算凝固度
    print 'Calculate candidates solidificaton...'
    freqs_share = sc.broadcast(freqs.collectAsMap()) # share频率，用于计算凝固度
    solids = candidates.map(solidificaton)
    '''
    print 'Top-100 solidificaton:'
    high_solids = solids.top(100, key=lambda x:x[1])
    for k,v in high_solids:
        print k.encode('utf-8'), v
    '''

    # 计算右熵
    print 'Calculate candidates right entropy...'
    candidates_share = sc.broadcast(set(candidates.map(lambda line:line[0]).collect()))
    entropy_right = freqs.filter(lambda line: len(line[0]) >= 3 and line[0][:-1] in candidates_share.value)\
                            .map(lambda line: (line[0][:-1], line[1])).groupByKey().map(entropy)
    '''
    print 'Top-100 right entropy:'
    high_right = entropy_right.top(100, key=lambda x:x[1])
    for k,v in high_right:
        print k.encode('utf-8'), v
    '''

    # 计算左熵
    print 'Calculate candidates left entropy...'
    candidates_r_share = sc.broadcast(set(candidates_r.map(lambda line:line[0]).collect()))
    entropy_left = freqs_r.filter(lambda line: len(line[0]) >= 3 and line[0][:-1] in candidates_r_share.value)\
                            .map(lambda line: (line[0][:-1], line[1])).groupByKey().map(entropy)
    '''
    print 'Top-100 left entropy:'
    high_left = entropy_left.top(100, key=lambda x:x[1])
    for k,v in high_left:
        print k[::-1].encode('utf-8'), v
    '''

    # 找新词
    high_freq_dict = dict(candidates.collect())
    high_solids_dict = dict(solids.collect())
    high_right_dict = dict(entropy_right.collect())
    high_left_dict = dict([(w[::-1],v) for w,v in entropy_left.collect()])

    words = set(high_freq_dict.keys()) & set(high_right_dict.keys()) & \
        set(high_solids_dict.keys()) & set(high_left_dict.keys())
    words = sc.parallelize(words) \
        .map(lambda w: (w.encode('utf-8'), str(high_freq_dict[w]), str(high_solids_dict[w]), str(high_right_dict[w]), str(high_left_dict[w]))) \
        .sortBy(lambda word: -float(word[1]))
    words = words.collect()
    words = ['\t'.join(w) for w in words]
    with open('result.txt', 'wb') as f:
        f.write('\n'.join(words))
