from __future__ import division
import re, random, pickle, stats
from math import log

def entropy(X):
    p = {}
    values = set(X)
    for x in values:
        p[x] = X.count(x) / len(X)
    return -sum(p[x]*log(p[x],2) for x in values)

def cross_entropy(X, w):
    # partition examples into 2 lists: examples where the word appears and doesn't appear
    word_on  = []
    word_off = []
    for (label, example) in X:
        if word in example:
            word_on.append(label)
        else:
            word_off.append(label)
    N = len(X)
    # there are 2 values for x on and off
    return (len(word_on) / N) * entropy(word_on) + (len(word_off) / N) * entropy(word_off)

# open and load data.
f = open('data','r')
data = pickle.load(f)
f.close()
print '|data|  =', len(data)

# reformat data
for i in range(len(data)):
    x,y = data[i]
    data[i] = (int(x) > 5, set(re.findall('[a-z\'\-]+', y.lower())))

alldata = data
data    = data[0:4000]      # only look at the first 4000 examples

# get a list of possible labels
labels = set([x for x,y in data])
print '|labels|=', len(labels)

# get a list of possible words
words = set()
for label, text in data:
    words.update(text)
print '|words| =', len(words)

#---------------------------------------------------

# fill in the tally
# initialize tallys to zeros
tally = dict((w,{}) for w in words)
for w in words:
    for y in labels:
        tally[w][y] = 0.0
for label, example in data:
    for word in example:
        tally[word][label] += 1.0

# chi-squared test requires >= 5 marks in each cell
toss = set()
for w in words:
    for y in labels:
        if tally[w][y] < 5:
            toss.add(w)
            break
print 'tossing out:', len(toss)
for w in toss:
    tally.pop(w)
    words.remove(w)


#---------------------------------------------------
print '---------------------------------------'

# calculate the entropy in the labels
data_entropy = entropy([x for x,y in data])
print 'data entropy'

IG = {}
for word in words:
    IG[word] = data_entropy - cross_entropy(data, word)
print 'IG calculated'

X = [(y,x) for x,y in IG.iteritems()]   # make the IG table easy to sort.
X.sort()                                # NOTE: sorting goes smallest first, we want biggest
for i in range(100):
    print X[-i][1],
