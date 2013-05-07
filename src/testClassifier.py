'''
Created on Mar 30, 2012

@author: Mujtaba Badat
'''

#must get 80% to win

from __future__ import division
from sklearn.utils import *
import numpy as np
from matplotlib import *
from pylab import *
from classify import MyClassifier
import scipy.io
import scipy.sparse
import time
from lil2 import *
import sys

#test_data = scipy.io.mmread('tweets_small.mtx').tocsr()
data = scipy.io.mmread('tweets_small.mtx').tocsr()

print "data loaded"

#lil2 = lil2(data)

#print "about to remove rows"

'''
for i in range(490000, 510000):
    print i
    lil2.removerow(490000)
'''

#lil2 = lil2.tocsr()

X = data[:,1:]
y = data[:,0]

#Xtest = data[490001:509999,1:]
#ytest = data[490001:509999,0]
Xtest = data[3000:5000, 1:]
ytest = data[3000:5000]

y = np.array(y.todense()).flatten()
ytest = np.array(ytest.todense()).flatten()

classifier = MyClassifier()
classifier.fit(X, y)
classifier.save_params('params.npz')
#classifier.load_params('params.npz')

prediction_start = time.time()
yhat = classifier.predict ( Xtest )
prediction_end = time.time()
prediction_time = prediction_end-prediction_start
accuracy = np.sum( yhat == ytest )/ytest.size

print "accuracy " + str(accuracy)
print "predict time " + str(prediction_time)