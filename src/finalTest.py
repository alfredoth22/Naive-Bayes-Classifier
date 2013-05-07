'''
Created on Apr 2, 2012

@author: Mujtaba Badat
'''

from classify import MyClassifier

classifier = MyClassifier()
classifier.load_params('params.npz')
yhat = classifier.predict(Xtest)
accuracy = np.sum( yhat == ytest ) / ytest.size