import learning
import classifier
import datetime

parameters = [2, 1]

data = learning.get_data('SPY', '2010/1/1', '2016/12/31')

data = learning.applyRollMeanDelayedReturns(data, range(2, 11))

tr = data[data.index < datetime.datetime(2015,12,31)]
te = data[data.index > datetime.datetime(2015,12,31)]
clf = classifier.buildModel(tr, 'SVM', parameters)
print clf.predict(te)
