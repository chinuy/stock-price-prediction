import util
import classifier
import datetime
from sklearn import preprocessing

stocks = ['XLE', 'XLU', 'XLK', 'XLB', 'XLP', 'XLY', 'XLI', 'XLV', 'SPY']

class Profolio:

    def __init__(self, name):
        self.name = name

# baseline1
stock_name = 'SPY'
data = util.get_data(stock_name, '2016/1/1', '2016/12/31')

stock_name = stocks[-1]
method = 'SVM'
delta = 4

parameters = [64, 0.3125]

data = util.get_data(stock_name, '2010/1/1', '2016/12/31')

# keep a copy for unscaled data for later gain calculation
# TODO replace by MinMax_Scaler.inverse_transform()
#
# the first day of test is 2015/12/30. Using this data on this day to predict
# Up/Down of 2016/01/04
test = data[data.index > datetime.datetime(2015,12,30)]

le = preprocessing.LabelEncoder()
test['UpDown'] = (test['Close'] - test['Open']) / test['Open']
threshold = 0.001
test.UpDown[test.UpDown >= threshold] = 'Up'
test.UpDown[test.UpDown < threshold] = 'Down'
test.UpDown = le.fit(test.UpDown).transform(test.UpDown)
test.UpDown = test.UpDown.shift(-1) # shift 1, so the y is actually next day's up/down

dataMod = util.applyFeatures(data, range(1, delta))
dataMod = util.preprocessData(dataMod)

tr = dataMod[dataMod.index <= datetime.datetime(2015,12,31)]
te = dataMod[dataMod.index > datetime.datetime(2015,12,30)]
clf = classifier.buildModel(tr, method, parameters)
pred = clf.predict(te[te.columns[0:-1]])

profit = 0
for i in range(pred.size-1):
  if pred[i] == 0.0: # predict long
    profit += (test.Close[i+1] - test.Open[i+1]) / test.Open[i+1]
  else: # predict short
    profit -= (test.Close[i+1] - test.Open[i+1]) / test.Open[i+1]
  print test.index[i], test.UpDown[i], pred[i], profit
