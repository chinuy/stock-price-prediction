from sklearn import preprocessing
import numpy
import datetime
import matplotlib.pyplot as plt

import util
import classifier

ETFs = ['XLE', 'XLU', 'XLK', 'XLB', 'XLP', 'XLY', 'XLI', 'XLV', 'SPY']

class Profolio:

    def __init__(self, name):
        self.name = name
        self.profits = numpy.zeros(252)

    def accProfits(self):
        return numpy.cumsum(self.profits)

    def annualSharpeRatio(self, n = 252):
        return numpy.sqrt(n) * self.profits.mean() / self.profits.std()

def smart_trade(etf, method, delta):
    parameters = [0.5, 0.5]

    data = util.get_data(etf, '2010/1/1', '2016/12/31')

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
    te = te[te.columns[0:-1]] # remove Up/Down label from testing
    clf = classifier.buildModel(tr, method, parameters)

    if method == 'RNN':
        te = numpy.reshape(numpy.array(te), (te.shape[0], 1, te.shape[1]))

    pred = clf.predict(te)

    profits = numpy.zeros(pred.size)
    for i in range(pred.size):
      if pred[i] < 0.5: # predict long
        p = (test.Close[i+1] - test.Open[i+1]) / test.Open[i+1]
      else: # predict short
        p = -(test.Close[i+1] - test.Open[i+1]) / test.Open[i+1]
      profits[i] = p
    return profits

def compareMethods():
    """
    Run all four methods to compare performance
    """
    legend = []
    sharpeRatio = []
    methods = ['SVM', 'RF', 'KNN', 'RNN']
    best_delta = [4, 3, 99, 20]
    color = ['r', 'b', 'y', 'g']

    for i in range(len(methods)):
        name = methods[i]
        delta = best_delta[i]
        my = Profolio(name)
        for etf in ETFs:
            my.profits += smart_trade(etf, name, delta)
        label, = plt.plot(range(1, 253), my.accProfits(),
                color=color[i], label=my.name)
        legend.append(label)
        sharpeRatio.append(my.annualSharpeRatio())

    plt.legend(handles=legend)
    plt.show()
    print sharpeRatio

def main():

    legend = []

    p0 = Profolio('All short-only')
    for etf in ETFs:
        data = util.get_data(etf, '2016/1/1', '2016/12/31')
        p0.profits -= data.Return
    label, = plt.plot(range(1, 253), p0.accProfits(), 'y--', label=p0.name)
    legend.append(label)

    # baseline1 SPY long-only
    p1 = Profolio('SPY long-only')
    data = util.get_data('SPY', '2016/1/1', '2016/12/31')
    p1.profits = data.Return * 9
    label, = plt.plot(range(1, 253), p1.accProfits(), 'b--', label=p1.name)
    legend.append(label)

    # baseline2 All long-only
    p2 = Profolio('All long-only')
    for etf in ETFs:
        data = util.get_data(etf, '2016/1/1', '2016/12/31')
        p2.profits += data.Return
    label, = plt.plot(range(1, 253), p2.accProfits(), 'g--', label=p2.name)
    legend.append(label)

    # My strategy
    my = Profolio('My strategy')
    my.profits = numpy.zeros(252)
    for etf in ETFs:
        my.profits += smart_trade(etf, 'RNN', 40)
    label, = plt.plot(range(1, 253), my.accProfits(), 'r--', label=my.name)
    legend.append(label)

    plt.legend(handles=legend)
    plt.show()

    print p1.annualSharpeRatio(), p2.annualSharpeRatio(), my.annualSharpeRatio()

if __name__ == '__main__':
    compareMethods()
