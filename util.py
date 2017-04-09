# Ref: http://francescopochetti.com/stock-market-prediction-part-ii-feature-generation/
from sklearn import preprocessing
import pandas_datareader.data as web
import pandas as pd
import classifier
import numpy as np
import operator

pd.options.mode.chained_assignment = None

def preprocessData(dataset):

    le = preprocessing.LabelEncoder()

    # in case divid-by-zero
    dataset.Open[dataset.Open == 0] = 1

    # add prediction target: next day Up/Down
    threshold = 0.000
    dataset['UpDown'] = (dataset['Close'] - dataset['Open']) / dataset['Open']
    dataset.UpDown[dataset.UpDown >= threshold] = 'Up'
    dataset.UpDown[dataset.UpDown < threshold] = 'Down'
    dataset.UpDown = le.fit(dataset.UpDown).transform(dataset.UpDown)
    dataset.UpDown = dataset.UpDown.shift(-1) # shift 1, so the y is actually next day's up/down
    dataset = dataset.drop(dataset.index[-1]) # drop last one because it has no up/down value
    return dataset

def count_missing(dataframe):
    return (dataframe.shape[0] * dataframe.shape[1]) - dataframe.count().sum()

def performCV(X_train, y_train, folds, method, parameters, savemodel):
    """
    given complete dataframe, number of folds, the % split to generate
    train and test set and features to perform prediction --> splits
    dataframein test and train set. Takes train set and splits in k folds.
    - Train on fold 1, test on 2
    - Train on fold 1-2, test on 3
    - Train on fold 1-2-3, test on 4
    ....
    returns mean of test accuracies
    """
    print ''
    print 'Parameters --------------------------------> ', parameters
    print 'Size train set: ', X_train.shape

    k = int(np.floor(float(X_train.shape[0])/folds))

    print 'Size of each fold: ', k

    acc = np.zeros(folds-1)
    for i in range(2, folds+1):
        print ''
        split = float(i-1)/i
        print 'Splitting the first ' + str(i) + ' chuncks at ' + str(i-1) + '/' + str(i)
        data = X_train[:(k*i)]
        output = y_train[:(k*i)]
        print 'Size of train+test: ', data.shape
        index = int(np.floor(data.shape[0]*split))
        X_tr = data[:index]
        y_tr = output[:index]

        X_te = data[(index+1):]
        y_te = output[(index+1):]

        acc[i-2] = classifier.performClassification(X_tr, y_tr, X_te, y_te, method, parameters, savemodel)
        print 'Accuracy on fold ' + str(i) + ': ', acc[i-2]

    return acc.mean()

def performTimeSeriesSearchGrid(X_train, y_train, folds, method, grid, savemodel):
    """
    parameters is a dictionary with: keys --> parameter , values --> list of values of parameter
    """
    print ''
    print 'Performing Search Grid CV...'
    print 'Algorithm: ', method
    param = grid.keys()
    finalGrid = {}
    if len(param) == 1:
        for value_0 in grid[param[0]]:
            parameters = [value_0]
            accuracy = performCV(X_train, y_train, folds, method, parameters, savemodel)
            finalGrid[accuracy] = parameters
        final = sorted(finalGrid.iteritems(), key=operator.itemgetter(0), reverse=True)
        print ''
        print finalGrid
        print ''
        print 'Final CV Results: ', final
        return final[0]

    elif len(param) == 2:
        for value_0 in grid[param[0]]:
            for value_1 in grid[param[1]]:
                parameters = [value_0, value_1]
                accuracy = performCV(X_train, y_train, folds, method, parameters, savemodel)
                finalGrid[accuracy] = parameters
        final = sorted(finalGrid.iteritems(), key=operator.itemgetter(0), reverse=True)
        print ''
        print finalGrid
        print ''
        print 'Final CV Results: ', final
        return final[0]

def performFeatureSelection(stock_name, maxdeltas, start, end, start_test, savemodel, method, folds, parameters):
    """
    """
    finalGrid = {}
    print ''
    print '============================================================='
    print ''
    for maxdelta in range(3, maxdeltas, 2):
        dataset = get_data(stock_name, start, end)
        delta = range(1, maxdelta)
        print 'Delta days accounted: ', max(delta)
        dataset = applyFeatures(dataset, delta)
        dataset = preprocessData(dataset)
        print 'Number of NaN: ', count_missing(dataset)
        X_train, y_train, X_test, y_test  = \
            classifier.prepareDataForClassification(dataset, start_test)

        print ''
        accuracy = performCV(X_train, y_train, folds, method, parameters, savemodel)
        finalGrid[accuracy] = maxdelta

    final = sorted(finalGrid.iteritems(), key=operator.itemgetter(0), reverse=True)
    print ''
    print finalGrid
    print ''
    print 'Final CV Results: ', final
    return final[0]


def performParameterSelection(stock_name, bestdelta, start, end, start_test, savemodel, method, folds, grid):
    """
    """
    dataset = get_data(stock_name, start, end)
    delta = range(1, bestdelta + 1)
    print 'Delta days accounted: ', max(delta)
    dataset = applyFeatures(dataset, delta)
    dataset = preprocessData(dataset)
    X_train, y_train, X_test, y_test  = \
            classifier.prepareDataForClassification(dataset, start_test)

    return performTimeSeriesSearchGrid(X_train, y_train, folds, method, grid, savemodel)

def addFeatures(dataframe, close, returns, n):
    """
    operates on two columns of dataframe:
    - append previous n days' OHLC and Volumn information
    - given Return_* computes the return of day i respect to day i-n.
    - given Close_* computes its moving average on n days

    """
    for c in dataframe.columns[0:5]:
        dataframe[c + str(n)] = dataframe[c].shift(n)

    return_n = "Return" + str(n)
    dataframe[return_n] = dataframe[close].pct_change(n)

    roll_n = "RolMean" + str(n)
    dataframe[roll_n] = dataframe[returns].rolling(n).mean()

def applyFeatures(dataset, delta):
    """
    applies rolling mean and delayed returns to each dataframe in the list
    """
    columns = dataset.columns
    close = columns[-3]
    returns = columns[-1]
    for n in delta:
        addFeatures(dataset, close, returns, n)

    dataset = dataset.drop(dataset.index[0:max(delta)]) #drop NaN due to delta spanning

    # normalize columns
    scaler = preprocessing.MinMaxScaler()
    return pd.DataFrame(scaler.fit_transform(dataset),\
            columns=dataset.columns, index=dataset.index)


def get_data(name, start, end):
    data = web.get_data_yahoo(name, start, end)
    del data['Adj Close'] # we don't need Adj Close
    data['Return'] = (data['Close'] - data['Open']) / data['Open']

    return data
