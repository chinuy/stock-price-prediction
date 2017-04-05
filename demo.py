# Ref: http://francescopochetti.com/stock-market-prediction-part-ii-feature-generation/
import pandas_datareader.data as web
import pandas as pd

def addFeatures(dataframe, close, returns, n):
    """
    operates on two columns of dataframe:
    - n >= 2
    - given Return_* computes the return of day i respect to day i-n.
    - given Close_* computes its moving average on n days

    """

    return_n = "Time" + str(n)
    dataframe[return_n] = dataframe[close].pct_change(n)

    roll_n = "RolMean" + str(n)
    dataframe[roll_n] = pd.rolling_mean(dataframe[returns], n)

def applyRollMeanDelayedReturns(dataset, delta):
    """
    applies rolling mean and delayed returns to each dataframe in the list
    """
    columns = dataset.columns
    close = columns[-3]
    returns = columns[-1]
    for n in delta:
        addFeatures(dataset, close, returns, n)

    return dataset

def get_data(name):
    data = web.get_data_yahoo(name, '1/1/2016', '12/31/2016')
    del data['Adj Close'] # we don't need Adj Close
    data['Return'] = (data['Close'] - data['Open']) / data['Open']

    return applyRollMeanDelayedReturns(data, [3, 7, 11])

def main():
    get_data('SPY')

if __name__ == '__main__':
    main()
