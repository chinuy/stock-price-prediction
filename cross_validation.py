import util
import datetime

def main():

    stock_name = 'SPY'
    method = 'KNN'

    maxdeltas = 99 # min is 3
    folds = 10

    start = datetime.datetime(2014,1,1)
    end = datetime.datetime(2015,12,31)
    start_test = datetime.datetime(2015,1,1)

    # UNCOMMENT to do Feature selection
    # parameters = [16, 0.125]
    # util.performFeatureSelection(stock_name, maxdeltas, start, end, start_test, False, method, folds,  parameters)

    #grid = {'c': [2**x for x in range(10, -2, -1)], 'g': [2**x for x in range(-15,1, 2)]}
    grid = {'k': range(3, 10)}
    util.performParameterSelection(stock_name, maxdeltas, start, end, start_test, False, method, folds, grid)

if __name__ == '__main__':
    main()
