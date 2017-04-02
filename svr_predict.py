# REF: https://github.com/llSourcell/predicting_stock_prices/demo.py
import sys
import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt


class SVR_Solver:

    def __init__(self):

        self.dates = []
        self.prices = []

    def get_data(self, filename):

        with open(filename, 'r') as csvfile:
            csvFileReader = csv.reader(csvfile)
            next(csvFileReader)     # skipping column names
            for row in csvFileReader:
                self.prices.append(float(row[1]))
            self.dates = range(1, len(self.prices)+1)
        return

    def training(self, c, g):

        self.dates = np.reshape(self.dates,(len(self.dates), 1))
        self.model = SVR(kernel= 'rbf', C= c, gamma= g)
        self.model.fit(self.dates, self.prices) # fitting the data points in the models

    def draw(self):

        plt.scatter(self.dates, self.prices, color= 'black', label= 'Data')
        plt.plot(self.dates, self.model.predict(self.dates), color= 'red', label= 'RBF model')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title('SVR test for SPY trimmed data (2014, 2015)')
        #plt.legend()
        plt.show()

    def predict(self, x):
        return self.model.predict(x)[0]

def main():
    solver = SVR_Solver()
    solver.get_data(sys.argv[1])
    solver.training(1e3, 0.1)
    solver.draw()
    #print solver.predict_price(505)

if __name__ == '__main__':
    main()
