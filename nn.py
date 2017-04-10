from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import numpy
import time
import util
import classifier
import datetime

def main():

    stock_name = 'SPY'
    delta = 4
    start = datetime.datetime(2010,1,1)
    end = datetime.datetime(2015,12,31)
    start_test = datetime.datetime(2015,1,1)

    dataset = util.get_data(stock_name, start, end)
    delta = range(1, delta)
    dataset = util.applyFeatures(dataset, delta)
    dataset = util.preprocessData(dataset)
    X_train, y_train, X_test, y_test  = \
        classifier.prepareDataForClassification(dataset, start_test)

    X_train = numpy.reshape(numpy.array(X_train), (X_train.shape[0], 1, X_train.shape[1]))

    X_test = numpy.reshape(numpy.array(X_test), (X_test.shape[0], 1, X_test.shape[1]))

    #Step 2 Build Model
    model = Sequential()

    model.add(LSTM(
        128,
        input_shape=(None, X_train.shape[2]),
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        240,
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        units=1))
    model.add(Activation('sigmoid'))

    start = time.time()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    #Step 3 Train the model
    model.fit(
        X_train,
        y_train,
        batch_size=4,
        epochs=4,
        validation_split=0.1)

    print model.predict(X_train)
    print model.evaluate(X_train, y_train)

if  __name__ == '__main__':
    main()
