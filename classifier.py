import datetime
import cPickle
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

def buildModel(dataset, method, parameters):
    """
    Build model for predicting real testing data
    """

    le = preprocessing.LabelEncoder()

    # scaling colums
    scaler = preprocessing.MinMaxScaler()
    dataset = pd.DataFrame(scaler.fit_transform(dataset),\
            columns=dataset.columns, index=dataset.index)

    # add prediction target: next day Up/Down
    dataset['UpDown'] = (dataset['Close'] - dataset['Open'])
    dataset.UpDown[dataset.UpDown >= 0] = 'Up'
    dataset.UpDown[dataset.UpDown < 0] = 'Down'
    dataset.UpDown = le.fit(dataset.UpDown).transform(dataset.UpDown)
    dataset.UpDown = dataset.UpDown.shift(-1) # shift 1, so the y is actually next day's up/down
    dataset = dataset.drop(dataset.index[-1]) # drop last one because it has no up/down value

    features = dataset.columns[0:-1]
    c = parameters[0]
    g =  parameters[1]
    clf = SVC(C=c, gamma=g)
    return clf.fit(dataset[features], dataset['UpDown'])

def prepareDataForClassification(dataset, start_test):
    """
    generates categorical output column, attach to dataframe
    label the categories and split into train and test
    """
    le = preprocessing.LabelEncoder()

    # scaling colums
    scaler = preprocessing.MinMaxScaler()
    dataset = pd.DataFrame(scaler.fit_transform(dataset),\
            columns=dataset.columns, index=dataset.index)

    # add prediction target: next day Up/Down
    dataset['UpDown'] = (dataset['Close'] - dataset['Open'])
    dataset.UpDown[dataset.UpDown >= 0] = 'Up'
    dataset.UpDown[dataset.UpDown < 0] = 'Down'
    dataset.UpDown = le.fit(dataset.UpDown).transform(dataset.UpDown)
    dataset.UpDown = dataset.UpDown.shift(-1) # shift 1, so the y is actually next day's up/down
    dataset = dataset.drop(dataset.index[-1]) # drop last one because it has no up/down value

    features = dataset.columns[0:-1]
    X = dataset[features]
    y = dataset.UpDown

    X_train = X[X.index < start_test]
    y_train = y[y.index < start_test]

    X_test = X[X.index >= start_test]
    y_test = y[y.index >= start_test]

    return X_train, y_train, X_test, y_test


def performClassification(X_train, y_train, X_test, y_test, method, parameters, savemodel):
    """
    performs classification on returns using serveral algorithms
    """

    print 'Performing ' + method + ' Classification...'
    print 'Size of train set: ', X_train.shape
    print 'Size of test set: ', X_test.shape

    if method == 'RF':
        return performRFClass(X_train, y_train, X_test, y_test, parameters, savemodel)

    elif method == 'KNN':
        return performKNNClass(X_train, y_train, X_test, y_test, parameters, savemodel)

    elif method == 'SVM':
        return performSVMClass(X_train, y_train, X_test, y_test, parameters, savemodel)

    elif method == 'ADA':
        return performAdaBoostClass(X_train, y_train, X_test, y_test, parameters, savemodel)

    elif method == 'GTB':
        return performGTBClass(X_train, y_train, X_test, y_test, parameters, savemodel)

####### Classifier Arsenal ####################################################

def performRFClass(X_train, y_train, X_test, y_test, parameters, savemodel):
    """
    Random Forest Binary Classification
    """
    clf = RandomForestClassifier(n_estimators=1000, n_jobs=-1)
    clf.fit(X_train, y_train)

    if savemodel == True:
        fname_out = '{}.pickle'.format('RF')
        with open(fname_out, 'wb') as f:
            cPickle.dump(clf, f, -1)

    accuracy = clf.score(X_test, y_test)

    return accuracy

def performKNNClass(X_train, y_train, X_test, y_test, parameters, savemodel):
    """
    KNN binary Classification
    """
    clf = neighbors.KNeighborsClassifier()
    clf.fit(X_train, y_train)

    if savemodel == True:
        fname_out = '{}-{}.pickle'.format(datetime.now())
        with open(fname_out, 'wb') as f:
            cPickle.dump(clf, f, -1)

    accuracy = clf.score(X_test, y_test)

    return accuracy

def performSVMClass(X_train, y_train, X_test, y_test, parameters, savemodel):
    """
    SVM binary Classification
    """
    c = parameters[0]
    g =  parameters[1]
    clf = SVC(C=c, gamma=g)
    clf.fit(X_train, y_train)

    if savemodel == True:
        fname_out = '{}-{}.pickle'.format(datetime.now())
        with open(fname_out, 'wb') as f:
            cPickle.dump(clf, f, -1)

    accuracy = clf.score(X_test, y_test)

    return accuracy

def performAdaBoostClass(X_train, y_train, X_test, y_test, parameters, savemodel):
    """
    Ada Boosting binary Classification
    """
    # n = parameters[0]
    # l =  parameters[1]
    clf = AdaBoostClassifier()
    clf.fit(X_train, y_train)

    if savemodel == True:
        fname_out = '{}-{}.pickle'.format(datetime.now())
        with open(fname_out, 'wb') as f:
            cPickle.dump(clf, f, -1)

    accuracy = clf.score(X_test, y_test)

    return accuracy

def performGTBClass(X_train, y_train, X_test, y_test, parameters, savemodel):
    """
    Gradient Tree Boosting binary Classification
    """
    clf = GradientBoostingClassifier(n_estimators=100)
    clf.fit(X_train, y_train)

    if savemodel == True:
        fname_out = '{}-{}.pickle'.format(datetime.now())
        with open(fname_out, 'wb') as f:
            cPickle.dump(clf, f, -1)

    accuracy = clf.score(X_test, y_test)

    return accuracy
