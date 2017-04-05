import datetime
import cPickle
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

def prepareDataForClassification(dataset, start_test):
    """
    generates categorical output column, attach to dataframe
    label the categories and split into train and test
    """
    le = preprocessing.LabelEncoder()

    dataset['UpDown'] = dataset['Return']
    dataset.UpDown[dataset.UpDown >= 0] = 'Up'
    dataset.UpDown[dataset.UpDown < 0] = 'Down'
    dataset.UpDown = le.fit(dataset.UpDown).transform(dataset.UpDown)

    features = dataset.columns[1:-1]
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
    clf = SVC()
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
    n = parameters[0]
    l =  parameters[1]
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
