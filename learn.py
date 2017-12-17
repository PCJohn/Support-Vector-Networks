import numpy as np
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

import inria

if __name__ == '__main__':
    X, Y, vX, vY = inria.load()
    C_range = np.outer(np.logspace(-1, 0, 2), np.array([1,5])).flatten()
    gamma_range = np.outer(np.logspace(-2, 0, 2), np.array([1, 5])).flatten()
    parameters = {'kernel': ['rbf'], 'C': C_range, 'gamma': gamma_range}
    clf = SVC()
    print "mems"
    grid_clf = GridSearchCV(estimator=clf, param_grid=parameters, verbose=2)
    print "done"
    grid_clf.fit(X, Y)
    joblib.dump(grid_clf, 'clf.pkl')
    accuracy = np.array(np.array(grid_clf.predict(vX) - vY, dtype='bool'), dtype='float').sum()
    print accuracy