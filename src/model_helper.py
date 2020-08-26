#!/usr/bin/env python3

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def load_and_split_data():
    ''' Loads processed dataset and splits it into train:test datasets
        in a ratio of 75:25. Also sets the random_state for reproducible 
        results each time model is run.
    
        Parameters: None

        Returns:  (X_train, X_test, y_train, y_test):  tuple of numpy arrays
                  column_names: numpy array containing the feature names
    '''
    df = pd.read_csv('data/processed_data.csv') #load processed dataset 
    X = df.copy()
    y = X.pop('price')
    X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.25, random_state=42)
    return X_train, X_test, y_train, y_test


def cross_val(estimator, X_train, y_train, nfolds):
    ''' Takes an instantiated model (estimator) and returns the average
        mean square error (mse) and coefficient of determination (r2) from
        kfold cross-validation.

        Parameters: estimator: model object
                    X_train: 2d numpy array
                    y_train: 1d numpy array
                    nfolds: the number of folds in the kfold cross-validation

        Returns:  mse: average mean_square_error of model over number of folds
                  r2: average coefficient of determination over number of folds
    
        There are many possible values for scoring parameter in cross_val_score.
        http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

        kfold is easily parallelizable, so set n_jobs = -1 in cross_val_score
    '''
    mse = cross_val_score(estimator, X_train, y_train, 
                          scoring='neg_mean_squared_error',
                          cv=nfolds, n_jobs=-1) * -1
    # mse multiplied by -1 to make positive
    r2 = cross_val_score(estimator, X_train, y_train, 
                         scoring='r2', cv=nfolds, n_jobs=-1)
    mean_mse = mse.mean()
    mean_r2 = r2.mean()
    name = estimator.__class__.__name__
    print("{0:<25s}  Train CV | MSE: {1:0.3f} | R2: {2:0.3f}".format(name,
                                                        mean_mse, mean_r2))
    return mean_mse, mean_r2

def rf_score_plot(randforest, X_train, y_train, X_test, y_test):
    '''
        Parameters: randforest: RandomForestRegressor
                    X_train: 2d numpy array
                    y_train: 1d numpy array
                    X_test: 2d numpy array
                    y_test: 1d numpy array

        Returns: The prediction of a random forest regressor on the test set
    '''
    randforest.fit(X_train, y_train)
    y_test_pred = randforest.predict(X_test)
    test_score = mean_squared_error(y_test, y_test_pred)
    plt.axhline(test_score, alpha = 0.7, c = 'y', lw=3, ls='-.', label = 
                                                        'Random Forest Test')


def stage_score_plot(estimator, X_train, y_train, X_test, y_test):
    '''
        Parameters: estimator: GradientBoostingRegressor or AdaBoostRegressor
                    X_train: 2d numpy array
                    y_train: 1d numpy array
                    X_test: 2d numpy array
                    y_test: 1d numpy array

        Returns: A plot of the number of iterations vs the MSE for the model for
        both the training set and test set.
    '''
    estimator.fit(X_train, y_train)
    name = estimator.__class__.__name__.replace('Regressor', '')
    learn_rate = estimator.learning_rate
    # initialize 
    train_scores = np.zeros((estimator.n_estimators,), dtype=np.float64)
    test_scores = np.zeros((estimator.n_estimators,), dtype=np.float64)
    # Get train score from each boost
    for i, y_train_pred in enumerate(estimator.staged_predict(X_train)):
        train_scores[i] = mean_squared_error(y_train, y_train_pred)
    # Get test score from each boost
    for i, y_test_pred in enumerate(estimator.staged_predict(X_test)):
        test_scores[i] = mean_squared_error(y_test, y_test_pred)
    plt.plot(train_scores, alpha=.5, label="{0} Train - learning rate {1}".format(
                                                                name, learn_rate))
    plt.plot(test_scores, alpha=.5, label="{0} Test  - learning rate {1}".format(
                                                      name, learn_rate), ls='--')
    plt.title(name, fontsize=16, fontweight='bold')
    plt.ylabel('MSE', fontsize=14)
    plt.xlabel('Iterations', fontsize=14)




if __name__ == '__main__':
    # 1) load and train-test-split data 
    X_train, X_test, y_train, y_test = load_and_split_data()
    print(f'X_train shape:  {X_train.shape}')
    print(f'y_train length: {len(y_train)}')
    print(f'y_train + y_test length: {len(y_train)+len(y_test)}')
        