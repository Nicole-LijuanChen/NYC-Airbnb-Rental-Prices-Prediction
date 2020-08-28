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


def load_and_split_data(path):
    ''' Loads processed dataset and splits it into train:test datasets
        in a ratio of 75:25. Also sets the random_state for reproducible 
        results each time model is run.
    
        Parameters: None

        Returns:  (X_train, X_test, y_train, y_test):  tuple of numpy arrays
                  column_names: numpy array containing the feature names
    '''
    df = pd.read_csv(path) #load processed dataset 
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



def rf_estimators_scores(num_estimator_list,X_train, y_train, X_test, y_test):
    train_errors_rf = []
    test_errors_rf = []
    for num_est in num_estimator_list:
        rf = RandomForestRegressor(n_estimators = num_est, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred_test =  rf.predict(X_test)
        y_pred_train =  rf.predict(X_train)
    
        train_errors_rf.append(mean_squared_error(y_pred_train, y_train)) 
        test_errors_rf.append(mean_squared_error(y_pred_test, y_test))
    return train_errors_rf,test_errors_rf


def rf_trees_scores(num_trees,X_train, y_train, X_test, y_test):
    train_errors_rf = []
    test_errors_rf = []
    for num in num_trees:
        rf = RandomForestRegressor(n_estimators = 300, max_depth=num, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred_test =  rf.predict(X_test)
        y_pred_train =  rf.predict(X_train)
    
        train_errors_rf.append(mean_squared_error(y_pred_train, y_train)) 
        test_errors_rf.append(mean_squared_error(y_pred_test, y_test))
    return train_errors_rf,test_errors_rf

def rf_features_scores(num_features,X_train, y_train, X_test, y_test):
    train_errors_rf = []
    test_errors_rf = []
    for num in num_features:
        rf = RandomForestRegressor(n_estimators = 300, max_depth=15,max_features=num, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred_test =  rf.predict(X_test)
        y_pred_train =  rf.predict(X_train)
    
        train_errors_rf.append(mean_squared_error(y_pred_train, y_train)) 
        test_errors_rf.append(mean_squared_error(y_pred_test, y_test))
    return train_errors_rf,test_errors_rf


# def rf_features_scores(num_features,X,y):
#     train_r2_rf = []
#     oob_r2_rf = []

#     for num in num_features:
#         rf = RandomForestRegressor(n_estimators = 300, max_depth=15, max_features=num, oob_score=True, n_jobs=-1)
#         rf.fit(X, y)
#         y_pred =  rf.predict(X)
    
#         train_r2_rf.append(r2_score(y, y_pred)) 
#         oob_r2_rf.append(rf.oob_score_)
#     return train_r2_rf, oob_r2_rf


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
    max_depth = estimator.max_depth
    # initialize 
    train_scores = np.zeros((estimator.n_estimators,), dtype=np.float64)
    test_scores = np.zeros((estimator.n_estimators,), dtype=np.float64)
    # Get train score from each boost
    for i, y_train_pred in enumerate(estimator.predict(X_train)):
        train_scores[i] = mean_squared_error(y_train, y_train_pred)
    # Get test score from each boost
    for i, y_test_pred in enumerate(estimator.predict(X_test)):
        test_scores[i] = mean_squared_error(y_test, y_test_pred)
    plt.plot(train_scores, alpha=.5, label="{0} Train - max_depth {1}".format(
                                                                name, max_depth))
    plt.plot(test_scores, alpha=.5, label="{0} Test  - max_depth {1}".format(
                                                      name, max_depth), ls='--')
    plt.title(name, fontsize=16, fontweight='bold')
    plt.ylabel('MSE', fontsize=14)
    plt.xlabel('Iterations', fontsize=14)


def gridsearch_with_output(estimator, parameter_grid, X_train, y_train):
    '''
        Parameters: estimator: the type of model (e.g. RandomForestRegressor())
                    paramter_grid: dictionary defining the gridsearch parameters
                    X_train: 2d numpy array
                    y_train: 1d numpy array

        Returns:  best parameters and model fit with those parameters
    '''
    model_gridsearch = GridSearchCV(estimator,
                                    parameter_grid,
                                    n_jobs=-1,
                                    verbose=True,
                                    scoring='neg_mean_squared_error')
    model_gridsearch.fit(X_train, y_train)
    best_params = model_gridsearch.best_params_ 
    model_best = model_gridsearch.best_estimator_
    print("\nResult of gridsearch:")
    print("{0:<20s} | {1:<8s} | {2}".format("Parameter", "Optimal", "Gridsearch values"))
    print("-" * 55)
    for param, vals in parameter_grid.items():
        print("{0:<20s} | {1:<8s} | {2}".format(str(param), 
                                                str(best_params[param]),
                                                str(vals)))
    return best_params, model_best



def display_default_and_gsearch_model_results(model_default, model_gridsearch, 
                                              X_test, y_test):
    '''
        Parameters: model_default: fit model using initial parameters
                    model_gridsearch: fit model using parameters from gridsearch
                    X_test: 2d numpy array
                    y_test: 1d numpy array
        Return: None, but prints out mse and r2 for the default and model with
                gridsearched parameters
    '''
    name = model_default.__class__.__name__.replace('Regressor', '') # for printing
    y_test_pred = model_gridsearch.predict(X_test)
    mse = mean_squared_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)
    print("Results for {0}".format(name))
    print("Gridsearched model mse: {0:0.3f} | r2: {1:0.3f}".format(mse, r2))
    y_test_pred = model_default.predict(X_test)
    mse = mean_squared_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)
    print("     Default model mse: {0:0.3f} | r2: {1:0.3f}".format(mse, r2))





if __name__ == '__main__':
    # 1) load and train-test-split data 
    path = '../data/processed_v2.csv'
    X_train, X_test, y_train, y_test = load_and_split_data(path)
    print(f'X_train shape:  {X_train.shape}')
    print(f'y_train length: {len(y_train)}')
    print(f'X_test shape:  {X_test.shape}')
    print(f'y_train + y_test length: {len(y_train)+len(y_test)}')
        