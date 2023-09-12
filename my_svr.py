"""
@author: valentin morel

Trained a SVR with the recorded data to map the gaze
"""

from typing import Tuple, List

import warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    explained_variance_score,
)
from sklearn.model_selection import train_test_split, GridSearchCV
from joblib import dump

warnings.filterwarnings("ignore", category=DeprecationWarning)


def my_svr_function() -> Tuple[GridSearchCV, GridSearchCV]:
    """
    Trains two SVR models to predict world coordinates (cx_world, cy_world) based on pupil coordinates.

    The function performs the following steps:
    1. Reads data from 'pupil_coord.csv'.
    2. Separates the features (pupil coordinates) and targets (world coordinates) from the data.
    3. Splits the data into training and testing sets, separately for the x and y coordinates of the world coordinates.
    4. Defines an SVR model with an RBF kernel and prepares a grid of hyperparameters for grid search cross-validation.
    5. Performs grid search to find the optimal hyperparameters for predicting the x and y coordinates of the world coordinates.
    6. Displays the results of the grid search including the best estimators found.
    7. Makes predictions on the test set using the best estimators found through grid search.
    8. Visualizes the results using scatter and line plots to show:
        - A scatter plot of the ground truth versus the predictions.
        - Line plots of the true and predicted x and y coordinates over the data index.
    9. Computes and prints various regression metrics including MSE, R2 score, MAE, and explained variance score to evaluate the performance of the models.
    10. Saves the trained models to the disk for future use.

    Returns:
        Tuple[GridSearchCV, GridSearchCV]: The trained GridSearchCV instances for the x and y coordinate predictions.

    Note:
        - The input CSV file should have the columns: ['cx_left', 'cx_right', 'cy_left', 'cy_right', 'cx_world', 'cy_world'].
        - The script saves the plots to the disk with names 'prediction.png', 'predictionX.png', and 'predictionY.png'.
        - The script saves the trained models to the disk with names 'finalized_model_svr_cx.sav' and 'finalized_model_svr_cy.sav'.
    """

    warnings.filterwarnings("ignore", category=DeprecationWarning, module='sklearn.svm')

    data = pd.read_csv('pupil_coord.csv')

    pupil_coord = data[['cx_left', 'cx_right', 'cy_left', 'cy_right']]
    target_world = data[['cx_world', 'cy_world']]
    
    # Stratified split of the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        pupil_coord, target_world, test_size=0.2, random_state=1
    )
    
    X_train_cx = X_train.copy()
    y_train_cx = y_train[['cx_world']]
    
    X_train_cy = X_train.copy()
    y_train_cy = y_train[['cy_world']]
    
    X_test_cx = X_test.copy()
    y_test_cx = y_test[['cx_world']]
    
    X_test_cy = X_test.copy()
    y_test_cy = y_test[['cy_world']]
    
    # Define SVR with rbf kernel
    my_svr = SVR(gamma="scale", kernel='rbf')

    # Define parameters for the Grid Search
    parameters_cx = {'C': [1150, 1000, 1100, 1500], 'epsilon': [0.5, 0.7, 0.22, 0.67]}
    parameters_cy = {'C': [1150, 1300, 1100, 1500], 'epsilon': [0.5, 0.7, 0.18, 0.67]}
    
    # Find best parameters for cx. 5-fold cross-validation
    clf_svr_cx = GridSearchCV(my_svr, parameters_cx, cv=5)
    clf_svr_cx.fit(X_train_cx, y_train_cx.values.ravel())
      
    # Find best parameters for cy. 5-fold cross-validation
    clf_svr_cy = GridSearchCV(my_svr, parameters_cy, cv=5)
    clf_svr_cy.fit(X_train_cy, y_train_cy.values.ravel())
    
    # Show the best estimator for cx    
    print(pd.DataFrame(clf_svr_cx.cv_results_)[['mean_test_score', 'std_test_score', 'params']])    
    print('Best SVR for cx: ', clf_svr_cx.best_estimator_)
    
    # Show the best estimator for cy    
    print(pd.DataFrame(clf_svr_cy.cv_results_)[['mean_test_score', 'std_test_score', 'params']])
    print('Best SVR for cy: ', clf_svr_cy.best_estimator_)
    
    # Prediction on the TEST sets
    pred_svr_cx = clf_svr_cx.predict(X_test_cx)
    pred_svr_cy = clf_svr_cy.predict(X_test_cy)
    
    fig = plt.figure(figsize=(16,8))
    plt.rcParams.update({'font.size': 30})
    plt.scatter('cx_world','cy_world', data = y_test, c='blue',marker='o', label = "Ground truth", s=30) 
    plt.scatter(pred_svr_cx, pred_svr_cy, c='red',marker='o', label = "Prediction with rbf kernel", s=30, )
    plt.xlabel("px")
    plt.ylabel("py")
    plt.title("epsilon-Support Vector Regression")
    plt.legend()
    fig.savefig('prediction.png', dpi = 300)

    fig = plt.figure(figsize=(20,11))
    plot_cx = y_test_cx.copy()
    plot_cx['predictionX'] = pred_svr_cx
    plot_cx.sort_index(inplace = True)
    plt.plot(plot_cx[['cx_world']], marker='D',c='b', label = "Ground truth", ms=10)
    plt.plot(plot_cx[['predictionX']], marker='o',c='r', label = "Prediction with rbf kernel", ms=10, linewidth = 2)
    plt.title("epsilon-Support Vector Regression for X coordinate", fontsize=30, fontweight="bold")
    plt.xlabel("data", fontsize=30)
    plt.ylabel("Coordinate px", fontsize=30)
    plt.legend(prop={'size': 30})
    plt.ylim([400,1400])
    plt.xlim([0,900])
    fig.savefig('predictionX.png', dpi = 300)
    
    fig = plt.figure(figsize=(20,11))
    plot_cy = y_test_cy.copy()
    plot_cy['predictionY'] = pred_svr_cy
    plot_cy.sort_index(inplace = True)
    plt.plot(plot_cy[['cy_world']], marker='D',c='b', label = "Ground truth", ms=10)
    plt.plot(plot_cy[['predictionY']], marker='o',c='r', label = "Prediction with rbf kernel", ms=10, linewidth = 2)
    plt.title("epsilon-Support Vector Regression for Y coordinate", fontsize=30, fontweight="bold")
    plt.xlabel("data", fontsize=30)
    plt.ylabel("Coordinate py", fontsize=30)
    plt.legend(prop={'size': 30})
    plt.ylim([0,900])
    plt.xlim([0,900])
    fig.savefig('predictionY.png', dpi = 300)
    plt.show()
    
        
    print('MSE cx: ',mean_squared_error(y_test_cx, pred_svr_cx))
    print('MSE cy: ',mean_squared_error(y_test_cy, pred_svr_cy))
    
    # Best possible score is 1.0
    print('r2 cx: ',r2_score(y_test_cx, pred_svr_cx))
    print('r2 cy: ',r2_score(y_test_cy, pred_svr_cy))
    
    print('MAE cx: ',mean_absolute_error(y_test_cx, pred_svr_cx))
    print('MAE cy: ',mean_absolute_error(y_test_cy, pred_svr_cy))
    
    # Best possible score is 1.0
    print('Explained variance cx: ',explained_variance_score(y_test_cx, pred_svr_cx))
    print('Explained variance cy: ',explained_variance_score(y_test_cy, pred_svr_cy))

    # Save the model
    filename_cx = 'finalized_model_svr_cx.sav'
    filename_cy = 'finalized_model_svr_cy.sav'
    dump(clf_svr_cx, filename_cx)
    dump(clf_svr_cy, filename_cy)

    return clf_svr_cx, clf_svr_cy


if __name__ == '__main__':
    my_svr_function()
