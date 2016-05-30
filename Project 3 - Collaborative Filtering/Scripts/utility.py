import numpy as np
import pandas as pd
import logging as logger
from scipy import linalg
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold
from numpy import linalg as LA
import cPickle
from sklearn.metrics import auc, roc_curve

logger.basicConfig(level=logger.INFO,format='> %(message)s')

def load_dataset():
    """
    :return: movie lens dataset separated in R & W matrices
    """
    logger.info("Loading Movie Lens Dataset.")

    data = pd.read_csv('../Dataset/ml-100k/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])

    # R_mat containing user ratings with users on rows and movies on columns. Missing values are filled with 0
    R_mat = data.pivot_table(index=['user_id'], columns=['movie_id'], values='rating', fill_value=0)
    W_mat = R_mat.copy()
    W_mat[W_mat > 0] = 1

    logger.info("Movies Lens Dataset Loaded")

    return data.as_matrix(), R_mat.as_matrix(), W_mat.as_matrix()


def nmf(R_mat, k, mask, lambda_reg=0, max_iter=100):
    """
    :param R_mat: matrix of user ratings with users on rows and movies on columns
    :param k: feature
    :param lambda_reg: regularization term
    :param max_iter: maximum number of iteration
    :return: R_mat = U * V decomposes
    """
    eps = 1e-5

    # mask = np.sign(R_mat)
    rows, columns = R_mat.shape
    U = np.random.rand(rows, k)
    U = np.maximum(U, eps)

    V = linalg.lstsq(U, R_mat)[0]
    V = np.maximum(V, eps)

    masked_X = mask * R_mat

    for i in range(1, max_iter + 1):

        top = np.dot(masked_X, V.T)
        bottom = (np.add(np.dot((mask * np.dot(U, V)), V.T), lambda_reg * U)) + eps
        U *= top / bottom
        U = np.maximum(U, eps)

        top = np.dot(U.T, masked_X)
        bottom = np.add(np.dot(U.T, mask * np.dot(U, V)), lambda_reg * V) + eps
        V *= top / bottom
        V = np.maximum(V, eps)

    return U,V


def calculate_error(predicted, actual, weights):
    """
    :return: sum of squared error
    """
    error = actual - predicted
    squared_error = np.multiply(error,error)
    squared_error = np.multiply(weights, squared_error)
    sum_squared_error = sum(sum(squared_error))

    return sum_squared_error


def plot_ROC_curve(predicted, actual, x, _lambda):
    """
    :return: plot roc curve for differenct values of threshold
    """

    tp = 0  # true positive
    fp = 0  # false positive
    fn = 0  # false negative

    threshold_value = np.arange(1,6,1)
    precision = np.zeros(len(threshold_value))
    recall = np.zeros(len(threshold_value))

    for k, t in enumerate(threshold_value):
        tp = np.sum(actual[predicted >= t] >= t)
        fp = np.sum(actual[predicted >= t] < t)
        fn = np.sum(actual[predicted < t] >= t)

        precision[k] = tp / float(tp+fp)  # calculating precision
        recall[k] = tp / float(tp+fn)  # calculating recall

    plt.figure(1)
    plt.ylabel('Recall')
    plt.xlabel('Precision')
    plt.title('ROC Curve k={0} lambda={1}'.format(x,_lambda))
    plt.scatter(precision, recall, s=60, marker='o')
    plt.plot(precision,recall)
    plt.show()

def matrix_factorization(R_mat, W_mat, k, regularized=False, _lambda=0):
    """
    :param R_mat: matrix of user ratings with users on rows and movies on columns
    :param W_mat: weight matrix
    :param k: features
    :return: sum squared error for the corresponding model
    """
    U, V = nmf(R_mat, k, W_mat, lambda_reg=_lambda)

    R_mat_predicted = np.dot(U, V)

    if regularized:
        plot_ROC_curve(R_mat_predicted,R_mat, k, _lambda)

    sum_squared_error = calculate_error(R_mat_predicted, R_mat, W_mat)

    return sum_squared_error

def compute_least_squared_error(R_mat, W_mat, features=(10, 50, 100), alphas=(0.01, 0.1, 1), regularized=False):
    """
    :param R_mat: matrix of user ratings with users on rows and movies on columns
    :param W_mat: weight matrix
    :param features: factorization features
    :param alphas: regularization term
    :param regularized: to regularize or not (boolean)
    :return: total least squared error
    """
    logger.info("Computing Least Squared Error.")

    for k in features:
        if regularized:
            for a in alphas:
                sum_squared_error = matrix_factorization(R_mat, W_mat, k, _lambda=a, regularized=True)
                logger.info('Least Squared Error of Rated Movies (k = {0} & lambda = {1}): {2}'.format(k, a, sum_squared_error))
        else:
            sum_squared_error = matrix_factorization(R_mat, W_mat, k)
            logger.info('Least Squared Error of Rated Movies (k = {0}): {1}'.format(k, sum_squared_error))

def perform_cross_validation(data, R_mat, threshold=False, n_folds = 10):
    """
    :param data: movie lens dataset
    :param threshold: perform threshold cross validation (for question 3)
    :return: cross validation average error / precision & recall
    """
    logger.info("Performing Cross Validation")
    test_length = len(data) / n_folds  # length of test dataset
    train_length = test_length * (n_folds - 1)  # length of train dataset

    # variables to store result
    average_error = np.zeros(n_folds)
    threshold_value = np.arange(1,6,1)
    precision = np.zeros((len(threshold_value),n_folds))
    recall = np.zeros((len(threshold_value),n_folds))

    kf = KFold(n=len(data), n_folds=10, shuffle=True)

    for train_index, test_index in kf:
        logger.info("Performing Fold - {0}".format(10 - n_folds + 1))

        train_data = data[train_index]
        test_data = data[test_index]

        W_mat = np.zeros([max(data[:,0]),max(data[:,1])])

        for j in range(train_length):
            W_mat[train_data[j][0] - 1, train_data[j][1] - 1] = 1

        U,V = nmf(R_mat, 100, W_mat, lambda_reg=0)

        R_predicted = np.dot(U,V)
        errors = np.zeros(test_length)

        if not threshold:
            for j in range(test_length):
                errors[j] = abs(R_predicted[test_data[j][0]-1, test_data[j][1]-1] - test_data[j][2])
            average_error[10 - n_folds] = np.mean(errors)
        else:
            # get precision and recall
            for k, t in enumerate(threshold_value):

                tp = 0  # true positive
                fp = 0  # false positive
                fn = 0  # false negative

                # check test dataset
                for j in range(test_length):
                    uid_test = test_data[j][0] - 1
                    mid_test = test_data[j][1] - 1
                    rating_test = test_data[j][2]

                    if (R_predicted[uid_test,mid_test] >= t):
                        if rating_test >= t:
                            tp += 1
                        else:
                            fp += 1
                    elif rating_test >= t:
                        fn += 1

                precision[k,10-n_folds] = tp / float(tp+fp)  # calculating precision
                recall[k,10-n_folds] = tp / float(tp+fn)  # calculating recall

        n_folds -= 1

    if not threshold:
        logger.info("Cross Validation Error over all folds : {}".format(average_error))
        logger.info("Cross Validation Average Error : {}".format(np.mean(average_error)))
        logger.info("Highest Cross Validation Error : {}".format(max(average_error)))
        logger.info("Lowest Cross Validation Error : {}".format(min(average_error)))

    else:
        avg_precision = np.mean(precision,axis=1)
        avg_recall = np.mean(recall,axis=1)
        logger.info("Precision: {}".format(avg_precision))
        logger.info("Recall: {}".format(avg_recall))

        # Plot Precision vs. Recall
        plt.ylabel('Recall')
        plt.xlabel('Precision')
        plt.title('ROC Curves')
        plt.scatter(avg_precision, avg_recall, s=60, marker='o')
        plt.plot(avg_precision,avg_recall)
        plt.show()


def perform_cross_validation_question5(data, R_mat, W_mat):
    """
    :return: perform cross validation for top movies and get hit miss rate
    """
    L = 5
    n_folds = 10

    test_length = len(data) / n_folds  # length of test dataset
    top_movies_order = []
    kf = KFold(n=len(data), n_folds=10, shuffle=True)

    hit_cross_val = []
    miss_cross_val = []
    total_cross_val = []
    precision_cross_val = []

    for train_index, test_index in kf:
        logger.info("Performing Fold - {0}".format(10 - n_folds + 1))
        test_data = data[test_index]

        R_mat_train = W_mat
        W_mat_train = R_mat

        for j in range(test_length):
            W_mat_train[test_data[j][0] - 1, test_data[j][1] - 1] = 0

        U,V = nmf(R_mat_train,100, W_mat_train, lambda_reg=0.01)
        R_predicted = 5 * np.dot(U,V)

        R_predicted[R_mat_train == 0] = -1  # ignore data points without ratings

        for i in range(max(data[:,0])):
            user_ratings = R_predicted[i]
            top_movies = user_ratings.argsort()[-max(data[:,1]):][::-1]
            top_movies_order.append(top_movies)

        threshold = 3

        hit_val=[]
        miss_val=[]
        total_val=[]
        precision_val=[]

        for l in range(1,(L+1)):
            hit = 0  # correct
            miss = 0 # incorrect
            total = 0
            precision = 0
            for i in range(max(data[:,0])):
                rec_indices = R_predicted[i,0:l] # top L movies for every row
                for j in range(len(rec_indices)):
                    rating = R_predicted[i][rec_indices[j]]
                    if (rating < 0):
                        continue
                    if (rating > threshold): # user likes the movie
                        hit = hit + 1
                        total = total + 1
                        precision += 1
                    else:
                        miss = miss + 1
                        total = total + 1

            precision_val.append(precision/float(total))
            hit_val.append(hit)
            total_val.append(total)
            miss_val.append(miss)

        hit_cross_val.append(hit_val)
        miss_cross_val.append(miss_val)
        total_cross_val.append(total_val)
        precision_cross_val.append(precision_val)
        n_folds -= 1

    precision = np.sum(precision_cross_val,axis=0)
    hits = np.sum(hit_cross_val,axis=0)
    miss = np.sum(miss_cross_val,axis=0)
    total = np.sum(total_cross_val,axis=0)

    hits = hits / (total.astype(float))
    miss = miss / (total.astype(float))
    precision = precision / 10.0

    logger.info("Precision : {0}".format(precision))

    plt.figure(1)
    plt.ylabel('Hit rate')
    plt.xlabel('False Alarm')
    plt.title('Hit vs Miss')
    plt.scatter(miss, hits, s=60, marker='o')
    plt.plot(miss,hits)
    plt.show()