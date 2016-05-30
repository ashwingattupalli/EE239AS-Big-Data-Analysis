import utility
import logging as logger

logger.basicConfig(level=logger.INFO,format='> %(message)s')

def question_1():
    logger.info("EXECUTING: QUESTION 1 - Least Square Factorization")
    data, R_mat, W_mat = utility.load_dataset()  # load the dataset
    utility.compute_least_squared_error(R_mat, W_mat)  # compute the least squared error without regularization

def question_2():
    logger.info("EXECUTING: QUESTION 2 - 10-fold Cross-Validation on Recommendation System")
    data, R_mat, _ = utility.load_dataset()  # load the dataset
    utility.perform_cross_validation(data, R_mat)  # perform cross validation

def question_3():
    logger.info("EXECUTING: QUESTION 3 - Recommendation Systems with threshold limits")
    data, R_mat, _ = utility.load_dataset()  # load the dataset
    utility.perform_cross_validation(data, R_mat, threshold=True)  # perform cross validation with threshold limit

def question_4():
    logger.info("EXECUTING: QUESTION 4 - Recommendation Systems with Regularization")
    data, R_mat, W_mat = utility.load_dataset()
    logger.info("R & W Matrix - Interchanged")
    utility.compute_least_squared_error(W_mat, R_mat)  # interchange R & W matrix

    logger.info("R & W Matrix - Regularization")
    utility.compute_least_squared_error(R_mat, W_mat,regularized=True)

def question_5():
    logger.info("EXECUTING: QUESTION 5 - Recommendation System")
    data, R_mat, W_mat = utility.load_dataset()  # load the dataset
    utility.perform_cross_validation_question5(data,R_mat,W_mat)


