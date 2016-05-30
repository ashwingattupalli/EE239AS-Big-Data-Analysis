import os
import math

import time

from pysam.cvcf import defaultdict
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

import utility
import operator
import numpy as np
import pylab as pl
import copy
from sklearn import svm
import logging as logger
from collections import Counter
from sklearn import cross_validation
from sklearn.feature_extraction import text
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

logger.basicConfig(level=logger.INFO,format='%(message)s')

categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
              'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']

train_dataset, test_dataset = utility.load_dataset(categories)  # load CT & RA dataset

# combine the classes
processed_train_dataset = copy.deepcopy(train_dataset)
processed_test_dataset = copy.deepcopy(test_dataset)

def combine_classes():
    for i,j in enumerate(processed_train_dataset.target):
        if j >= 0 and j < 4:
            processed_train_dataset.target[i] = 0
        else:
            processed_train_dataset.target[i] = 1

    for i,j in enumerate(processed_test_dataset.target):
        if j >= 0 and j < 4:
            processed_test_dataset.target[i] = 0
        else:
            processed_test_dataset.target[i] = 1

    processed_train_dataset.target_names = ['Computer Technology', 'Recreational Activity']
    processed_test_dataset.target_names = ['Computer Technology', 'Recreational Activity']

combine_classes()

train_all_dataset, test_all_dataset = utility.load_dataset([])  # load entire dataset

category_CT = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware']  # CT categories
category_RA = ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']  # RA categories


def question_a():
    logger.info("EXECUTING: QUESTION A")
    logger.info("Plotting histogram of the number of documents per topic (Training Dataset)")

    count_train = {}
    count_test = {}

    # count the number of documents for each topic name in training dataset
    for record in xrange(len(train_dataset.target)):
        if train_dataset.target_names[train_dataset.target[record]] in count_train:
            count_train[train_dataset.target_names[train_dataset.target[record]]] += 1
        else:
            count_train[train_dataset.target_names[train_dataset.target[record]]]= 1

    # count the number of documents for each topic name in testing dataset
    for record in xrange(len(test_dataset.target)):
        if test_dataset.target_names[test_dataset.target[record]] in count_test:
            count_test[test_dataset.target_names[test_dataset.target[record]]] += 1
        else:
            count_test[test_dataset.target_names[test_dataset.target[record]]]= 1

    logger.info("Histogram plotted")

    # plot histogram for number of documents vs. topic name
    pl.figure(1)
    pl.ylabel('Topic Name')
    jet = pl.get_cmap('jet')
    pl.xlabel('Number of Topics')
    pos = pl.arange(len(count_train.keys())) + 0.5
    pl.title('Histogram of Number of Documents Per Topic')
    pl.yticks(pos, count_train.keys())
    pl.barh(pos, count_train.values(), align='center', color=jet(np.linspace(0, 1.0, len(count_train))))

    # count number of documents in CT and RA classes
    train_CT, train_RA, test_CT, test_RA = 0,0,0,0

    for i,j in zip(category_CT,category_RA):
        train_CT += count_train[i]
        train_RA += count_train[j]

        test_CT += count_test[i]
        test_RA += count_test[j]

    logger.info("TRAINING DATASET")
    logger.info("Number of Documents in Computer Technology : {}".format(train_CT))
    logger.info("Number of Documents in Recreational Activity : {}".format(train_RA))

    logger.info("TESTING DATASET")
    logger.info("Number of Documents in Computer Technology : {}".format(test_CT))
    logger.info("Number of Documents in Recreational Activity : {}".format(test_RA))

    pl.show()

def question_b():
    logger.info("EXECUTING: QUESTION B")
    utility.model_text_data(processed_train_dataset, processed_test_dataset)  # perform modelling with the dataset


def question_c():
    logger.info("EXECUTING: QUESTION C")
    # get training data for every category and terms with their frequency for every class
    all_categories = train_all_dataset.target_names

    freq_words_all_categories=[]
    words_all_categories =[]
    all_data_category = []
    words_in_classes = defaultdict(list)

    find_for_classes_list = [train_all_dataset.target_names.index("comp.sys.ibm.pc.hardware"),
                             train_all_dataset.target_names.index("comp.sys.mac.hardware"),
                             train_all_dataset.target_names.index("misc.forsale"),
                             train_all_dataset.target_names.index("soc.religion.christian")]

    logger.info("Collecting data for each category")

    for category in all_categories:
        train_category = utility.load_dataset([category])[0]
        data_category = train_category.data
        temp = ''
        for document in data_category:
            temp += ' ' + document
        all_data_category.append(temp)

    logger.info("Cleaning Data and Forming Frequency List for each Class")

    # pre-process all the docs
    for data,pos in zip(all_data_category,range(len(all_data_category))):
        logger.info("Forming Frequency List for Class: {}".format(train_all_dataset.target_names[pos]))
        processed_data = utility.preprocess_data(data)
        count = Counter(processed_data)
        freq_words_all_categories.append(count)
        unique_words = set(processed_data)
        words_all_categories.append(list(unique_words))
        for word in unique_words:
            words_in_classes[word].append(train_all_dataset.target_names[pos])

    # calculating tf-icf
    for category in find_for_classes_list:
        logger.info("Fetching top 10 significant terms for class: {}".format(train_all_dataset.target_names[category]))
        terms_of_class = words_all_categories[category]
        freq_of_all_terms = freq_words_all_categories[category]
        number_of_terms = len(terms_of_class)
        tficf = {}
        for each_term in range(number_of_terms):
            term= terms_of_class[each_term] # term for which we are finding tf-icf
            frequency_of_term = freq_of_all_terms.get(term)
            number_of_class_with_term = len(words_in_classes[term]) # number of classes with term t
            # tficf for term t
            calc = 0.5 + ((0.5 * frequency_of_term/number_of_terms) * math.log(len(train_all_dataset.target_names) / number_of_class_with_term))
            tficf[term]=calc

        # print top 10 significant term for this class
        significant_terms = dict(sorted(tficf.iteritems(), key=operator.itemgetter(1), reverse=True)[:10]) #get 10 significant terms

        logger.info(significant_terms.keys())

def question_d():
    logger.info("EXECUTING: QUESTION D")
    train_idf, test_idf = utility.model_text_data(processed_train_dataset, processed_test_dataset)  # perform modelling with the dataset
    _, _ = utility.apply_lsi(train_idf, test_idf) # apply LSI to the TFxIDF matrices


def perform_classification(classifier, algo, cv=False, roc=True):
    if os.path.isfile("../Dataset/Train_LSI.pkl") and os.path.isfile("../Dataset/Test_LSI.pkl"):  # load pickle file if it exists
        train_lsi, test_lsi = utility.apply_lsi([], [])
    else:
        train_idf, test_idf = utility.model_text_data(processed_train_dataset, processed_test_dataset)
        train_lsi, test_lsi = utility.apply_lsi(train_idf, test_idf)

    if cv: # SVM Cross Validation Testing
        logger.info("Calculating Best Parameter Value")
        C = [-3,-2,-1,0,1,2,3]
        best_scores = []

        for i in C:
            logger.info("Parameter Value: {}".format(i))
            clf = svm.SVC(kernel='linear', C = 10**C[i])
            scores = cross_validation.cross_val_score(clf, train_lsi,processed_train_dataset.target,cv=5)
            best_scores.append(np.mean(scores))

        logger.info(best_scores)
        logger.info("Best Parameter Value: {}".format(best_scores.index(max(best_scores))))
        classifier = svm.SVC(kernel='linear',C=10**best_scores.index(max(best_scores)))

    logger.info("Training {0} Classifier ".format(algo))
    classifier.fit(train_lsi, processed_train_dataset.target)  # fit the training data
    logger.info("Testing {0} Classifier".format(algo))  # predict the testing data
    test_predicted = classifier.predict(test_lsi)

    utility.calculate_statistics(processed_test_dataset.target, test_predicted)  # calculate classifier statistics

    if roc: # plot ROC curve
        utility.plot_ROC(processed_test_dataset.target, test_predicted, algo)


def question_e():
    logger.info("EXECUTING: QUESTION E")
    logger.info("SVM Classifier")

    clf = svm.SVC(kernel='linear',probability=True)  # svm classifier
    perform_classification(clf, "SVM")

def question_f():
    logger.info("EXECUTING: QUESTION F")
    logger.info("SVM Classifier using Cross Validation")
    logger.info("Best Parameter Value: 0 (Pre-calculated)")

    clf = svm.SVC(kernel='linear', C=10**0)  # cross validated svm classifier
    perform_classification(clf,algo="Cross Validated SVM",cv=False,roc=False)

def question_g():
    logger.info("EXECUTING: QUESTION G")
    logger.info("Naive Bayes Classifier")

    clf = GaussianNB()  # bernoulli naive bayes
    perform_classification(clf, "Naive Bayes")

def question_h():
    logger.info("EXECUTING: QUESTION H")
    logger.info("Logistic Regression")

    clf = LogisticRegression(C=10)  # logistic regression
    perform_classification(clf, "Logistic Regression")

def question_i():
    logger.info("EXECUTING: QUESTION I")
    logger.info("Multi-Class Classification")

    category = ['comp.sys.ibm.pc.hardware','comp.sys.mac.hardware','misc.forsale','soc.religion.christian']
    train, test = utility.load_dataset(category)

    logger.info("Processing Training Dataset")
    for data,pos in zip(train.data,range(len(train.data))):
        processedData = utility.preprocess_data(data)
        train.data[pos] = ' '.join(processedData)

    logger.info("Processing Testing Dataset")
    for data,pos in zip(test.data,range(len(test.data))):
        processedData = utility.preprocess_data(data)
        test.data[pos] = ' '.join(processedData)

    logger.info("Creating TFxIDF Vector Representations")

    stop_words = text.ENGLISH_STOP_WORDS  # omit stop words

    # using CountVectorizer and TFxIDF Transformer
    count_vect = CountVectorizer(stop_words=stop_words, lowercase=True)
    train_counts = count_vect.fit_transform(train.data)
    test_counts = count_vect.transform(test.data)
    tfidf_transformer = TfidfTransformer(norm='l2', sublinear_tf=True)
    train_idf = tfidf_transformer.fit_transform(train_counts)
    test_idf = tfidf_transformer.transform(test_counts)

    logger.info("Performing LSI on TFxIDF Matrices")
    # apply LSI to TDxIDF matrices
    svd = TruncatedSVD(n_components=50)
    train_lsi = svd.fit_transform(train_idf)
    test_lsi = svd.transform(test_idf)

    logger.info("TFxIDF Matrices Transformed")

    logger.info("Size of Transformed Training Dataset: {0}".format(train_lsi.shape))
    logger.info("Size of Transformed Testing Dataset: {0}".format(test_lsi.shape))

    clf_list = [OneVsOneClassifier(GaussianNB()), OneVsOneClassifier(svm.SVC(kernel='linear')), OneVsRestClassifier(GaussianNB()), OneVsRestClassifier(svm.SVC(kernel='linear'))]
    clf_name = ['OneVsOneClassifier Naive Bayes', 'OneVsOneClassifier SVM','OneVsRestClassifier Naive Bayes', 'OneVsRestClassifier SVM']

    # perform classification
    for clf,clf_n in zip(clf_list,clf_name):
        logger.info("Training {0} Classifier ".format(clf_n))
        clf.fit(train_lsi, train.target)
        logger.info("Testing {0} Classifier".format(clf_n))
        test_predicted = clf.predict(test_lsi)
        utility.calculate_statistics(test.target, test_predicted)


