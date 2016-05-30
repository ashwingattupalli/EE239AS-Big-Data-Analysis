import os
import re
import string
import cPickle
import numpy as np
import pylab as pl
import logging as logger
import sklearn.metrics as smet
from nltk import SnowballStemmer
from sklearn.metrics import roc_curve, auc
from sklearn.feature_extraction import text
from sklearn.decomposition import TruncatedSVD
from sklearn.datasets import fetch_20newsgroups
from sklearn.preprocessing import label_binarize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer

logger.basicConfig(level=logger.INFO,format='> %(message)s')

# categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
#              'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']

categories = ['Computer Technology', 'Recreational Activity']

def load_dataset(category_list):
    """
    :return: Load the 20_newsgroup dataset depending on category_list.
             If [] provided return everything
    """

    if category_list == []:  # read all categories from news20 dataset
        train = fetch_20newsgroups(subset='train', shuffle=True, random_state=42)
        test = fetch_20newsgroups(subset='test', shuffle=True, random_state=42)
    else:            # read only computer technology & recreational activity categories
        train = fetch_20newsgroups(subset='train',  shuffle=True, random_state=42, categories=category_list)
        test = fetch_20newsgroups(subset='test', shuffle=True, random_state=42,  categories=category_list)

    return train, test


def clean_dataset(data, load_data=True, dataset='Train'):
    """
    :param data: data required to be pre-processed
    :param load_data: if True check if pickle object saved on drive or not
    :param dataset: dataset being processed
    :return: pre-processed data
    """

    if os.path.isfile("../Dataset/Pre_Processed_{0}.pkl".format(dataset)):  # load pickle file if it exists
        logger.info("Pre-processed Dataset located at ../Dataset/Pre_Processed_{0}.pkl. Loading.".format(dataset))
        data = cPickle.load(open("../Dataset/Pre_Processed_{0}.pkl".format(dataset), "r"))
    else:
        for text,pos in zip(data,range(len(data))):
            stemmed_data = preprocess_data(text) # pre-process the docs
            data[pos] = ' '.join(stemmed_data)  # combine all stemmed words

        logger.info("Dumping pickle file at ../Dataset/Pre_Processed_{0}.pkl".format(dataset))
        cPickle.dump(data,open("../Dataset/Pre_Processed_{0}.pkl".format(dataset), "wb"))

    return data


def preprocess_data(data):
    """
    :param data: data to be pre-processed
    :return: pre-processed data
    """

    stemmer2 = SnowballStemmer("english") # for removing stem words
    stop_words = text.ENGLISH_STOP_WORDS  # omit stop words

    temp = data
    temp = re.sub("[,.-:/()?{}*$#&]"," ",temp)  # remove all symbols
    temp = "".join([ch for ch in temp if ch not in string.punctuation])  # remove all punctuation
    temp = "".join(ch for ch in temp if ord(ch) < 128)  # remove all non-ascii characters
    temp = temp.lower() # convert to lowercase
    words = temp.split()
    no_stop_words = [w for w in words if not w in stop_words]  # stemming of words
    stemmed_data = [stemmer2.stem(plural) for plural in no_stop_words]

    return stemmed_data


def model_text_data(train, test):
    """
    :param train: train dataset data
    :param test: test dataset data
    :return: TFxIDF Matrices for both training and testing dataset
    """
    logger.info("Preprocessing dataset - Training & Testing Dataset")

    train.data = clean_dataset(train.data)  # training dataset
    test.data = clean_dataset(test.data,dataset='Test')  # testing dataset

    logger.info("Creating TFxIDF Vector Representations")

    stop_words = text.ENGLISH_STOP_WORDS  # omit stop words

    '''
    # using TfidfVectorizer
    vectorizer = TfidfVectorizer(min_df=1, stop_words=stop_words)
    train_idf = vectorizer.fit_transform(train.data[:])  # fit pre-processed dataset to vectorizer
    test_idf = vectorizer.transform(test.data[:])
    '''

    # using CountVectorizer and TFxIDF Transformer
    count_vect = CountVectorizer(stop_words=stop_words, lowercase=True)
    train_counts = count_vect.fit_transform(train.data)
    test_counts = count_vect.transform(test.data)
    tfidf_transformer = TfidfTransformer(norm='l2', sublinear_tf=True)
    train_idf = tfidf_transformer.fit_transform(train_counts)
    test_idf = tfidf_transformer.transform(test_counts)


    logger.info("TFxIDF Matrices Created")
    logger.info("Number of Terms Extracted : {}".format(train_idf.shape[1]))

    return train_idf.toarray(),test_idf.toarray()


def apply_lsi(train_data, test_data):
    """
    :param train_data: train dataset data
    :param test_data: testing dataset data
    :return: apply LSI on TFxIDF matrices and return transformed matrices
    """

    logger.info("Performing LSI on TFxIDF Matrices")

    if os.path.isfile("../Dataset/Train_LSI.pkl") and os.path.isfile("../Dataset/Test_LSI.pkl"):  # load pickle file if it exists
        logger.info("TFxIDF Matrices located at ../Dataset. Loading.")
        train_lsi = cPickle.load(open("../Dataset/Train_LSI.pkl", "r"))
        test_lsi = cPickle.load(open("../Dataset/Test_LSI.pkl", "r"))

    else:
        svd = TruncatedSVD(n_components=50)  # LSI applied with k=50
        train_lsi = svd.fit_transform(train_data)
        test_lsi = svd.transform(test_data)

        logger.info("TFxIDF Matrices Transformed")
        logger.info("Dumping TFxLSI Matrices to ../Dataset/")
        cPickle.dump(train_lsi,open("../Dataset/Train_LSI.pkl", "wb"))
        cPickle.dump(test_lsi,open("../Dataset/Test_LSI.pkl", "wb"))

    logger.info("Size of Transformed Training Dataset: {0}".format(train_lsi.shape))
    logger.info("Size of Transformed Testing Dataset: {0}".format(test_lsi.shape))

    return train_lsi, test_lsi

def calculate_statistics(target, predicted):
    """
    :param target: target class
    :param predicted: predicted class
    :return: statistics of classifier
    """

    accuracy = smet.accuracy_score(target,predicted)
    precision = smet.precision_score(target, predicted, average='macro')
    recall = smet.recall_score(target, predicted, average='macro')
    confusion_matrix = smet.confusion_matrix(target,predicted)

    logger.info("Statistics for Classifier:")
    logger.info("Accuracy : {0}".format(accuracy * 100))
    logger.info("Precision : {0}".format(precision * 100))
    logger.info("Recall : {0}".format(recall * 100))
    logger.info("Confusion Matrix : \n{0}".format(confusion_matrix))

    return True


def plot_ROC(test_target, test_predicted, algo):
    """
    :param test_target: target class numeric
    :param test_predicted: predicted class
    :param algo: classifier name
    :return: ROC curve for the given classifier and for given names
    """

    logger.info("ROC Curve for Categories: {}".format(categories))
    # compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    fpr, tpr, _ = roc_curve(test_target, test_predicted)
    roc_auc = auc(fpr, tpr)

    # plot curve
    pl.figure(1)
    pl.plot(fpr, tpr, label='ROC curve(area = {0:0.4f})'
                                   ''.format(roc_auc))

    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.05])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('ROC Curves for {0} Classifier'.format(algo))
    pl.legend(loc="lower right")
    pl.show()

    return True
