import json
from utility import *
import logging as logger

logger.basicConfig(level=logger.INFO, format='> %(message)s')

input_files = ['gopatriots', 'gohawks', 'nfl', 'patriots', 'sb49', 'superbowl']  # tweet-data file names

logger.info("EXECUTING: QUESTION 4 - CROSS VALIDATION LINEAR REGRESSION")

for file_name in input_files:
    logger.info("Working on #{}".format(file_name))
    tweets = open('../Dataset/tweet_data/tweets_#{0}.txt'.format(file_name), 'rb')
    first_tweet = json.loads(tweets.readline())
    start_time = first_tweet.get('firstpost_date')

    # features for model construction
    features = extra_feature_dict()
    flag_between = True  # flag to set new window start point
    flag_after = True  # flag to set new window start point
    current_window = 1
    end_time_of_window = start_time + current_window * 3600
    tweets.seek(0, 0)

    # store data
    tweet_features_before = []
    tweet_features_between = []
    tweet_features_after = []
    tweet_class_before = []
    tweet_class_between = []
    tweet_class_after = []

    logger.info("Extracting features from tweets")

    for tweet in tweets:
        tweet_data = json.loads(tweet)
        end_time = tweet_data.get('firstpost_date')

        '''
        features : retweets, followers, max_followers, impressions, favorite_count,
                   ranking_score, hour_of_day, number_of_users, number_of_long tweets
        '''

        if end_time < 1422777600:  # tweeted Before Feb. 1, 8:00 a.m.
            if end_time < end_time_of_window:
                features = calculate_statistic(features, tweet_data)
            else:
                extracted_features = get_features(features)
                tweet_class_before.append(extracted_features[0])
                tweet_features_before.append(extracted_features[1:])
                features = reset_features_dict()
                features = calculate_statistic(features, tweet_data)
                current_window += 1
                end_time_of_window = start_time + current_window * 3600

        elif (end_time > 1422777600) and (end_time < 1422820800):  # tweeted Between Feb. 1, 8:00 a.m to 8 p.m
            if flag_between:  # reset start_time
                flag_between = False
                current_window = 1
                start_time = end_time
                end_time_of_window = start_time + (current_window * 3600)

            if end_time < end_time_of_window:
                features = calculate_statistic(features, tweet_data)
            else:
                extracted_features = get_features(features)
                tweet_class_between.append(extracted_features[0])
                tweet_features_between.append(extracted_features[1:])
                features = reset_features_dict()
                features = calculate_statistic(features, tweet_data)
                current_window += 1
                end_time_of_window = start_time + current_window * 3600

        else:  # tweeted After Feb. 1, 8 p.m
            if flag_after:  # reset start_time
                flag_after = False
                current_window = 1
                start_time = end_time
                end_time_of_window = start_time + current_window * 3600

            if end_time < end_time_of_window:
                features = calculate_statistic(features, tweet_data)
            else:
                extracted_features = get_features(features)
                tweet_class_after.append(extracted_features[0])
                tweet_features_after.append(extracted_features[1:])
                features = reset_features_dict()
                features = calculate_statistic(features, tweet_data)
                current_window += 1
                end_time_of_window = start_time + current_window * 3600

    tweet_class_before = np.roll(np.array(tweet_class_before), -1)
    tweet_class_between = np.roll(np.array(tweet_class_between), -1)
    tweet_class_after = np.roll(np.array(tweet_class_after), -1)

    # perform 10 fold cross validation for each time-period
    logger.info("BEFORE MODEL - Classification")
    perform_classification(np.array(tweet_features_before), np.array(tweet_class_before))
    logger.info("BETWEEN MODEL - Classification")
    perform_classification(np.array(tweet_features_between), np.array(tweet_class_between))
    logger.info("AFTER MODEL - Classification")
    perform_classification(np.array(tweet_features_after), np.array(tweet_class_after))

    logger.info("*************************************")
