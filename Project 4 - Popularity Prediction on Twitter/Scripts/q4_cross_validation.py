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
    current_window = 1
    end_time_of_window = start_time + current_window * 3600
    tweets.seek(0, 0)

    # store data
    tweet_features = []
    tweet_class = []
    logger.info("Extracting features from tweets")

    for tweet in tweets:
        tweet_data = json.loads(tweet)
        end_time = tweet_data.get('firstpost_date')

        if end_time < end_time_of_window:
            features = calculate_statistic(features, tweet_data)
        else:
            '''
            features : retweets, followers, max_followers, impressions, favorite_count,
                       ranking_score, hour_of_day, number_of_users, number_of_long tweets
            '''
            # append features to data variables
            extracted_features = get_features(features)
            tweet_class.append(extracted_features[0])
            tweet_features.append(extracted_features[1:])

            features = reset_features_dict() # reset features for new window calculation
            features = calculate_statistic(features, tweet_data)  # update stats of tweet

            current_window += 1
            end_time_of_window = start_time + current_window * 3600  # update window

    logger.info("Performing 10 Folds Cross Validation")

    tweet_class = np.roll(np.array(tweet_class), -1)
    tweet_class = collections.deque(tweet_class)
    tweet_class = np.delete(tweet_class, -1)
    del (tweet_features[-1])

    perform_classification(np.array(tweet_features), np.array(tweet_class))  # 10 fold cross validation
    logger.info("**************************************************************************")
