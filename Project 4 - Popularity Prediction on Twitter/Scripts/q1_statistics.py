import json
import logging as logger
from collections import defaultdict
from matplotlib import pyplot as plt

logger.basicConfig(level=logger.INFO, format='> %(message)s')

input_files = ['gohawks', 'gopatriots', 'nfl', 'patriots', 'sb49', 'superbowl']  # tweet-data file names

logger.info("EXECUTING: QUESTION 1 - STATISTICS CALCULATION")

# loop through each file in tweet-data folder
for file_name in input_files:
    logger.info("Calculating Statistics for #{}".format(file_name))
    tweets = open('../Dataset/tweet_data/tweets_#{0}.txt'.format(file_name), 'rb')
    first_tweet = json.loads(tweets.readline())

    user_ids = defaultdict()
    number_of_tweets = len(tweets.readlines())  # get number of tweets
    number_of_retweets = 0
    tweets.seek(0, 0)  # set start point to 0 again

    current_window = 1
    start_time = first_tweet.get('firstpost_date')  # get the first tweet start time
    user_ids[first_tweet.get('tweet').get('user').get('id')] = first_tweet.get('author').get('followers')  # assign the first user_id and its no. of followers
    end_time_of_window = start_time + current_window * 3600  # window to keep track of number of hours of data

    number_of_tweets_hour = []
    current_hour_count = 0

    # loop through each tweet and calculate statistics
    for tweet in tweets:
        tweet_data = json.loads(tweet)
        end_time = tweet_data.get('firstpost_date')

        number_of_retweets += tweet_data.get('metrics').get('citations').get('total')
        if end_time < end_time_of_window:
            current_hour_count += 1
        else:
            number_of_tweets_hour.append(current_hour_count)
            current_window += 1
            current_hour_count = 0
            end_time_of_window = start_time + current_window * 3600

        user_id = tweet_data.get('tweet').get('user').get('id')
        number_of_followers = tweet_data.get('author').get('followers')
        user_ids[user_id] = number_of_followers

    # output statistics
    logger.info("Statistics for #{}".format(file_name))
    logger.info("Total Number of Tweets : {}".format(number_of_tweets))
    logger.info("Average Number of Tweets per hour : {}".format(number_of_tweets / ((end_time - start_time) / 3600.)))
    logger.info("Average Number of Retweets : {}".format(number_of_retweets / float(number_of_tweets)))
    logger.info("Average Number of Followers of Users : {}".format(sum(user_ids.values()) / float(len(user_ids.keys()))))

    # plot graph for #superbowl and #nfl
    if (file_name == 'superbowl' or file_name == "nfl"):
        plt.figure(2 if file_name == 'superbowl' else 1)
        plt.ylabel('Number of Tweets')
        plt.xlabel('Hour')
        plt.title('Number of Tweets per hour for {}'.format(file_name))
        plt.bar(range(len(number_of_tweets_hour)), number_of_tweets_hour)

    logger.info("*************************************")

plt.show()