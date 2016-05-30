import json
import pandas as pd
from utility import *
from celebrities import *
from advertisements import *
from topic_modelling import *

logger.basicConfig(level=logger.INFO, format='> %(message)s')
input_files = ['superbowl']  # tweet-data file names
latitude = []
longitude = []

logger.info("EXECUTING: QUESTION 6 - TWITTER AD-CELEB WEEK - EVENT SEQUENCING")

for file_name in input_files:
    logger.info("EXECUTING: Analysis of #{}".format(file_name))

    tweets = open('../Dataset/tweet_data/tweets_#{0}.txt'.format(file_name), 'rb')

    # create hour windows of tweet text
    all_tweets = []
    hourly_tweet_texts = []
    current_window = 1
    first_tweet = json.loads(tweets.readline())
    start_time = first_tweet.get('firstpost_date')  # get the first tweet start time
    end_time_of_window = start_time + current_window * 3600
    number_of_tweets = len(tweets.readlines())

    tweets.seek(0, 0)  # set start point of file to 0

    logger.info("Total Number of Tweets : {}".format(number_of_tweets))

    for count, tweet in enumerate(tweets):
        tweet_data = json.loads(tweet)
        tweet_text = tweet_data.get("tweet").get("text")
        end_time = tweet_data.get('firstpost_date')

        geolocation = tweet_data.get('tweet').get("coordinates")
        if geolocation is not None:
            latitude.append(geolocation['coordinates'][0])
            longitude.append(geolocation['coordinates'][1])

        if count % 10000 == 0:
            logger.info("Tweets Analyzed : {} %".format(str(count * 100 /float(number_of_tweets))[:4]))

        # one hour window to collect tweet text and then gather topics from text
        if end_time < end_time_of_window:
            hourly_tweet_texts.append(tweet_text)
        else:
            # preprocess the hour text data
            word_list, other_hash_tags, key_words, bigrams_counter = preprocess_data(hourly_tweet_texts)
            ads_df = get_advertisements(other_hash_tags, key_words, bigrams_counter)
            celeb_df = get_celebrities(other_hash_tags, key_words)
            perform_modelling(celeb_df, ads_df, key_words)
            all_tweets.append(key_words)

            # set to default values for next hour
            current_window += 1
            hourly_tweet_texts = []
            end_time_of_window = start_time + current_window * 3600

    create_timeseries_ads(start_time)
    create_timeseries_celebrities(start_time)
    create_timeseries_topics(start_time)

    # raw_data = {'latitude': latitude, 'longitude': longitude}
    # df = pd.DataFrame(raw_data, columns = ['latitude', 'longitude'])
    # plot_distribution(df)