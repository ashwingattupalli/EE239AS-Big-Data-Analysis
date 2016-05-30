import pandas
import vincent
import datetime
import numpy as np
import logging as logger

logger.basicConfig(level=logger.INFO, format='> %(message)s')

hour_status = []

# advertisements liked by Mansee
ads = [u"tmobile", u"budweiser", u"snickers", u"McDonalds", u"CocaCola", u"Toyota", u"Doritos"]  # used for data fetching
graph_ads = ["T-Mobile", "Budweiser", "Snickers", "McDonalds", "Coca Cola", "Toyota", "Doritos"]  # used for graph
ads_taglines = [u"oneupped", u"lostdog", u"verybrady", u"paywithlovin", u"makeithappy", u"mybolddad", u"middleseat"]
bigram_ads = [u"", u"", u"McDonald s", u"", u"Coca Cola"]  # some ad names are bigrams


def get_advertisements(other_hash_tags, key_words, bigrams_counter):
    local_ad_count = 0
    ads_count = np.zeros(len(ads))

    # only the ad name
    for count, tweet in enumerate(key_words.keys()):
        for i in range(len(ads)):
            if tweet.find(ads[i].lower()) > -1:
                ads_count[i] += key_words.get(tweet)
                local_ad_count += key_words.get(tweet)

    # hash tag can have either the ad or the tag line
    for count, tweet in enumerate(other_hash_tags.keys()):
        for i in range(len(ads)):
            if tweet.find(ads[i].lower()) > -1 or tweet.find(ads_taglines[i].lower()) > -1:
                ads_count[i] += other_hash_tags.get(tweet)
                local_ad_count += other_hash_tags.get(tweet)

    # For 2 word ad names
    for count, tweet in enumerate(bigrams_counter.keys()):
        tweet_dup = " ".join(x for x in tweet)
        for i in range(len(bigram_ads)):
            if tweet_dup.find(bigram_ads[i].lower()) > -1:
                ads_count[i] += bigrams_counter.get(tweet)
                local_ad_count += bigrams_counter.get(tweet)

    hour_status.append(ads_count)

    return local_ad_count


# to execute it on terminal type : pushd ./; sudo python -m SimpleHTTPServer 8888

def create_timeseries_ads(start_time):
    """
    :return: json file to get timeseries for advertisements
    """
    df = pandas.DataFrame(hour_status)
    df.columns = graph_ads

    data = (df - df.mean()) / (df.max() - df.min())     # normalize
    data[data < 0] = 0

    timestamp_rows = []

    for i in range(len(hour_status)):
        time = start_time + i * 3600
        timestamp_rows.append(datetime.datetime.fromtimestamp(time))

    idx = pandas.DatetimeIndex(timestamp_rows)
    data = data.set_index(idx)

    match_data = dict(data)  # all the data together
    all_matches = pandas.DataFrame(match_data)
    all_matches[all_matches < 0] = 0
    # all_matches[all_matches == np.NaN] = 0

    # plotting the time-series
    time_chart = vincent.Line(all_matches[470:])
    time_chart.axis_titles(x='Time in hours', y='Tweet Count')
    time_chart.legend(title='Advertisement Names')
    time_chart.to_json('../Graphs/Question 6/time_chart_ads.json')

    return all_matches
