import pandas
import vincent
import datetime
import numpy as np
import logging as logger

logger.basicConfig(level=logger.INFO, format='> %(message)s')

hour_status = []

celeb_first = ["John", "Idina", "Katy", "Lenny", "Missy", "Nina", "Josh"]  # first name of celebrities
celeb_last = ["Legend", "Menzel", "Perry", "Kravitz", "Elliott", "Dobrev", "Duhamel"]  # last name of celebrities


def get_celebrities(hash_tags, key_words):
    local_celeb_count = 0
    celeb_count = np.zeros(len(celeb_first))

    # the celeb name
    for count, tweet in enumerate(key_words.keys()):
        for i in range(len(celeb_count)):
            if tweet.find(celeb_first[i].lower()) > -1 or tweet.find(celeb_last[i].lower()) > -1:
                celeb_count[i] += key_words.get(tweet)
                local_celeb_count += key_words.get(tweet)

    # hash tag can have them as well
    for count, tweet in enumerate(hash_tags.keys()):
        for i in range(len(celeb_count)):
            if tweet.find(celeb_first[i].lower()) > -1 or tweet.find(celeb_last[i].lower()) > -1:
                celeb_count[i] += hash_tags.get(tweet)
                local_celeb_count += hash_tags.get(tweet)

    hour_status.append(celeb_count)
    return local_celeb_count


# to execute it on terminal type : pushd ./; sudo python -m SimpleHTTPServer 8888

def create_timeseries_celebrities(start_time):
    """
    :return: json file to get timeseries for celebrities
    """
    data = pandas.DataFrame(hour_status)
    celebrity_names = [celeb_first[ix] + " " + celeb_last[ix] for ix in range(len(celeb_first))]
    data.columns = celebrity_names

    timestamp_rows = []

    for i in range(len(hour_status)):
        time = start_time + i * 3600
        timestamp_rows.append(datetime.datetime.fromtimestamp(time))

    idx = pandas.DatetimeIndex(timestamp_rows)
    data = data.set_index(idx)

    match_data = dict(data)  # data converted into a dictionary
    all_matches = pandas.DataFrame(match_data)
    all_matches[all_matches < 0] = 0

    # plotting
    time_chart = vincent.Line(all_matches[470:])
    time_chart.axis_titles(x='Time in hours', y='Tweet Count')
    time_chart.legend(title='Celebrities')
    time_chart.to_json('../Graphs/Question 6/time_chart_celeb.json')

    return all_matches
