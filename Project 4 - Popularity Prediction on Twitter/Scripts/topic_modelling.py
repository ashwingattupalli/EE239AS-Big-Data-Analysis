import pandas
import vincent
import datetime
import numpy as np

# topic modelling for - Advertisements, Celebrities, Goal, Team(Patriots, SeaHawks)
ads_hour_count = []
celeb_hour_count = []
goal_hour_count = []
team_hour_count = []


def perform_modelling(celeb_df, ads_df, key_words):
    try:
        ads_hour_count.append(ads_df)
    except:
        ads_hour_count.append(0)

    try:
        celeb_hour_count.append(celeb_df)
    except:
        celeb_hour_count.append(0)

    goal_count = 0
    team_count = 0
    for key in key_words.keys():
        if key.lower().find("goal") > -1 or key.lower().find("touchdown") > -1:
            goal_count += 1

        if key.lower().find("seahawks") > -1 or key.lower().find("hawks") > -1 or key.lower().find(
                "seattle") > -1 or key.lower().find("england") > -1 or key.lower().find("patriots") > -1:
            team_count += 1

    goal_hour_count.append(goal_count)
    team_hour_count.append(team_count)


def create_timeseries_topics(start_time):
    # normalize data
    advertisements = np.array(ads_hour_count) / float(max(ads_hour_count))
    celebrities = np.array(celeb_hour_count) / float(max(celeb_hour_count))
    goals = np.array(goal_hour_count) / float(max(goal_hour_count))
    teams = np.array(team_hour_count) / float(max(team_hour_count))

    # print celeb_hour_count
    data = pandas.DataFrame(
        {"Advertisements": advertisements, "Celebrities": celebrities, "Goal Chatter": goals, "Team Chatter": teams})
    data[data < 0] = 0

    timestamp_rows = []

    for i in range(len(goal_hour_count)):
        time = start_time + i * 3600
        timestamp_rows.append(datetime.datetime.fromtimestamp(time))

    idx = pandas.DatetimeIndex(timestamp_rows)
    data = data.set_index(idx)

    # data converted into a dictionary
    match_data = dict(data)
    # Dataframe for multiple series
    all_matches = pandas.DataFrame(match_data)

    # plotting
    time_chart = vincent.Line(all_matches[470:])
    time_chart.axis_titles(x='Time in hours', y='Tweet Count')
    time_chart.legend(title='Topic Modelling')
    time_chart.to_json('../Graphs/Question 6/time_chart_topics.json')
