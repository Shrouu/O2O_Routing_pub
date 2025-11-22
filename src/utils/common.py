import numpy as np
import math
from datetime import datetime

def convert_to_datetime(year, month, day):
    try:
        return datetime(int(year), int(month), int(day))
    except ValueError:
        return None  # Handle invalid date values


def date_matching(date_index, date_list):
    min_diff = float("inf")
    closest_index = -1
    for i, date in enumerate(date_list):
        diff = abs(date_index - date)
        if diff < min_diff:
            min_diff = diff
            closest_index = i
    return closest_index


def ecl_distance(coor1, coor2):
    x1, y1 = float(coor1[0]), float(coor1[1])
    x2, y2 = float(coor2[0]), float(coor2[1])
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def average_of_three_smallest(arr, K):
    sorted_arr = np.sort(arr)
    return np.mean(sorted_arr[:K])


def order_delay_pred(loaded_model, features):

    y_pred = loaded_model.predict(features)

    return y_pred


def minutes_difference(time_str1, time_str2, fmt="%Y-%m-%d %H:%M:%S"):
    dt1 = datetime.strptime(time_str1, fmt)
    dt2 = datetime.strptime(time_str2, fmt)

    # Calculate time difference (in seconds), then convert to minutes
    diff_minutes = abs((dt2 - dt1).total_seconds()) / 60
    return diff_minutes