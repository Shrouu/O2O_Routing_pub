import pickle
import random


def get_opt_test_driver_orders(order_dict, order_numbers, sample_threshold, driver_dict, driver_numbers, seed):
    sorted_order_dict = dict(
        sorted(
            order_dict.items(),
            key=lambda item: (item[1][2][4], item[1][2][3], item[1][2][2], item[1][2][0]),
        )
    )
    # item[1][2][4] is year, item[1][2][3] is month, item[1][2][2] is day, item[1][2][0] is hour
    # Count the orders with same hour of the same day of the same month of the same year
    count = 0
    max_count = 0
    greater_than_100 = 0
    previous_hour = -1
    previous_day = -1
    previous_month = -1
    previous_year = -1
    order_lists = []
    order_list = []
    for key, value in sorted_order_dict.items():
        if previous_hour == value[2][0] and previous_day == value[2][2] and previous_month == value[2][3] and previous_year == value[2][4]:
            count += 1
            order_list.append(key)
        else:
            if count > max_count:
                max_count = count
            if count > order_numbers:
                if random.random() < sample_threshold:
                    order_lists.append(order_list)
                greater_than_100 += 1
            count = 1
            order_list = [key]
        previous_hour = value[2][0]
        previous_day = value[2][2]
        previous_month = value[2][3]
        previous_year = value[2][4]
    random_index = random.randint(0, len(order_lists) - 1)
    order_list = order_lists[random_index][0:order_numbers]

    # Save the order list
    with open("order_list_" + str(seed) + "_60.pkl", "wb") as f:
        pickle.dump(order_list, f)

    full_list = []
    part_list = []
    occ_list = []

    for driver_id in driver_dict.keys():
        data_keys = list(driver_dict[driver_id].keys())
        driver_class = driver_dict[driver_id][data_keys[-1]][1]
        if driver_class[0] == 1:
            full_list.append(driver_id)
        elif driver_class[1] == 1:
            part_list.append(driver_id)
        else:
            occ_list.append(driver_id)

    # Randomly select several drivers from each class
    # full_list=random.sample(full_list, int(0.2*driver_numbers))
    # part_list=random.sample(part_list, int(0.3*driver_numbers))
    # occ_list=random.sample(occ_list, driver_numbers-len(full_list)-len(part_list))
    # driver_list=full_list+part_list+occ_list
    full_list = random.sample(full_list, 7)
    part_list = random.sample(part_list, 10)
    occ_list = random.sample(occ_list, 13)
    driver_list = full_list + part_list + occ_list
    with open("driver_list_" + str(seed) + "_60_30.pkl", "wb") as f:
        pickle.dump(driver_list, f)

    return order_list, driver_list


if __name__ == "__main__":
    path = "./Resources/"
    order_numbers = 60
    driver_numbers = 30

    with open(path + "order_dict_two_order.pkl", "rb") as f:
        order_dict = pickle.load(f)

    with open(path + "two_order_external_data_dict.pkl", "rb") as f:
        order_external_data_dict = pickle.load(f)

    with open(path + "experience_optimize.pickle", "rb") as f:
        experience_optimize = pickle.load(f)

    with open(path + "order_to_driver.pickle", "rb") as f:
        order_to_driver = pickle.load(f)

    with open(path + "experience_dict4.pickle", "rb") as f:
        experience_dict = pickle.load(f)

    for i in range(3):
        seed = 202 + i + order_numbers * 100000 + driver_numbers * 1000
        random.seed(seed)
        # Get order and driver data
        test_order_list, test_driver_list = get_opt_test_driver_orders(
            order_dict, order_numbers=order_numbers, sample_threshold=0.01, driver_dict=experience_optimize, driver_numbers=driver_numbers, seed=seed
        )
