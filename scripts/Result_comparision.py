import pandas as pd
import pickle
import numpy as np
import os
import random


def optimized_distribute_elements(arr, num_subarrays=10, max_length=3):
    # Initialize target number of empty subarrays
    subarrays = [[] for _ in range(num_subarrays)]

    # Randomly shuffle the original array
    random.shuffle(arr)

    for elem in arr:
        # Get indices of subarrays that still have capacity
        available_indices = [i for i in range(num_subarrays) if len(subarrays[i]) < max_length]

        if available_indices:
            # Randomly choose one from available subarrays
            chosen_index = random.choice(available_indices)
            subarrays[chosen_index].append(elem)
        else:
            # All subarrays are full, stop distribution
            break
    # Convert subarrays of length 1 to int, subarrays of length > 1 to tuple
    processed_subarrays = [item[0] if len(item) == 1 else tuple(sorted(item)) for item in subarrays]

    return processed_subarrays


def get_true_costs(assignment, cost_matrix, order_number, two_combinations, three_combinations):
    objective_value = 0
    for sol in assignment:
        batch = sol[0]
        driver = sol[1]
        if len(batch) == 1:
            batch_index = batch[0]
            objective_value += cost_matrix[driver][batch_index]
        elif len(batch) == 2:
            batch_index = two_combinations.index(tuple(batch))
            objective_value += cost_matrix[driver][order_number + batch_index]
        elif len(batch) == 3:
            batch_index = three_combinations.index(tuple(batch))
            objective_value += cost_matrix[driver][order_number + len(two_combinations) + batch_index]
    return objective_value


def get_order_details(assignment, cost_matrix, order_number, two_combinations, three_combinations, two_matrix, three_matrix, routes):
    costs = []
    for sol in assignment:
        batch = sol[0]
        driver = sol[1]
        if len(batch) == 1:
            batch_index = batch[0]
            costs.append(cost_matrix[driver][batch_index])
        elif len(batch) == 2:
            batch_index = two_combinations.index(tuple(batch))
            route = routes[driver, order_number + batch_index]
            if route == 0:
                temp_cost = two_matrix[driver, batch[0], batch[1]]
                costs += temp_cost.tolist()
            else:
                temp_cost = two_matrix[driver, batch[1], batch[0]]
                costs += temp_cost.tolist()
        elif len(batch) == 3:
            batch_index = three_combinations.index(tuple(batch))
            route = routes[driver, order_number + len(two_combinations) + batch_index]
            if route == 0:
                temp_cost = three_matrix[driver, batch[0], batch[1], batch[2]]
                costs += temp_cost.tolist()
            elif route == 1:
                temp_cost = three_matrix[driver, batch[0], batch[2], batch[1]]
                costs += temp_cost.tolist()
            elif route == 2:
                temp_cost = three_matrix[driver, batch[1], batch[0], batch[2]]
                costs += temp_cost.tolist()
            elif route == 3:
                temp_cost = three_matrix[driver, batch[1], batch[2], batch[0]]
                costs += temp_cost.tolist()
            elif route == 4:
                temp_cost = three_matrix[driver, batch[2], batch[0], batch[1]]
                costs += temp_cost.tolist()
            elif route == 5:
                temp_cost = three_matrix[driver, batch[2], batch[1], batch[0]]
                costs += temp_cost.tolist()
    return costs


path = "./"
order_numbers = [30,45,60]
driver_numbers = [[10, 12, 15],[15, 20, 23],[20, 25, 30]]
seeds = [202, 203, 204]

# generate ID list
with open(path + "Results/ID_list.pkl", "rb") as f:
    ID_list = pickle.load(f)
with open(path + "Results/assignments.pkl", "rb") as f:
    assignments_list = pickle.load(f)

# Read all pkl files in the combinations folder and store them in a dict
two_combinations = {}
three_combinations = {}
for file in os.listdir(path + "Resources/Combinations/"):
    with open(path + "Resources/combinations/" + file, "rb") as f:
        data = pickle.load(f)
    if "2" in file:
        two_combinations[file.split(".")[0]] = data
    elif "3" in file:
        three_combinations[file.split(".")[0]] = data


compared_results = {}
compared_results_delay_rate = {}
compared_results_delay_exp = {}
compared_results_delay_extra = {}

for ID_index in range(len(ID_list)):
    ID = ID_list[ID_index]
    order_numbers = ID[0]
    driver_numbers = ID[1]
    order_seed = ID[2]
    driver_seed = ID[3]
    if ID_index % 27 == 0:
        with np.load(path + f"Results/cost_matrices_{order_numbers}.npz") as data:
            one_matrix = data["one_matrix"]
            two_matrix = data["two_matrix"]
            three_matrix = data["three_matrix"]
            expectation_matrix = data["expectation_matrix"]
            routes = data["routes"]

    matrix_index = ID_index % 27
    if matrix_index % 3 == 0:
        matrix_index = matrix_index // 3
        # File path
        one_matrix_temp = one_matrix[matrix_index]
        two_matrix_temp = two_matrix[matrix_index]
        three_matrix_temp = three_matrix[matrix_index]
        expectation_matrix_temp = expectation_matrix[matrix_index]
        routes_temp = routes[matrix_index]
    result = np.zeros(3)
    result_delay_rate = np.zeros(3)
    result_delay_exp = np.zeros(3)
    result_delay_extra = np.zeros(12)
    assignment = assignments_list[ID_index]
    routing = assignment["routing"]
    driver_feature = assignment["driver_feature"]
    benchmark = assignment["benchmark"]
    routing_cost_details = get_order_details(
        routing,
        one_matrix_temp,
        order_numbers,
        two_combinations[str(order_numbers) + "_2"],
        three_combinations[str(order_numbers) + "_3"],
        two_matrix_temp,
        three_matrix_temp,
        routes_temp,
    )
    driver_cost_details = get_order_details(
        driver_feature,
        one_matrix_temp,
        order_numbers,
        two_combinations[str(order_numbers) + "_2"],
        three_combinations[str(order_numbers) + "_3"],
        two_matrix_temp,
        three_matrix_temp,
        routes_temp,
    )
    benchmark_cost_details = get_order_details(
        benchmark,
        one_matrix_temp,
        order_numbers,
        two_combinations[str(order_numbers) + "_2"],
        three_combinations[str(order_numbers) + "_3"],
        two_matrix_temp,
        three_matrix_temp,
        routes_temp,
    )
    routing_cost_details = np.array(routing_cost_details)
    driver_cost_details = np.array(driver_cost_details)
    benchmark_cost_details = np.array(benchmark_cost_details)
    routing_cost_expectaion = get_true_costs(
        routing, expectation_matrix_temp, order_numbers, two_combinations[str(order_numbers) + "_2"], three_combinations[str(order_numbers) + "_3"]
    )
    driver_cost_expectaion = get_true_costs(
        driver_feature,
        expectation_matrix_temp,
        order_numbers,
        two_combinations[str(order_numbers) + "_2"],
        three_combinations[str(order_numbers) + "_3"],
    )
    benchmark_cost_expectaion = get_true_costs(
        benchmark, expectation_matrix_temp, order_numbers, two_combinations[str(order_numbers) + "_2"], three_combinations[str(order_numbers) + "_3"]
    )
    result[0] = np.sum(routing_cost_details)
    result[1] = np.sum(driver_cost_details)
    result[2] = np.sum(benchmark_cost_details)

    result_delay_rate[0] = (routing_cost_details > 0).sum()
    result_delay_rate[1] = (driver_cost_details > 0).sum()
    result_delay_rate[2] = (benchmark_cost_details > 0).sum()

    result_delay_exp[0] = routing_cost_expectaion
    result_delay_exp[1] = driver_cost_expectaion
    result_delay_exp[2] = benchmark_cost_expectaion

    routing_positive_details = routing_cost_details[routing_cost_details > 0]
    driver_positve_details = driver_cost_details[driver_cost_details > 0]
    benchmark_positive_details = benchmark_cost_details[benchmark_cost_details > 0]

    if len(routing_positive_details) > 0:
        result_delay_extra[0] = np.mean(routing_positive_details)
        result_delay_extra[3] = np.median(routing_positive_details)
        result_delay_extra[6] = np.max(routing_positive_details)
        result_delay_extra[9] = np.var(routing_positive_details)
    if len(driver_positve_details) > 0:
        result_delay_extra[1] = np.mean(driver_positve_details)
        result_delay_extra[4] = np.median(driver_positve_details)
        result_delay_extra[7] = np.max(driver_positve_details)
        result_delay_extra[10] = np.var(driver_positve_details)
    if len(benchmark_positive_details) > 0:
        result_delay_extra[2] = np.mean(benchmark_positive_details)
        result_delay_extra[5] = np.median(benchmark_positive_details)
        result_delay_extra[8] = np.max(benchmark_positive_details)
        result_delay_extra[11] = np.var(benchmark_positive_details)

    compared_results[ID] = result
    compared_results_delay_rate[ID] = result_delay_rate
    compared_results_delay_exp[ID] = result_delay_exp
    compared_results_delay_extra[ID] = result_delay_extra


# Convert compared_results to DataFrame
data = []
for key, values in compared_results.items():
    key_string = ", ".join(map(str, key))  # Join elements with commas
    row = [key_string] + values.tolist()
    data.append(row)

delay_rate_data = []
for key, values in compared_results_delay_rate.items():
    key_string = ", ".join(map(str, key))  # Join elements with commas
    row = [key_string] + values.tolist()
    delay_rate_data.append(row)

delay_extra_data = []
for key, values in compared_results_delay_extra.items():
    key_string = ", ".join(map(str, key))  # Join elements with commas
    row = [key_string] + values.tolist()
    delay_extra_data.append(row)

delay_exp_data = []
for key, values in compared_results_delay_exp.items():
    key_string = ", ".join(map(str, key))  # Join elements with commas
    row = [key_string] + values.tolist()
    delay_exp_data.append(row)

# Create DataFrame
df = pd.DataFrame(data, columns=["ID", "Routing True Cost", "Benchmark True Cost1", "Benchmark True Cost2"])
delay_rate_df = pd.DataFrame(delay_rate_data, columns=["ID", "Routing True Delay Rate", "Benchmark True Delay Rate1", "Benchmark True Delay Rate2"])
delay_posi_df = pd.DataFrame(
    delay_exp_data, columns=["ID", "Routing True Delay Expectation", "Benchmark True Delay Expectation1", "Benchmark True Delay Expectation2"]
)
deviation_df = pd.DataFrame(
    delay_extra_data,
    columns=["ID", "Mean R", "Mean D", "Mean B", "Median R", "Median D", "Median B", "Max R", "Max D", "Max B", "Var R", "Var D", "Var B"],
)

# Save as CSV file
df.to_csv(path + "Results/compared_results_delay.csv", index=False)
delay_rate_df.to_csv(path + "Results/compared_results_delay_rate.csv", index=False)
delay_posi_df.to_csv(path + "Results/compared_results_delay_exp.csv", index=False)
deviation_df.to_csv(path + "Results/compared_results_delay_extra.csv", index=False)

print("CSV file saved to:", path + "compared_results.csv")
