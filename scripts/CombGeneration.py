import itertools
import pickle

order_numbers = 60
path="./combinations/"

elements = list(range(order_numbers))
two_pairs = list(itertools.combinations(elements, 2))
three_pairs = list(itertools.combinations(elements, 3))

# Save data
with open(path + str(order_numbers)+"_2.pkl", "wb") as f:
    pickle.dump(two_pairs, f)

with open(path + str(order_numbers)+"_3.pkl", "wb") as f:
    pickle.dump(three_pairs, f)