"""Implement entropy measure using Python. The function should accept a set of data points and their class labels and
return the entropy value.
Implement information gain measures. The function should accept data points for parents, data points for both children and
return an information gain value."""
import math
from collections import Counter


def entropy_gain(labels):
    data_len = len(labels)
    count_labels = Counter(labels)

    entropy = 0
    for count in count_labels.values():
        prob = count / data_len
        entropy -=  prob * math.log2(prob)
    return entropy

def information_gain(parent_labels, child_labels_1, child_labels_2):

    total_parent = len(parent_labels)
    total_child_labels_1 = len(child_labels_1)
    total_child_labels_2 = len(child_labels_2)

    print(f"Entropy gain of parent: {entropy_gain(parent_labels)}")
    print(f"Entropy gain of child 1: {entropy_gain(child_labels_1)}")
    print(f"Entropy gain of child 2: {entropy_gain(child_labels_2)}")

    parent_entropy = entropy_gain(parent_labels)

    weighted_entropy = (total_child_labels_1 / total_parent) * entropy_gain(child_labels_1) + (total_child_labels_2 / total_parent) * entropy_gain(child_labels_2)
    print(f"Weighted entropy: {weighted_entropy}")

    information_g = parent_entropy - weighted_entropy

    return information_g





def main():
    labels = ['p','p','p','p','p','p','q','q','q','q','q','q']
    parent_labels = ['p','p','p','p','p','p','q','q','q','q','q','q']
    child_labels_1 = ['p','q','q','p','p']
    child_labels_2 = ['p','p','p','q','q','q','q']

    print(entropy_gain(labels))


    print(information_gain(parent_labels,child_labels_1,child_labels_2))

if __name__ == '__main__':
    main()