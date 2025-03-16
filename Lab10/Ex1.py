# import math
# from collections import Counter
#
# def compute_entropy(labels):
#     """Computes entropy for a given set of labels."""
#     if len(labels) == 0:  # Handle empty set case
#         return 0
#
#     label_counts = Counter(labels)  # Count occurrences of each label
#     total_samples = len(labels)
#
#     entropy = 0
#     for count in label_counts.values():
#         prob = count / total_samples  # Probability of each label
#         entropy -= prob * math.log2(prob)  # Shannon entropy formula
#
#     return entropy
#
# def information_gain(parent_labels, child_labels_1, child_labels_2):
#     """Computes information gain for a split."""
#
#     total_parent = len(parent_labels)
#     total_child_1 = len(child_labels_1)
#     total_child_2 = len(child_labels_2)
#
#     if total_parent == 0:  # Avoid division by zero
#         return 0
#
#     parent_entropy = compute_entropy(parent_labels)
#     weighted_entropy = (
#         (total_child_1 / total_parent) * compute_entropy(child_labels_1) +
#         (total_child_2 / total_parent) * compute_entropy(child_labels_2)
#     )
#
#     info_gain = parent_entropy - weighted_entropy  # IG formula
#
#     # Debugging prints
#     print(f"Parent Entropy: {parent_entropy}")
#     print(f"Child 1 Entropy: {compute_entropy(child_labels_1)}, Child 2 Entropy: {compute_entropy(child_labels_2)}")
#     print(f"Weighted Entropy: {weighted_entropy}")
#     print(f"Information Gain: {info_gain}")
#
#     return info_gain
#
#
# # Example usage
# def main():
#     parent_labels = ['p', 'p', 'p', 'p', 'p', 'p', 'q', 'q', 'q', 'q', 'q', 'q']
#     child_labels_1 = ['p', 'q', 'q', 'p', 'p']
#     child_labels_2 = ['p', 'p', 'p', 'q', 'q', 'q', 'q']
#
#     print("Entropy of Parent Set:", compute_entropy(parent_labels))
#     print("Information Gain from Split:", information_gain(parent_labels, child_labels_1, child_labels_2))
#
# if __name__ == "__main__":
#     main()


import math
from collections import Counter


def compute_entropy(labels):
    """Computes entropy for a given set of labels."""
    if len(labels) == 0:  # Handle empty set case
        return 0

    label_counts = Counter(labels)  # Count occurrences of each label
    total_samples = len(labels)

    entropy = sum(
        -(count / total_samples) * math.log2(count / total_samples)
        for count in label_counts.values()
    )  # Using sum() for cleaner code

    return entropy


def information_gain(parent_labels, child_labels_1, child_labels_2):
    """Computes information gain for a split."""
    total_parent = len(parent_labels)
    total_child_1 = len(child_labels_1)
    total_child_2 = len(child_labels_2)

    if total_parent == 0:  # Avoid division by zero
        return 0

    parent_entropy = compute_entropy(parent_labels)

    # Compute entropy only if the child set is not empty
    entropy_child_1 = compute_entropy(child_labels_1) if total_child_1 > 0 else 0
    entropy_child_2 = compute_entropy(child_labels_2) if total_child_2 > 0 else 0

    weighted_entropy = (
            (total_child_1 / total_parent) * entropy_child_1 +
            (total_child_2 / total_parent) * entropy_child_2
    )

    info_gain = parent_entropy - weighted_entropy  # IG formula

    return info_gain


# Example usage
def main():
    parent_labels = ['p', 'p', 'p', 'p', 'p', 'p', 'q', 'q', 'q', 'q', 'q', 'q']
    child_labels_1 = ['p', 'q', 'q', 'p', 'p']
    child_labels_2 = ['p', 'p', 'p', 'q', 'q', 'q', 'q']

    print("Entropy of Parent Set:", compute_entropy(parent_labels))
    print("Information Gain from Split:", information_gain(parent_labels, child_labels_1, child_labels_2))


if __name__ == "__main__":
    main()
