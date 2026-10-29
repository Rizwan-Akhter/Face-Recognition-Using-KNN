
import numpy as np


def distance(v1, v2):
    """Calculate Euclidean distance between two vectors."""
    v1 = np.asarray(v1, dtype=np.float32)
    v2 = np.asarray(v2, dtype=np.float32)

    return float(np.linalg.norm(v1 - v2))


def knn(train, test, k=5, threshold=25.0):
    """
    Predict a person's label using KNN.

    train: 2D array containing feature vectors and labels
    test: 1D feature vector
    k: number of nearest neighbors
    threshold: maximum acceptable distance
    """
    train = np.asarray(train, dtype=np.float32)
    test = np.asarray(test, dtype=np.float32)

    if train.ndim != 2 or train.shape[0] == 0:
        return -1

    if train.shape[1] != test.size + 1:
        raise ValueError("Training and test feature sizes do not match.")

    k = max(1, min(int(k), len(train)))

    distances = np.linalg.norm(
        train[:, :-1] - test,
        axis=1
    )

    nearest_indices = np.argsort(distances)[:k]
    nearest_distances = distances[nearest_indices]

    # Reject faces that are too different from known samples.
    if nearest_distances[0] > threshold:
        return -1

    nearest_labels = train[nearest_indices, -1].astype(int)

    unique_labels, counts = np.unique(
        nearest_labels,
        return_counts=True
    )

    # Resolve ties using the closest sample among tied labels.
    max_count = counts.max()
    candidates = unique_labels[counts == max_count]

    if len(candidates) == 1:
        return int(candidates[0])

    for index in nearest_indices:
        label = int(train[index, -1])
        if label in candidates:
            return label

    return int(candidates[0])
