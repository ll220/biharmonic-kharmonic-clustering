import numpy as np

def get_purity(
    clustering_results : np.array,
    true_labels : np.array,
    num_clusters : int
) -> float:
    num_correct = 0
    for cluster in range(num_clusters):
        ground_truth = true_labels[clustering_results == cluster]
        majority_label = np.bincount(ground_truth).argmax()
        num_correct += np.count_nonzero(ground_truth == majority_label)

    purity = num_correct / clustering_results.shape[0]
    return purity

if __name__=="__main__":
    NUM_CLUSTERS = 3

    results = np.array([0, 0, 1, 1, 2, 1, 1, 0, 1, 2])
    true_clusters = np.array([1, 1, 2, 0, 0, 2, 2, 1, 2, 0])

    purity = get_purity(results, true_clusters, NUM_CLUSTERS)
    print(purity)