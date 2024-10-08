import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
import torch
import warnings

def split_features(data, T=30, labda=1.0):
    """
    Partition features into T subsets to maximize mutual information with labels
    and minimize redundancy between partitions.

    Parameters:
    - X: numpy array of shape (N, F), feature matrix with N samples and F features
    - Y: numpy array of shape (N,), class labels
    - T: int, number of partitions
    - labda: float, trade-off parameter between mutual information and redundancy
    

    Returns:
    - partitions: list of lists, where each sublist contains indices of features in that partition
    """
    warnings.filterwarnings("ignore")
    X = data.x
    Y = data.y
    N, F = X.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    X = X.to('cpu')
    Y = Y.to('cpu')

    # Step 1: Compute Mutual Information between each feature and labels
    print("Computing mutual information...")
    mi_features_labels = mutual_info_classif(X, Y, discrete_features=False)

    # Step 2: Compute redundancy between features using absolute Pearson correlation
    print("Computing redundancy...")
    # Standardize features
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    # Compute correlation matrix
    corr_matrix = np.abs(np.corrcoef(X_std.T))
    # Set diagonal to zero to ignore self-correlation
    np.fill_diagonal(corr_matrix, 0)

    # Step 3: Initialize partitions
    partitions = [[] for _ in range(T)]
    unassigned_features = set(range(F))

    # Step 4: Assign the most informative features to each partition
    print("Initial assigning...")
    sorted_features = np.argsort(-mi_features_labels)  # Descending order
    for t in range(T):
        if len(unassigned_features) == 0:
            break
        # Select the most informative unassigned feature
        for f in sorted_features:
            if f in unassigned_features:
                selected_feature = f
                break
        # Assign to partition t
        partitions[t].append(selected_feature)
        unassigned_features.remove(selected_feature)

    # Step 5: Iteratively assign remaining features
    print("Iteratively assigning features...")
    while unassigned_features:
        for f in list(unassigned_features):
            max_net_gain = -np.inf

            # Calculate net gain for assigning feature f to each partition
            for t in range(T):
                # Information gain
                delta_I_label = mi_features_labels[f]

                # Redundancy with features in partition t
                redundancy = 0.0
                for f_prime in partitions[t]:
                    redundancy += corr_matrix[f, f_prime]

                delta_I_redundancy = redundancy  # Using correlation as proxy
                net_gain = delta_I_label - labda * delta_I_redundancy

                if net_gain > max_net_gain:
                    max_net_gain = net_gain
                    best_partition = t

            # Assign feature f to the best partition
            partitions[best_partition].append(f)
            unassigned_features.remove(f)

      # Step 6: Compute partition sizes
    output = []
    sizes = []
    for partition in partitions:
        data_partition = X[:, partition].to(device)
        data_partition = torch.tensor(data_partition).to(device)
        output.append(data_partition)
        sizes.append(len(partition))



      

    print("Partition done!")
    return output, sizes
