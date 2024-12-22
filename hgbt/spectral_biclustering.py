## https://scikit-learn.org/stable/auto_examples/bicluster/plot_spectral_biclustering.html

from matplotlib import pyplot as plt
import numpy as np

from sklearn.datasets import make_checkerboard
from sklearn.cluster import SpectralBiclustering
from sklearn.metrics import consensus_score

n_clusters = (4, 3)
data, rows, columns = make_checkerboard(
    shape=(300, 300), n_clusters=n_clusters, noise=10, shuffle=False, random_state=42
)

# Creating lists of shuffled row and column indices
rng = np.random.RandomState(0)
row_idx_shuffled = rng.permutation(data.shape[0])
col_idx_shuffled = rng.permutation(data.shape[1])

data = data[row_idx_shuffled][:, col_idx_shuffled]

### Shuffled dataset example ###
# ~ plt.matshow(data, cmap=plt.cm.Blues)
# ~ plt.title("Shuffled dataset")
# ~ _ = plt.show()

model = SpectralBiclustering(n_clusters=n_clusters, method="log", random_state=0)
model.fit(data)

# Compute the similarity of two sets of biclusters
score = consensus_score(
    model.biclusters_, (rows[:, row_idx_shuffled], columns[:, col_idx_shuffled])
)
print(f"consensus score: {score:.1f}")

### Comment out one of the plot.show() options below ###

### Original dataset example ###
# ~ plt.matshow(data, cmap=plt.cm.Blues)
# ~ plt.title("Original dataset")
# ~ _ = plt.show()

### Reordering first the rows and then the columns ###
# ~ reordered_rows = data[np.argsort(model.row_labels_)]
# ~ reordered_data = reordered_rows[:, np.argsort(model.column_labels_)]
# ~ plt.matshow(reordered_data, cmap=plt.cm.Blues)
# ~ plt.title("After biclustering; rearranged to show biclusters")
# ~ _ = plt.show()

### Checkerboard Structure of rearranged data ###
plt.matshow(
    np.outer(np.sort(model.row_labels_) + 1, np.sort(model.column_labels_) + 1),
    cmap=plt.cm.Blues,
)
plt.title("Checkerboard structure of rearranged data")
plt.show()
