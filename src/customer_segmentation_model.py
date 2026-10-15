import numpy as np


class CustomerSegmentationModel:
    """K-Means model for customer segmentation using NumPy."""

    def __init__(self, n_clusters=2, random_state=42, max_iterations=100):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_iterations = max_iterations
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None

    def fit(self, data):
        """Train the model on already-scaled customer features."""
        X = np.asarray(data, dtype=float)

        if X.ndim != 2 or len(X) < self.n_clusters:
            raise ValueError("Invalid data or too few customers.")

        rng = np.random.default_rng(self.random_state)
        indices = rng.choice(len(X), size=self.n_clusters, replace=False)
        centroids = X[indices].copy()

        for _ in range(self.max_iterations):
            distances = np.sqrt(
                ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
            )
            labels = np.argmin(distances, axis=1)

            new_centroids = centroids.copy()

            for cluster in range(self.n_clusters):
                members = X[labels == cluster]
                if len(members) > 0:
                    new_centroids[cluster] = members.mean(axis=0)

            if np.allclose(centroids, new_centroids, atol=1e-6, rtol=0):
                centroids = new_centroids
                break

            centroids = new_centroids

        distances = np.sqrt(
            ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        )
        labels = np.argmin(distances, axis=1)

        self.centroids = centroids
        self.labels_ = labels
        self.inertia_ = float(
            np.sum((X - centroids[labels]) ** 2)
        )

        return self

    def predict(self, data):
        """Assign customers to the nearest trained centroid."""
        if self.centroids is None:
            raise ValueError("Train the model before making predictions.")

        X = np.asarray(data, dtype=float)

        if X.ndim != 2 or X.shape[1] != self.centroids.shape[1]:
            raise ValueError("Input must have the same number of features.")

        distances = np.sqrt(
            ((X[:, None, :] - self.centroids[None, :, :]) ** 2).sum(axis=2)
        )
        return np.argmin(distances, axis=1)