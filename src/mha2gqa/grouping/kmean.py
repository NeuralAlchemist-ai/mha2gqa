import torch

class PyTorchKMeans:
    """
    Легковесный K-Means на чистом PyTorch для open-source проектов.
    Не имеет сторонних зависимостей, поддерживает CPU/GPU.
    Модифицировано для создания кластеров примерно одинакового размера (Same-Size K-Means).
    """
    def __init__(self, n_clusters, max_iter=300, tol=1e-4, device=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.device = device
        self.cluster_centers_ = None

    def fit(self, X):
        device = self.device if self.device is not None else X.device
        X = X.to(device)
        n_samples, n_features = X.shape

        if n_samples < self.n_clusters:
            raise ValueError(f"n_samples={n_samples} должно быть больше или равно n_clusters={self.n_clusters} для Same-Size K-Means.")

        # Инициализация центроидов (случайный выбор из данных)
        permutation = torch.randperm(n_samples, device=device)
        self.cluster_centers_ = X[permutation[:self.n_clusters]].clone()

        # Calculate target cluster sizes once for the fit process
        base_cluster_size = n_samples // self.n_clusters
        remainder = n_samples % self.n_clusters
        target_cluster_capacities = torch.full((self.n_clusters,), base_cluster_size, dtype=torch.long, device=device)
        target_cluster_capacities[:remainder] += 1 # Distribute remainder points to the first 'remainder' clusters

        for _ in range(self.max_iter):
            # Store old centers for convergence check
            old_centers = self.cluster_centers_.clone()

            # 1. Calculate distances from each point to each centroid
            distances = torch.cdist(X, self.cluster_centers_) # Shape (n_samples, n_clusters)

            # 2. Perform balanced assignment
            # Flatten distances to get all (distance, point_idx, centroid_idx) triplets implicitly
            flat_distances = distances.flatten() 
            
            # Create corresponding point_indices and cluster_indices for these flattened distances
            point_indices = torch.arange(n_samples, device=device).unsqueeze(1).repeat(1, self.n_clusters).flatten()
            cluster_indices = torch.arange(self.n_clusters, device=device).repeat(n_samples)

            # Sort all (distance, point_idx, cluster_idx) by distance
            _, sort_order = torch.sort(flat_distances)

            # Initialize labels, current cluster counts, and point assignment status
            labels = -1 * torch.ones(n_samples, dtype=torch.long, device=device)
            current_cluster_counts = torch.zeros(self.n_clusters, dtype=torch.long, device=device)
            point_assigned = torch.zeros(n_samples, dtype=torch.bool, device=device)

            # Iterate through sorted assignments and assign points respecting capacities
            for i in range(n_samples * self.n_clusters):
                # Get the point and cluster index for the current smallest distance
                p_idx = point_indices[sort_order[i]]
                c_idx = cluster_indices[sort_order[i]]

                # If the point hasn't been assigned yet AND the cluster has capacity
                if not point_assigned[p_idx] and current_cluster_counts[c_idx] < target_cluster_capacities[c_idx]:
                    labels[p_idx] = c_idx
                    current_cluster_counts[c_idx] += 1
                    point_assigned[p_idx] = True

                # Optimization: if all points are assigned, break early
                if point_assigned.all():
                    break
            
            # This check is crucial for robustness, though with correct target capacities it should not trigger.
            if not point_assigned.all():
                 raise RuntimeError("Error: Not all points were assigned in Same-Size K-Means during fit. This indicates a logic error or unexpected state.")

            # 3. Recalculate centroids
            new_centers = torch.empty_like(self.cluster_centers_)
            for k in range(self.n_clusters):
                mask = labels == k
                if mask.any(): # Ensure the mask is not empty
                    new_centers[k] = X[mask].mean(dim=0)
                else:
                    # If a cluster is empty, keep its old centroid.
                    new_centers[k] = old_centers[k]
            
            # Check for convergence
            center_shift = torch.sum((self.cluster_centers_ - new_centers) ** 2)
            self.cluster_centers_ = new_centers

            if center_shift < self.tol:
                break

        return self

    def predict(self, X):
        if self.cluster_centers_ is None:
            raise RuntimeError("Модель еще не обучена. Вызовите .fit() перед .predict()")

        device = self.cluster_centers_.device
        X = X.to(device)
        n_samples, _ = X.shape

        # For prediction, we also enforce the same-size constraint
        base_cluster_size = n_samples // self.n_clusters
        remainder = n_samples % self.n_clusters
        target_cluster_capacities = torch.full((self.n_clusters,), base_cluster_size, dtype=torch.long, device=device)
        target_cluster_capacities[:remainder] += 1

        distances = torch.cdist(X, self.cluster_centers_) # Shape (n_samples, n_clusters)
        
        flat_distances = distances.flatten() 
        point_indices = torch.arange(n_samples, device=device).unsqueeze(1).repeat(1, self.n_clusters).flatten()
        cluster_indices = torch.arange(self.n_clusters, device=device).repeat(n_samples)
        _, sort_order = torch.sort(flat_distances)

        labels = -1 * torch.ones(n_samples, dtype=torch.long, device=device)
        current_cluster_counts = torch.zeros(self.n_clusters, dtype=torch.long, device=device)
        point_assigned = torch.zeros(n_samples, dtype=torch.bool, device=device)

        for i in range(n_samples * self.n_clusters):
            p_idx = point_indices[sort_order[i]]
            c_idx = cluster_indices[sort_order[i]]

            if not point_assigned[p_idx] and current_cluster_counts[c_idx] < target_cluster_capacities[c_idx]:
                labels[p_idx] = c_idx
                current_cluster_counts[c_idx] += 1
                point_assigned[p_idx] = True

            if point_assigned.all():
                break
        
        if not point_assigned.all():
             raise RuntimeError("Error: Not all points were assigned during prediction in Same-Size K-Means. This indicates a logic error or unexpected state.")

        return labels

    def fit_predict(self, X):
        return self.fit(X).predict(X)