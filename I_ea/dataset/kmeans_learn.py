import os
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import joblib
import yaml


def get_km_model(
    n_clusters,
    init,
    max_iter,
    batch_size,
    tol,
    max_no_improvement,
    n_init,
    reassignment_ratio,
):
    return MiniBatchKMeans(
        n_clusters=n_clusters,
        init=init,
        max_iter=max_iter,
        batch_size=batch_size,
        verbose=1,
        compute_labels=False,
        tol=tol,
        max_no_improvement=max_no_improvement,
        init_size=None,
        n_init=n_init,
        reassignment_ratio=reassignment_ratio,
    )


def load_feature_shard(feat_dir, split):
    feat_path = f"{feat_dir}/{split}.npy"
    return np.load(feat_path, mmap_mode="r")


def load_feature(feat_dir, split, seed):
    feat = load_feature_shard(feat_dir, split)
    print(f"loaded feature with dimension {feat.shape}")
    return feat


def learn_kmeans(
    feat_dir,
    split,
    km_path,
    n_clusters,
    seed,
    init,
    max_iter,
    batch_size,
    tol,
    n_init,
    reassignment_ratio,
    max_no_improvement,
):
    np.random.seed(seed)
    feat = load_feature(feat_dir, split, seed)
    km_model = get_km_model(
        n_clusters,
        init,
        max_iter,
        batch_size,
        tol,
        max_no_improvement,
        n_init,
        reassignment_ratio,
    )
    km_model.fit(feat)
    km_folder = os.path.dirname(km_path)
    os.makedirs(km_folder, exist_ok=True)
    joblib.dump(km_model, km_path)

    inertia = -km_model.score(feat) / len(feat)
    print("total intertia: %.5f", inertia)
    print("finished successfully")


if __name__ == "__main__":
    file_path = 'config.yaml'
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    dataset_name = data['dataset']['name']
    feat_dir = data['km_model'][dataset_name]['feat_dir']
    split = 'train_valid'
    n_clusters = data['km_model']['n_clusters']
    km_path = os.path.join(data['km_model'][dataset_name]['km_path'],
                           f'km_model_{n_clusters}/model.km')
    seed = 1234
    init = "k-means++"
    max_iter = 100
    batch_size = 1024
    tol = 0.0
    max_no_improvement = 100
    n_init = 'auto'
    reassignment_ratio = 0.01
    learn_kmeans(feat_dir=feat_dir,
                 split=split,
                 km_path=km_path,
                 n_clusters=n_clusters,
                 seed=seed,
                 init=init,
                 max_iter=max_iter,
                 batch_size=batch_size,
                 tol=tol,
                 max_no_improvement=max_no_improvement,
                 n_init=n_init,
                 reassignment_ratio=reassignment_ratio)
