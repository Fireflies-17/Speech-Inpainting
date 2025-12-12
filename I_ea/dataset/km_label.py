import os
import numpy as np
import joblib
import torch
from tqdm import tqdm
import yaml
from Inpainting.dataset.kmeans_learn import load_feature_shard


class ApplyKmeans(object):

    def __init__(self, km_path, device=None):
        self.km_model = joblib.load(km_path)
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np**2).sum(0, keepdims=True)

        self.C = torch.from_numpy(self.C_np)
        self.Cnorm = torch.from_numpy(self.Cnorm_np)
        if device is not None:
            self.C = self.C.to(device)
            self.Cnorm = self.Cnorm.to(device)
        elif torch.cuda.is_available():
            self.C = self.C.cuda()
            self.Cnorm = self.Cnorm.cuda()

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            dist = (x.pow(2).sum(1, keepdim=True) -
                    2 * torch.matmul(x, self.C) + self.Cnorm)
            return dist.argmin(dim=1).cpu().numpy()
        else:
            dist = ((x**2).sum(1, keepdims=True) -
                    2 * np.matmul(x, self.C_np) + self.Cnorm_np)
            return np.argmin(dist, axis=1)


def dump_label(feat_dir, split, km_path, lab_dir):
    """
    Apply K-means clustering to all the features extracted.

    Args:
        feat_dir (str): The directory containing the feature.
        split (str): The name of the file.
        km_path (str): The file path to the pre-fitted K-means model.
        lab_dir (str): The directory to store the generated labels.

    Returns:
        None. The generated labels will be saved in lab_dir/split.km
    """
    apply_kmeans = ApplyKmeans(km_path)
    lab_path = os.path.join(lab_dir, split + ".km")
    os.makedirs(lab_dir, exist_ok=False
                )  # delete previous labeling generated from different configs.
    with open(lab_path, "w") as f:
        feats = load_feature_shard(feat_dir, split)
        for feat in tqdm(feats):
            feat = torch.from_numpy(feat.copy()).unsqueeze(0).cuda()
            lab = apply_kmeans(feat).tolist()
            f.write(" ".join(map(str, lab)) + "\n")
    print("finished successfully")


def dump_split_label(feat_dir, split, km_path, lab_dir, splits_path):
    """
    Apply K-means clustering to all the features extracted.

    Args:
        feat_dir (str): The directory containing the feature.
        split (str): training/validation splits.
        km_path (str): The file path to the pre-fitted K-means model.
        lab_dir (str): The directory to store the generated labels.
        splits_path (str): The path of splits (training.txt, validaiton.txt)

    Returns:
        None. The generated labels will be saved in lab_dir/training
    """
    apply_kmeans = ApplyKmeans(km_path)

    os.makedirs(lab_dir, exist_ok=True)
    path_split = os.path.join(splits_path, split + '.txt')
    with open(path_split, 'r') as f:
        wavs_files = [line.split('|')[0] for line in f]

    for wave_file in tqdm(wavs_files):
        abs_wave_file = os.path.join(lab_dir, wave_file)
        abs_wave_pt = os.path.join(feat_dir, wave_file + '.pt')
        mel_feats = torch.load(abs_wave_pt)
        mel_labels = torch.tensor([
            apply_kmeans(mel_feats[i, :].unsqueeze(0).cuda())[0]
            for i in range(mel_feats.shape[0])
        ],
            dtype=torch.long)
        # consider transpose this tensor to make the timesteps first
        mel_centroids = apply_kmeans.C[:, mel_labels]
        torch.save(mel_labels, abs_wave_file + '_labels.pt')
        torch.save(mel_centroids, abs_wave_file + '_mel_c.pt')
    print("finished successfully")


def check_cos_sim(km_path):
    apply_kmeans = ApplyKmeans(km_path)
    k = apply_kmeans.C.shape[1]
    max_cos_sim = -1
    min_dist = np.inf
    for i in range(k):
        for j in range(i + 1, k):
            cos_sim = torch.nn.functional.cosine_similarity(apply_kmeans.C[:, i],
                                                            apply_kmeans.C[:, j],
                                                            dim=0)
            if cos_sim > max_cos_sim:
                max_cos_sim = cos_sim
            dist_ = torch.dist(apply_kmeans.C[:, i], apply_kmeans.C[:, j])
            if dist_ < min_dist:
                min_dist = dist_
    return max_cos_sim, min_dist


if __name__ == "__main__":
    file_path = 'config.yaml'
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

    # label all dataset
    dataset_name = data['dataset']['name']
    feat_dir = data['km_model'][dataset_name]['feat_dir']
    split = 'train_valid'
    n_clusters = data['km_model']['n_clusters']
    km_path = os.path.join(data['km_model'][dataset_name]['km_path'],
                           f'km_model_{n_clusters}/model.km')
    km_folder = os.path.dirname(km_path)
    label_dir = os.path.join(km_folder, 'label_dir/all_dataset')

    dump_label(feat_dir=feat_dir,
               split=split,
               km_path=km_path,
               lab_dir=label_dir)

    # label train/valid dataset
    feat_dir = data['km_model'][dataset_name]['mel_dir']
    split = 'validation'
    splits_path = data['dataset'][dataset_name]['splits']
    label_dir = os.path.join(km_folder, 'label_dir/validation')
    dump_split_label(feat_dir, split, km_path, label_dir, splits_path)

    split = 'training'
    label_dir = os.path.join(km_folder, 'label_dir/training')
    dump_split_label(feat_dir, split, km_path, label_dir, splits_path)
