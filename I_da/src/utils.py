import glob
import gc
import os
import logging
import random
import shutil
from typing import List, Tuple
import matplotlib
import matplotlib.pylab as plt
import scipy.stats as st
import tqdm
import numpy as np
import kaldi_io
import torch
from torch.nn.utils import weight_norm
from pathlib import Path

from src.hubert_feature_reader import HubertFeatureReader


matplotlib.use("Agg")


def get_feature_reader(feature_type):
    """Get the wrapper class to extract features

    Args:
        feature_type (string): "logmel", "hubert" or "cpc"

    Raises:
        NotImplementedError: _description_

    Returns:
        class: wrapper class to run inference on model
    """
    if feature_type == "hubert":
        return HubertFeatureReader
    else:
        raise NotImplementedError(f"{feature_type} is not supported.")


def get_feature_iterator(
    feature_type, checkpoint_path, layer, manifest_path, sample_pct, channel_id
):
    """_summary_

    Args:
        feature_type (_type_): _description_
        checkpoint_path (_type_): _description_
        layer (_type_): _description_
        manifest_path (_type_): _description_
        sample_pct (_type_): _description_
        channel_id (_type_): _description_

    Returns:
        _type_: _description_

    Yields:
        _type_: _description_
    """
    feature_reader_cls = get_feature_reader(feature_type)
    with open(manifest_path, "r") as fp:
        lines = fp.read().split("\n")
        root = lines.pop(0).strip()
        file_path_list = [
            os.path.join(root, line.split("\t")[0]) for line in lines if len(line) > 0
        ]
        if sample_pct < 1.0:
            file_path_list = random.sample(
                file_path_list, int(sample_pct * len(file_path_list))
            )
        num_files = len(file_path_list)
        reader = feature_reader_cls(checkpoint_path=checkpoint_path, layer=layer)

        def iterate():
            for file_path in file_path_list:
                feats = reader.get_feats(file_path, channel_id=channel_id)
                yield feats.cpu().numpy(), file_path

    return iterate, num_files


def get_features(
    feature_type, checkpoint_path, layer, manifest_path, sample_pct, flatten, channel_id
):
    generator, num_files = get_feature_iterator(
        feature_type=feature_type,
        checkpoint_path=checkpoint_path,
        layer=layer,
        manifest_path=manifest_path,
        sample_pct=sample_pct,
        channel_id=channel_id,
    )
    iterator = generator()

    features_list = []
    files_list = []
    for features, file in tqdm.tqdm(iterator, total=num_files):
        features_list.append(features)
        files_list.append(file)

    # Explicit clean up
    del iterator
    del generator
    gc.collect()
    torch.cuda.empty_cache()

    if flatten:
        return np.concatenate(features_list)

    return features_list, files_list


def get_and_dump_features(
    feature_type,
    checkpoint_path,
    layer,
    manifest_path,
    sample_pct,
    flatten,
    out_features_path,
):
    # Feature extraction
    features_batch, temp = get_features(
        feature_type=feature_type,
        checkpoint_path=checkpoint_path,
        layer=layer,
        manifest_path=manifest_path,
        sample_pct=sample_pct,
        flatten=flatten,
        channel_id=None,
    )

    # Save features
    out_dir_path = os.path.dirname(out_features_path)
    os.makedirs(out_dir_path, exist_ok=True)
    shutil.copyfile(
        manifest_path,
        os.path.join(out_dir_path, os.path.basename(manifest_path)),
    )
    np.save(out_features_path, features_batch)

    return features_batch, temp


def get_audio_files(manifest_path: str) -> Tuple[str, List[str], List[int]]:
    fnames, sizes = [], []
    with open(manifest_path, "r") as f:
        root_dir = f.readline().strip()
        for line in f:
            items = line.strip().split("\t")
            assert (
                len(items) > 1
            ), f"File must have two columns separated by tab. Got {line}"
            fnames.append(items[0])
            sizes.append(int(items[1]))
    return root_dir, fnames, sizes


def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def apply_weight_norm(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)


def get_padding(kernel_size, dilation=1):
    """_summary_

    Args:
        kernel_size (_type_): _description_
        dilation (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    return int((kernel_size * dilation - dilation) / 2)


def load_checkpoint(filepath, device):
    """_summary_

    Args:
        filepath (_type_): _description_
        device (_type_): _description_

    Returns:
        _type_: _description_
    """
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def save_checkpoint(filepath, obj):
    """_summary_

    Args:
        filepath (_type_): _description_
        obj (_type_): _description_
    """
    print("Saving checkpoint to {}".format(filepath))
    torch.save(obj, filepath)
    print("Complete.")


def scan_checkpoint(cp_dir, prefix):
    """_summary_

    Args:
        cp_dir (_type_): _description_
        prefix (_type_): _description_

    Returns:
        _type_: _description_
    """
    pattern = os.path.join(cp_dir, prefix + "*")
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]


def build_env(config, config_name, path):
    t_path = os.path.join(path, config_name)
    if config != t_path:
        os.makedirs(path, exist_ok=True)
        shutil.copyfile(config, os.path.join(path, config_name))


def get_logger():
    log_format = "[%(asctime)s] [%(levelname)s]: %(message)s"
    logging.basicConfig(format=log_format, level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger


def parse_speaker(path, method) -> str:
    """Parse file path as speaker name.

    Args:
        path
        method - MethodIdentifier or function for speaker name extraction
    Returns:
        - Speaker name
    """
    if type(path) == str:
        path = Path(path)

    if method == "parent_name":
        return path.parent.name
    elif method == "parent_parent_name":
        return path.parent.parent.name
    elif method == "_":
        return path.name.split("_")[0]
    elif method == "single":
        return "A"
    elif callable(method):
        return method(path)
    else:
        raise NotImplementedError()


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def dynamic2static(feat):
    me = np.mean(feat, 0)

    std = np.std(feat, 0)
    sk = st.skew(feat)
    ku = st.kurtosis(feat)

    return np.hstack((me, std, sk, ku))


def dynamic2statict(feat):
    me = []
    std = []
    sk = []
    ku = []
    for k in feat:
        me.append(np.mean(k, 0))
        std.append(np.std(k, 0))
        sk.append(st.skew(k))
        ku.append(st.kurtosis(k))
    return np.hstack((me, std, sk, ku))


def dynamic2statict_artic(feat):
    me = []
    std = []
    sk = []
    ku = []
    for k in feat:
        if k.shape[0] > 1:
            me.append(np.mean(k, 0))
            std.append(np.std(k, 0))
            sk.append(st.skew(k))
            ku.append(st.kurtosis(k))
        elif k.shape[0] == 1:
            me.append(k[0, :])
            std.append(np.zeros(k.shape[1]))
            sk.append(np.zeros(k.shape[1]))
            ku.append(np.zeros(k.shape[1]))
        else:
            me.append(np.zeros(k.shape[1]))
            std.append(np.zeros(k.shape[1]))
            sk.append(np.zeros(k.shape[1]))
            ku.append(np.zeros(k.shape[1]))

    return np.hstack((np.hstack(me), np.hstack(std), np.hstack(sk), np.hstack(ku)))


def get_dict(feat_mat, IDs):
    uniqueids = np.unique(IDs)
    df = {}
    for k in uniqueids:
        p = np.where(IDs == k)[0]
        featid = feat_mat[p, :]
        df[str(k)] = featid
    return df


def save_dict_kaldimat(dict_feat, temp_file):
    ark_scp_output = (
        "ark:| copy-feats --compress=true ark:- ark,scp:"
        + temp_file
        + ".ark,"
        + temp_file
        + ".scp"
    )
    with kaldi_io.open_or_fd(ark_scp_output, "wb") as f:
        for key, mat in dict_feat.items():
            kaldi_io.write_mat(f, mat, key=key)


def multi_find(s, r):
    s_len = len(s)
    r_len = len(r)
    _complete = []
    if s_len < r_len:
        return -1
    for i in range(s_len):
        # search for r in s until not enough characters are left
        if s[i : i + r_len] == r:
            _complete.append(i)
        else:
            i = i + 1
    return _complete


def fill_when_empty(array):
    if len(array) == 0:
        return np.zeros((0, 1))
    return array
