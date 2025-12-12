import os
import numpy as np
from tqdm import tqdm
import yaml
import torch
from librosa.filters import mel as librosa_mel_fn
from scipy.io.wavfile import read
from librosa.util import normalize
from npy_append_array import NpyAppendArray

num_mels = 80
num_freq = 1025
n_fft = 1024
hop_size = 441
win_size = 1024
padd_ = 312
sampling_rate = 22050
fmin = 0
fmax = 8000
MAX_WAV_VALUE = 32768.0

mel_basis = {}
hann_window = {}


def load_wav(full_path):
    sr, data = read(full_path)
    return data, sr


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size,
                    fmin, fmax):
    """
    Compute the Mel spectrogram of an audio waveform using the provided parameters.

    Args:
        y (torch.Tensor): The input audio waveform tensor.
        n_fft (int): The number of Fourier bins.
        num_mels (int): The number of Mel bands to generate.
        sampling_rate (int): The sampling rate of the audio.
        hop_size (int): The number of samples between consecutive frames.
        win_size (int): The size of the STFT window.
        fmin (float): The minimum frequency for the Mel filter banks.
        fmax (float): The maximum frequency for the Mel filter banks.

    Returns:
        torch.Tensor: The computed Mel spectrogram of the input audio.

    Note:
        This function internally uses a global `mel_basis` dictionary and a `hann_window` dictionary
        to store precomputed Mel basis and Hann window for performance optimization.
    """
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax) + '_' +
                  str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    # y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = torch.nn.functional.pad(y.unsqueeze(1), (padd_, padd_), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y,
                      n_fft,
                      hop_length=hop_size,
                      win_length=win_size,
                      window=hann_window[str(y.device)],
                      center=False,
                      pad_mode='reflect',
                      normalized=False,
                      onesided=True,
                      return_complex=False)

    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))
    spec = torch.matmul(mel_basis[str(fmax) + '_' + str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


def get_mel(x, hop_size=441):
    return mel_spectrogram(x, n_fft, num_mels, sampling_rate, hop_size,
                           win_size, fmin, fmax)


def dump_feature(mel_dir, feat_dir):
    feat_path = f"{feat_dir}/train_valid.npy"
    leng_path = f"{feat_dir}/train_valid.len"

    os.makedirs(feat_dir, exist_ok=False)
    if os.path.exists(feat_path):
        os.remove(feat_path)

    feat_f = NpyAppendArray(feat_path)
    filelist = os.listdir(mel_dir)
    with open(leng_path, "w") as leng_f:
        for mel_spec in tqdm(filelist):
            feat = torch.load(os.path.join(mel_dir, mel_spec))
            feat_f.append(feat.numpy())
            leng_f.write(f"{len(feat)}\n")
    print("finished successfully")


def get_mel_train_valid_dataset(path2wavs, mel_dir, max_length=222621):
    filelist = os.listdir(path2wavs)
    os.makedirs(mel_dir, exist_ok=False)
    for filename in tqdm(filelist):
        audio, sr = load_wav(os.path.join(path2wavs, filename))
        audio = audio / MAX_WAV_VALUE
        audio = normalize(audio) * 0.95
        padded_arr = np.pad(audio, (0, max_length - len(audio)),
                            mode='constant')
        audio = torch.FloatTensor(padded_arr)
        mel_feats = get_mel(audio.unsqueeze(0))
        feats = mel_feats
        feats = feats.squeeze(0).transpose(0, 1).contiguous()
        output_file = os.path.join(mel_dir, filename[:-4] + '.pt')
        torch.save(feats, output_file)


def get_max_len_wavs(path2wavs):
    max_len = 0
    filelist = os.listdir(path2wavs)
    for filename in tqdm(filelist):
        audio, sr = load_wav(os.path.join(path2wavs, filename))
        assert sr == sampling_rate
        if max_len < len(audio):
            max_len = len(audio)
    return max_len


if __name__ == "__main__":
    file_path = 'config.yaml'
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    dataset_name = data['dataset']['name']
    path2wavs = data['dataset'][dataset_name]['wavs_path']
    feat_dir = data['km_model'][dataset_name]['feat_dir']
    mel_dir = data['km_model'][dataset_name]['mel_dir']

    maxlen_wavs = get_max_len_wavs(path2wavs)
    print("Maximum wav length is: ", maxlen_wavs)

    get_mel_train_valid_dataset(path2wavs=path2wavs,
                                mel_dir=mel_dir,
                                max_length=maxlen_wavs)
    dump_feature(mel_dir, feat_dir)
