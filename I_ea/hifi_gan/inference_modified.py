from __future__ import absolute_import, division, print_function, unicode_literals
import os
import argparse
import json
import glob
import joblib
import numpy as np
import torch
import torch.nn.functional as F
from scipy.io.wavfile import write
from .env import AttrDict
from .meldataset import mel_spectrogram, MAX_WAV_VALUE, load_wav
from .models import Generator


def extend_mel(spec):
    tensor_n2m = F.interpolate(spec.unsqueeze(0), scale_factor=(
        1, 441/256), mode='bilinear', align_corners=False)
    return tensor_n2m.squeeze(0)


h = None
device = None


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath)  # , map_location=device)
    print("Complete.")
    return checkpoint_dict


def get_mel(x):
    return mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate, 441, h.win_size, h.fmin, h.fmax)


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


class ApplyKmeans(object):
    def __init__(self, km_path):
        self.km_model = joblib.load(km_path)
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np ** 2).sum(0, keepdims=True)

        self.C = torch.from_numpy(self.C_np)
        self.Cnorm = torch.from_numpy(self.Cnorm_np)
        if torch.cuda.is_available():
            self.C = self.C.cuda()
            self.Cnorm = self.Cnorm.cuda()

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            dist = (
                x.pow(2).sum(1, keepdim=True)
                - 2 * torch.matmul(x, self.C)
                + self.Cnorm
            )
            return dist.argmin(dim=1).cpu().numpy()
        else:
            dist = (
                (x ** 2).sum(1, keepdims=True)
                - 2 * np.matmul(x, self.C_np)
                + self.Cnorm_np
            )
            return np.argmin(dist, axis=1)


def replace_mel_cluster(wave, model_path, clustering='kmean'):
    if clustering == 'kmean':
        # load kmean pretrained model, and replace some timestamp mels with the corresponding embedding of the closest centroid:
        apply_kmeans = ApplyKmeans(model_path)
        new_wav = wave.clone()
        tr_wave = wave.transpose(1, 2)
        init_idx = 50
        len_ = 200  # 25*20ms = 0.5 s
        for idx in range(init_idx, init_idx+len_):
            feat = tr_wave[0, idx, :].unsqueeze(0).cuda()
            lab = apply_kmeans(feat).tolist()
            mel_centroids = apply_kmeans.C[:, lab[0]]
            new_wav[0, :, idx] = mel_centroids
        print("finished successfully")
        return new_wav[0, :80, :].unsqueeze(0)


def inference(a):
    generator = Generator(h).to(device)

    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])

    filelist = os.listdir(a.input_wavs_dir)

    generator.eval()
    generator.remove_weight_norm()
    with torch.no_grad():
        for i, filname in enumerate(filelist):
            filname = 'LJ001-0001_22k.wav'
            wav, sr = load_wav(os.path.join(a.input_wavs_dir, filname))
            wav = wav / MAX_WAV_VALUE

            max_length = 300*320+79
            x_ = max_length*22050//16000
            if x_ > len(wav):
                padded_wav = np.pad(wav, (0, x_ - len(wav)), mode='constant')
            else:
                padded_wav = wav
            x = torch.FloatTensor(padded_wav)
            x = get_mel(x[:x_].unsqueeze(0))

            cluster_model_path = '/localdata/asaadi/LJSpeech/LJSpeech-1.1/km_model_100/model_80_100.km'
            print("X shape before replacing with kmean: ",
                  x.shape, x.mean(), x.std())
            x = replace_mel_cluster(x, cluster_model_path, clustering='kmean')
            print("X shape after replacing: ", x.shape, x.mean(), x.std())
            x = extend_mel(x).to(device)
            print("X shape after extend: ", x.shape)
            y_g_hat = generator(x)
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')

            output_file = os.path.join(a.output_dir, os.path.splitext(filname)[
                                       0] + '_generated.wav')
            write(output_file, h.sampling_rate, audio)
            print(output_file)
            break


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_wavs_dir', default='test_files')
    parser.add_argument('--output_dir', default='generated_files')
    parser.add_argument('--checkpoint_file', required=True)
    a = parser.parse_args()

    config_file = os.path.join(os.path.split(
        a.checkpoint_file)[0], 'config.json')
    print(config_file)
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    inference(a)


if __name__ == '__main__':
    main()
