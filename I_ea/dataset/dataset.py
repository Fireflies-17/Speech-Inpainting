import os
import librosa
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from transformers import AutoProcessor


class AudioDataset(Dataset):
    """
    Dataset class for audio data.

    Args:
        path2splits (str): Path to splits done by preprocessing.
        path2wavs (str): Path to the directory containing audio files.
        path2pt (str): Path to save processed audio tensors.
        path2centroids (str): Path to centroids.
        max_wav_len (int): Maximum length of audio files.
        min_mask_len (int): Minimum length of the masking region.
        max_mask_len (int): Maximum length of the masking region.
        epoch_num (int): Current epoch number.
        epochs (int): Total number of epochs.
        sr (int): Sampling rate of the audio files. Defaults to 16000.
    """

    def __init__(self, path2splits, path2wavs, path2pt, path2centroids, max_wav_len=161539, min_mask_len=2, max_mask_len=20, epoch_num=0, epochs=1, sr=16000):
        self.path2splits = path2splits
        self.path2wavs = path2wavs
        self.path2centroids = path2centroids
        self.sr = sr
        self.tokenizer = AutoProcessor.from_pretrained(
            "facebook/hubert-large-ls960-ft")
        self.path2pt = path2pt
        self.min_mask_len = min_mask_len
        self.max_mask_len = max_mask_len
        self.epoch_num = epoch_num
        self.epochs = epochs
        self.max_length = max_wav_len
        self.max_len_ms = int((self.max_length-80)/320)
        with open(self.path2splits, 'r') as f:
            audio_files = [line.split('|')[0] for line in f]
        if os.path.exists(self.path2pt):
            print(
                f"the wavs have already been saved as tensor. If you have changed the configs, delete {self.path2pt} and rerun this script. Otherwise continue to mel_dump.py")
        else:
            os.makedirs(self.path2pt)
            print("Saving audios as tensors:")
            pbar = tqdm(total=len(audio_files))
            for audio in audio_files:
                audio_path = os.path.join(self.path2wavs, audio) + '.wav'
                audio_, sr_ = librosa.load(audio_path, sr=self.sr)
                assert sr_ == self.sr
                tokenized_values = self.tokenizer(audio_, sampling_rate=self.sr, padding='max_length',
                                                  max_length=self.max_length, truncation=True, return_attention_mask=True, return_tensors='pt')
                input_values, attention_mask = tokenized_values.input_values, tokenized_values.attention_mask
                torch.save(input_values, os.path.join(
                    self.path2pt, audio + '_values.pt'))
                torch.save(attention_mask, os.path.join(
                    self.path2pt, audio + '_mask.pt'))
                torch.save(torch.tensor(audio_.shape, dtype=torch.int),
                           os.path.join(self.path2pt, audio + '_len.pt'))
                pbar.update(1)
            pbar.close()
        self.pt_paths = [os.path.join(self.path2pt, audio)
                         for audio in audio_files]
        self.len = len(self.pt_paths)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        wave_path = self.pt_paths[idx]
        wave_name = wave_path.split('/')[-1]
        inputvalues_pt, attention_mask_pt, len_pt = torch.load(
            wave_path+'_values.pt'), torch.load(wave_path+'_mask.pt'), torch.load(wave_path+'_len.pt')
        mask_50ms = self.max_mask_len
        max_mask_pos = int((min(len_pt[0], self.max_length)-80)/320)
        mask_pos = torch.randint(0, max_mask_pos-mask_50ms, size=(1,))
        inputvalues_pt = inputvalues_pt.squeeze(0)
        attention_mask_pt = attention_mask_pt.squeeze(0)
        # affected samples are : mask_pos,mask_pos+1, ...., mask_pos+mask_50ms-1
        inputvalues_pt[mask_pos*320+80:(mask_pos+mask_50ms)*320-1] = 0
        labels = torch.load(os.path.join(
            self.path2centroids, wave_name+'_labels.pt')).t()
        labels = labels[mask_pos:mask_pos+mask_50ms]
        return wave_name, inputvalues_pt, attention_mask_pt, mask_pos, mask_50ms, labels
