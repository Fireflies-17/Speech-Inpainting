# This script is to create a dictionary of dictionary of lists:
# the first key is the wav name, and its value is a dictionary of (mask length as key, list of mask positions)
# we will create 20 random mask postions for each mask length of each wave.

import os
import random
import json
import yaml
import librosa
from tqdm import tqdm

file_path = 'predict.yaml'
with open(file_path, 'r') as file:
    data = yaml.safe_load(file)

dataset_name = data['training_config']['dataset']
mask_lengths = data['wavs']['mask_lengths']
wavs_path = data['wavs'][dataset_name]['path']
validation_txt = data['wavs'][dataset_name]['validation_txt']

with open(validation_txt, "r") as file:
    lines = file.readlines()
result_dict = {}
for line in tqdm(lines):
    wave_name, _ = line.strip().split("|")
    wave_path = os.path.join(wavs_path, wave_name+'.wav')

    wav, sr = librosa.load(wave_path, sr=16000)
    wav_length = len(wav)
    wave_dict = {}

    for mask_length in mask_lengths:
        random_positions = [random.randint(
            0, wav_length - mask_length*16) for _ in range(20)]
        wave_dict[mask_length] = random_positions

    result_dict[wave_name] = wave_dict

with open("mask_pos_len.json", "w") as output_file:
    json.dump(result_dict, output_file, indent=2)
