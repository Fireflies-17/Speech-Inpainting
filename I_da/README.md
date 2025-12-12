## Quick Links
- [Setup](#setup)
- [Training](#training)
- [Inference](#inference)
- [Inpainting](#inpaining)

## Setup

### Data

#### For LJSpeech:
1. Download LJSpeech dataset from [here](https://keithito.com/LJ-Speech-Dataset/) into ```data/LJSpeech``` folder.
2. Downsample audio from 22.05 kHz to 16 kHz and pad
   ```bash
   python ./scripts/preprocess.py \
   --srcdir data/LJSpeech/wavs \
   --outdir data/LJSpeech/wavs_16khz \
   --pad
   ```

#### For VCTK:
1. Download VCTK dataset from [here](https://datashare.ed.ac.uk/handle/10283/3443) into ```data/VCTK``` folder.
2. Downsample audio to 16 kHz, trim trailing silences and pad
   ```bash
   python ./scripts/preprocess.py \
   --srcdir data/VCTK \
   --outdir data/VCTK/wav16_silence_trimmed_padded \
   --trim --pad
   ```

## Training

If you want to use our pre-trained models, please refer to [I_da](https://huggingface.co/jacquelm/speech-inpainting/tree/main/I_da)

For training from scratch, you needs below steps:  

1. Encoder<sub>content</sub> training
2. z_c extraction
3. Encoder<sub>f<sub>o</sub></sub> training
4. z_<sub>f<sub>o</sub></sub> extraction
5. Decoder training

This repository provides some shortcuts. In minimum case, you needs to execute only step 3 and 5.  

Currently, we support the following training schemes:

| Dataset  | Encoder<sub>c</sub> SSL | Dictionary Size | Config Path                               |
| -------- |------------------------ | --------------- | ------------------------------------------|
| LJSpeech | HuBERT                  | 100             | ```configs/LJSpeech/hubert_lut.json``` |
| VCTK     | HuBERT                  | 500             | ```configs/VCTK/hubert_lut.json```     |

### 1. Encoder<sub>content</sub> Training
It's tough work, but you can try SSL by yourself.  
Fortunately, pretrained HuBERT is provided. You can skip this step.

If you hope to train other models, follow the instructions described in the [GSLM code](https://github.com/pytorch/fairseq/tree/master/examples/textless_nlp/gslm). 

For quantizing speech with a given acoustic representation:
```bash
python scripts/cluster_kmeans.py \
--num_clusters number_of_clusters_used_for_kmeans \
--checkpoint_path path_of_pretrained_acoustic_model \
--layer layer_of_acoustic_model_to_extract_features_from \
--manifest_path tab_separated_manifest_of_audio_files_for_training_kmeans \
--out_kmeans_model_path output_path_of_the_kmeans_model
```

### 2. z_c Extraction
For LJSpeech or VCTK, we provide extracted z_c! In this case, you can skip this step.  

If you hope to encode other datasets, follow the instructions described in the [GSLM code](https://github.com/pytorch/fairseq/tree/master/examples/textless_nlp/gslm). 

You can use the following script :
```bash
python scripts/quantize_with_kmeans.py \
--kmeans_model_path path_to_quantization_model (.bin) \
--acoustic_model_path path_to_acoustic_model (.pt) \
--manifest_path path_to_all_files_of_datasatet_in_manuscrit (.tsv) \
--out_quantized_file_path path_to_output_discrete_representation (hubert_km100) \
--extension ".flac"
```

The manifest is a file with all signals and duration. You may use the following script:
```bash
python scripts/create_manifest.py \
--root /path/to/waves \
--dest /manifest/path \
--ext $ext \
--valid-percent $valid
```

To parse HuBERT output:
```bash
python scripts/parse_hubert_codes.py \
--codes hubert_output_file \
--manifest hubert_tsv_file \
--outdir parsed_hubert 
--extension ".flac"
```

### 3. Encoder<sub>f<sub>o</sub></sub> training
Train f<sub>o</sub> VQVAE for Encoder<sub>f<sub>o</sub></sub>:
```bash
python scripts/train_f0_vq.py \
--checkpoint_path checkpoints/lj_f0_vq \
--config configs/LJSpeech/f0_vqvae.json
```
### 4. z_<sub>f<sub>o</sub></sub> extraction
This step is automatically executed in step 5.  

### 5. Decoder training
The final step is below:
```bash
python scripts/train.py \
--checkpoint_path checkpoints/vctk_hubert500 \
--config configs/LJSpeech/hubert_lut.json
```
 
## Inference
To generate, simply run:
```bash
python scripts/inference.py \
--checkpoint_file checkpoints/vctk_hubert500 \
-n 10 \
--output_dir generations
```

To synthesize multiple speakers:
```bash
python scripts/inference.py \
--checkpoint_file checkpoints/vctk_hubert500 \
-n 10 \
--vc \
--input_code_file datasets/VCTK/hubert100/test.txt \
--output_dir generations_multispkr
```

You can also generate with codes from a different dataset:
```bash
python scripts/inference.py \
--checkpoint_file checkpoints/lj_hubert100 \
-n 10 \
--input_code_file datasets/VCTK/hubert100/test.txt \
--output_dir generations_vctk_to_lj
```

## Inpainting
To inpaint, simply run:
```bash
python scripts/inpainting.py \
--acoustic_model_path path_to_acoustic_model (.pt) \
--kmeans_model_path path_to_quantization_model (.bin) \
--checkpoint_file path_to_trained_model (checkpoints/...)\
--manifest_path path_to_all_files_of_datasatet_in_manuscrit (.tsv) \
--output_dir path_to_output_inpainted_signals (data/...) \

-n 10 \
--output_dir generations
```

## Acknowledgements
This code is an unofficial reimplementation of the method described in the INTERSPEECH 2021 paper - Speech Resynthesis from Discrete Disentangled Self-Supervised Representations :  

- [Speech-Resynthesis](https://github.com/facebookresearch/speech-resynthesis)

## Citation
```
@inproceedings{polyak21_interspeech,
  author={Adam Polyak and Yossi Adi and Jade Copet and 
          Eugene Kharitonov and Kushal Lakhotia and 
          Wei-Ning Hsu and Abdelrahman Mohamed and Emmanuel Dupoux},
  title={{Speech Resynthesis from Discrete Disentangled Self-Supervised Representations}},
  year=2021,
  booktitle={Proc. Interspeech 2021},
}
``` 
