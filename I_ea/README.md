# Training:

## Data Preparation
Go to dataset folder to download the dataset and create labels.

## Fine-Tuning HuBERT
Modify the configs in `config.yaml` which includes: 
- The configs to the dataset used in # Data Preparation.
- The hyperparameters of training (dataset name, max_wav_length, model type...)

then run `python main.py` to train the HuBERT model.

You can start from a checkpoint by modifiying the `model_checkpoint`.

## Fine-Tuning HiFi-GAN

We used the [code of HiFi-GAN paper](https://github.com/jik876/hifi-gan) to fine-tune the pretrained model on the kmeans centroids (the quantized vectors).

Move to `hifi_gan` folder and change the following:
- The dataset name (LJSpeech or VCTK)

- checkpoint_path where the pretrained version is along with the configs. Only [UNIVERSAL_V1](https://drive.google.com/drive/folders/1-eEYTB5Av9jNql0WGBlRoi-WH2J7bp5Y) version has the weights for the discriminator. Download it and modify the checkpoint_path in `train_modified.py` file.

- Change `segment_size` which has to be equal to k*256 (now it is 8192), choosing k = 173 will be equal to 100 frames of mel-spectrogram (since the ratio is 441/256).

- Change `mask_len`: by default it is 20 frames which is 400 ms in length.

- Change `km_path` to the prefitted kmeans model. 

Then fine-tune HiFi-GAN using train_modified.py file instead of train.py as in the [paper](https://github.com/jik876/hifi-gan)

# Prediction:

Change the checkpoints for both HuBERT and HiFi-GAN to the fine-tuned paths.

The prediction is done on a wave from LJspeech validation dataset. Specify the wave path and the segment to be masked in predict.yaml file, then run the predict.py file.

predict.py file load the wave with two different sampling rates: 22.05Khz for the HiFi-gan model and 16 Khz for the HuBERT model.

The 22.05 Khz wave is masked, then its mel spectorgram is cacluated and passed to HiFi-gan.

The 16 Khz wave is masked, preprocessed and passed to a pretrained HuBERT model (with the path model_checkpoint in .yaml file) to predict the codewords of the masked segments. The segement of the mel spectrogram of the masked wave is then replaced by the embedding of these codewords and the resulted mel is passed to HiFi-gan to generate the inpainted wave.

# Metrics:
PESQ, STOI, CER.

In `prediction` folder, you can find an example of masked/inpaited wavs.

# Our Models:
We fine-tuned HuBERT and HiFi-GAN, you can find them [here](https://huggingface.co/jacquelm/speech-inpainting/tree/main/I_ea).
