# Is Self-Supervised Learning Enough to Fill in the Gap? A Study on Speech Inpainting

### Ihab Asaad, Maxime Jacquelin, Olivier Perrotin, Laurent Girin, Thomas Hueber

This paper investigates using an unsupervised SSL model to inpaint speech signals.

**Abstract :**
Speech inpainting consists in reconstructing corrupted or missing speech segments using surrounding context, a process that closely resembles the pretext tasks in Self-Supervised Learning (SSL) for speech encoders. This study investigates using SSL-trained speech encoders for inpainting without any additional training beyond the initial pretext task, and simply adding a decoder to generate a waveform. We compare this approach to supervised fine-tuning of speech encoders for a downstream task---here, inpainting. Practically, we integrate HuBERT as the SSL encoder and HiFi-GAN as the decoder in two configurations: (1) fine-tuning the decoder to align with the frozen pre-trained encoder's output and (2) fine-tuning the encoder for an inpainting task based on a frozen decoder's input. Evaluations are conducted under single- and multi-speaker conditions using in-domain datasets and out-of-domain datasets (including unseen speakers, diverse speaking styles, and noise). Both informed and blind inpainting scenarios are considered, where the position of the corrupted segment is either known or unknown. The proposed SSL-based methods are benchmarked against several baselines, including a text-informed method combining automatic speech recognition with zero-shot text-to-speech synthesis. Performance is assessed using objective metrics and perceptual evaluations. The results demonstrate that both approaches outperform baselines, successfully reconstructing speech segments up to 200 ms, and sometimes up to 400 ms. Notably, fine-tuning the SSL encoder achieves more accurate speech reconstruction in single-speaker settings, while a pre-trained encoder proves more effective for multi-speaker scenarios. This demonstrates that an SSL pretext task can transfer to speech inpainting, enabling successful speech reconstruction with a pre-trained encoder.

Visit our [demo website](http://www.ultraspeech.com/demo/csl_2025_inpainting/) for audio samples.

## Pre-requisites:
- Python >=3.8 (Our python version is 3.8.10)
If there is any conficlts in the libraries versions in requirements.txt, please let us know.
- GPUs used: Single GPU of type NVIDIA Quadro RTX 8000.

- Create a virtual environment.

- Download the project and run : 'pip install -e .' to install the libarary in editable mode.

Go to `Inpainting` folder to train/predict.

## Usage
You can use our [demo](speech_inpainting_demo.ipynb) to test our methods, but you need to download the pre-trained [models](https://huggingface.co/jacquelm/speech-inpainting/tree/main) first. 

## Acknowledgements
This implementation uses code from the following repos:  

- [HiFi-GAN](https://github.com/jik876/hifi-gan)
- [Jukebox](https://github.com/openai/jukebox)
- [Fairseq](https://github.com/facebookresearch/fairseq)
- [Speech-Resynthesis](https://github.com/facebookresearch/speech-resynthesis)