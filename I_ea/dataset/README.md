1. Choose the dataset (`LJSpeech` or `VCTK`) in the config.yaml and change the configs accordingly.
For LJSpeech dataset, we used the same splits (training/validation) in [training HiFi-GAN](https://github.com/jik876/hifi-gan/tree/master/LJSpeech-1.1)

2. Run `python preprocessing.py`

3. Follow the same steps done in [HuBERT training from fairseq](https://github.com/facebookresearch/fairseq/tree/main/examples/hubert). You can change the configurations in config.yaml:

    3.1. Mel-Spectrogram Extranction:

    run `python mel_dump.py` to extract and save the mel-spectrogam features.

    3.2. K-means clustering:

    run `python kmeans_learn.py` to fit kmeans with 100 clusters on the extracted features.

    3.3. K-means application:

    run `python km_label.py` to apply the fitted kmeans on training/validation sets to create the labels.