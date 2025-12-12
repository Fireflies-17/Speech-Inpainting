import os
import zipfile
import tarfile
import requests
import yaml
from tqdm import tqdm
import random
import librosa
import soundfile as sf
from collections import defaultdict


def download_dataset(url, output_file):
    """
    Download a dataset from a specified URL and save it to a file.
    """
    if os.path.exists(output_file):
        print("dataset already downloaded.")
    else:
        print("downloading...")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024
            progress_bar = tqdm(total=total_size, unit='B', unit_scale=True)

            with open(output_file, 'wb') as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)

            progress_bar.close()
            print(f"File downloaded successfully as {output_file}")
        else:
            print(
                f"Failed to download file. Status code: {response.status_code}"
            )


def extract(file_path, extract_to):
    """
    Extract the contents of a file to a specified directory.
    """
    if os.path.exists(extract_to):
        print(f"dataset already extracted to {extract_to}.")
    else:
        if file_path.endswith('.zip'):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall('extract_to')
            for file_name in os.listdir(extract_to):
                if file_name.endswith(".zip"):
                    zip_file_path = os.path.join(extract_to, file_name)
                    extraction_path = os.path.join(extract_to, extract_to)
                    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                        zip_ref.extractall(extraction_path)
        else:
            with tarfile.open(file_path, "r:bz2") as tar:
                tar.extractall('.')
        print(f"File extracted successfully as {extract_to}")


def summary(training_txt, validation_txt):
    """
    Summarize the training and validation VCTK datasets.
    """
    uni_training_speakers, uni_validation_speakers = set(), set()
    uni_training_texts, uni_validation_texts = set(), set()
    training_speakers, validation_speakers = [], []
    training_texts, validation_texts = [], []
    with open(training_txt, 'r') as training_file:
        data = training_file.readlines()
        for line in data:
            line = line.strip()
            speaker_textnum, text = line.split('|')
            speaker, _ = speaker_textnum.split('_')
            training_speakers.append(speaker)
            training_texts.append(text)

    with open(validation_txt, 'r') as validation_file:
        data = validation_file.readlines()
        for line in data:
            line = line.strip()
            speaker_textnum, text = line.split('|')
            speaker, _ = speaker_textnum.split('_')
            validation_speakers.append(speaker)
            validation_texts.append(text)

    uni_training_speakers, uni_validation_speakers = set(
        training_speakers), set(validation_speakers)
    uni_training_texts, uni_validation_texts = set(training_texts), set(
        validation_texts)

    print(f"# of texts in training set: ", len(training_texts))
    print(f"# of texts in validation set: ", len(validation_texts))
    print(
        f"{len(validation_texts)/len(training_texts)*100:.2f}% texts as validation"
    )

    print(
        f"# of unique speakers in training set: {len(uni_training_speakers)}")
    print(
        f"# of unique speakers in validation set: {len(uni_validation_speakers)}"
    )
    print(
        f"{len(uni_validation_speakers)/len(uni_training_speakers)*100:.2f}% speakers as validation"
    )

    print(f"# of unique texts in training set: {len(uni_training_texts)}")
    print(f"# of unique texts in validation set: {len(uni_validation_texts)}")
    print(
        f"{len(uni_validation_texts)/len(uni_training_texts)*100:.2f}% unique texts as validation"
    )

    print(
        f"# of Common unique speakers between training&validation: {len(uni_training_speakers&uni_validation_speakers)}"
    )
    print(
        f"# of Common unique texts between training&validation: {len(uni_training_texts&uni_validation_texts)}"
    )


if __name__ == '__main__':
    file_path = 'config.yaml'
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    dataset_config = data['dataset']
    dataset_name = dataset_config['name']
    dataset_url = dataset_config[dataset_name]['url']
    output_file = dataset_config[dataset_name]['out_file']
    extract_to = dataset_config[dataset_name]['extract_to']

    download_dataset(dataset_url, output_file)
    extract(output_file, extract_to)

    # for LJSpeech dataset, the same split as in HiFi-GAN training
    if dataset_name == 'LJSpeech':
        exit()

    txts_path = dataset_config[dataset_name]['txts_path']
    wavs_path = dataset_config[dataset_name]['wavs_path']
    flacs_path = dataset_config[dataset_name]['flacs_path']
    vctk_splits = dataset_config[dataset_name]['splits']
    wavs_sr = dataset_config[dataset_name]['wavs_sr']
    ratio = dataset_config[dataset_name]['ratio']

    if not os.path.exists(vctk_splits):
        os.makedirs(vctk_splits)

    ex_speakers = dataset_config[dataset_name]['execlude_speakers']

    # create all.txt file which contains all wavs' names along with their texts if its .flac is available
    list_txts = os.listdir(
        txts_path)  # speaker p315 already execluded from texts.
    list_flacs = os.listdir(flacs_path)
    all_texts = os.path.join(vctk_splits, 'all.txt')
    if not os.path.exists(all_texts):
        with open(all_texts, 'w') as file:
            for speaker in list_txts:
                if speaker in ex_speakers:
                    continue
                speaker_txt_path = os.path.join(txts_path, speaker)
                speaker_flac_path = os.path.join(flacs_path, speaker)
                list_txts_speaker = os.listdir(speaker_txt_path)
                for txt_speaker in list_txts_speaker:
                    if not os.path.exists(
                            os.path.join(speaker_flac_path,
                                         txt_speaker[:-4] + '_mic1.flac')):
                        continue
                    with open(os.path.join(speaker_txt_path,
                                           txt_speaker)) as file_sp:
                        text_ = file_sp.readlines()[0]
                    file.write(f"{txt_speaker[:-4]}|{text_}")
            print(f"Text file '{file_path}' created successfully.")

    # Count the number of unique texts in VCTK dataset
    set_unique_texts = set()
    with open(all_texts, 'r') as file:
        texts = file.readlines()
        for line in texts:
            text = line.split('|')[1]
            set_unique_texts.add(text)
    print("# of unique texts: ", len(set_unique_texts))

    lst_speakers = os.listdir(flacs_path)
    for speaker in lst_speakers:
        if speaker in ex_speakers or not os.path.isdir(
                os.path.join(flacs_path, speaker)):
            lst_speakers.remove(speaker)

    print("# of speakers: ", len(lst_speakers))

    training_txt = os.path.join(vctk_splits, 'training.txt')
    validation_txt = os.path.join(vctk_splits, 'validation.txt')

    random.seed(0)
    if not dataset_config[dataset_name]['all_speakers'] and not dataset_config[
            dataset_name]['all_texts']:  # the hardest case:
        # create a dict of (text, list of speakers)
        # choose a random speaker for each line.
        text_speakers = defaultdict(list)
        valid_text_speakers = defaultdict(list)
        split_index = int(ratio * len(lst_speakers))
        training_set = lst_speakers[:split_index]
        validation_set = lst_speakers[split_index:]
        for speaker in training_set:
            speaker_txt_path = os.path.join(txts_path, speaker)
            speaker_flac_path = os.path.join(flacs_path, speaker)
            texts = os.listdir(speaker_txt_path)
            for text in texts:
                if not os.path.exists(
                        os.path.join(speaker_flac_path,
                                     text[:-4] + '_mic1.flac')):
                    continue
                text_path = os.path.join(speaker_txt_path, text)
                with open(text_path, 'r') as file:
                    line = file.readlines()[0].strip()
                text_speakers[line].append(text[:-4])

        lst_training_texts = list(text_speakers.keys())
        for speaker in validation_set:
            speaker_txt_path = os.path.join(txts_path, speaker)
            speaker_flac_path = os.path.join(flacs_path, speaker)
            texts = os.listdir(speaker_txt_path)
            for text in texts:
                if not os.path.exists(
                        os.path.join(speaker_flac_path,
                                     text[:-4] + '_mic1.flac')):
                    continue
                text_path = os.path.join(speaker_txt_path, text)
                with open(text_path, 'r') as file:
                    line = file.readlines()[0].strip()
                if line not in lst_training_texts:
                    valid_text_speakers[line].append(text[:-4])

        with open(training_txt, 'w') as training_file:
            for text, speakers in text_speakers.items():
                if dataset_config[dataset_name]['multi_speaker_per_text']:
                    for speaker in speakers:
                        training_file.write(f"{speaker}|{text}\n")
                else:
                    len_ = len(speakers)
                    idx = random.randint(0, len_ - 1)
                    training_file.write(f"{speakers[idx]}|{text}\n")

        with open(validation_txt, 'w') as validation_file:
            for text, speakers in valid_text_speakers.items():
                if dataset_config[dataset_name]['multi_speaker_per_text']:
                    for speaker in speakers:
                        validation_file.write(f"{speaker}|{text}\n")
                else:
                    len_ = len(speakers)
                    idx = random.randint(0, len_ - 1)
                    validation_file.write(f"{speakers[idx]}|{text}\n")

    elif not dataset_config[dataset_name]['all_speakers'] and dataset_config[
            dataset_name]['all_texts']:  # first case
        split_index = int(ratio * len(lst_speakers))
        training_set = lst_speakers[:split_index]
        validation_set = lst_speakers[split_index:]

        text_speakers = defaultdict(list)
        valid_text_speakers = defaultdict(list)
        for speaker in training_set:
            speaker_txt_path = os.path.join(txts_path, speaker)
            speaker_flac_path = os.path.join(flacs_path, speaker)
            texts = os.listdir(speaker_txt_path)
            for text in texts:
                if not os.path.exists(
                        os.path.join(speaker_flac_path,
                                     text[:-4] + '_mic1.flac')):
                    continue
                text_path = os.path.join(speaker_txt_path, text)
                with open(text_path, 'r') as file:
                    line = file.readlines()[0].strip()
                text_speakers[line].append(text[:-4])

        lst_training_texts = list(text_speakers.keys())
        for speaker in validation_set:
            speaker_txt_path = os.path.join(txts_path, speaker)
            speaker_flac_path = os.path.join(flacs_path, speaker)
            texts = os.listdir(speaker_txt_path)
            for text in texts:
                if not os.path.exists(
                        os.path.join(speaker_flac_path,
                                     text[:-4] + '_mic1.flac')):
                    continue
                text_path = os.path.join(speaker_txt_path, text)
                with open(text_path, 'r') as file:
                    line = file.readlines()[0].strip()
                valid_text_speakers[line].append(text[:-4])

        with open(training_txt, 'w') as training_file:
            for text, speakers in text_speakers.items():
                if dataset_config[dataset_name]['multi_speaker_per_text']:
                    for speaker in speakers:
                        training_file.write(f"{speaker}|{text}\n")
                else:
                    len_ = len(speakers)
                    idx = random.randint(0, len_ - 1)
                    training_file.write(f"{speakers[idx]}|{text}\n")

        with open(validation_txt, 'w') as validation_file:
            for text, speakers in valid_text_speakers.items():
                if dataset_config[dataset_name]['multi_speaker_per_text']:
                    for speaker in speakers:
                        validation_file.write(f"{speaker}|{text}\n")
                else:
                    len_ = len(speakers)
                    idx = random.randint(0, len_ - 1)
                    validation_file.write(f"{speakers[idx]}|{text}\n")

    elif dataset_config[dataset_name]['all_speakers'] and not dataset_config[
            dataset_name]['all_texts']:  # second case:
        text_speakers = defaultdict(list)
        for speaker in lst_speakers:
            speaker_txt_path = os.path.join(txts_path, speaker)
            speaker_flac_path = os.path.join(flacs_path, speaker)
            texts = os.listdir(speaker_txt_path)
            for text in texts:
                if not os.path.exists(
                        os.path.join(speaker_flac_path,
                                     text[:-4] + '_mic1.flac')):
                    continue
                text_path = os.path.join(speaker_txt_path, text)
                with open(text_path, 'r') as file:
                    line = file.readlines()[0].strip()
                text_speakers[line].append(text[:-4])

        # split by text:
        unique_texts = list(text_speakers.keys())

        # shuffling:
        random.shuffle(unique_texts)

        split_index = int(ratio * len(unique_texts))
        training_texts = unique_texts[:split_index]
        validation_texts = unique_texts[split_index:]

        with open(training_txt, 'w') as training_file:
            for text in training_texts:
                speakers = text_speakers[text]
                if dataset_config[dataset_name]['multi_speaker_per_text']:
                    for speaker in speakers:
                        training_file.write(f"{speaker}|{text}\n")
                else:
                    len_ = len(speakers)
                    idx = random.randint(0, len_ - 1)
                    training_file.write(f"{speakers[idx]}|{text}\n")

        with open(validation_txt, 'w') as validation_file:
            for text in validation_texts:
                speakers = text_speakers[text]
                if dataset_config[dataset_name]['multi_speaker_per_text']:
                    for speaker in speakers:
                        validation_file.write(f"{speaker}|{text}\n")
                else:
                    len_ = len(speakers)
                    idx = random.randint(0, len_ - 1)
                    validation_file.write(f"{speakers[idx]}|{text}\n")

    summary(training_txt, validation_txt)
    # save the wavs with sr = 22050
    if os.path.exists(wavs_path):
        print(
            f"wavs path already exists. If you have changed the configs, delete {wavs_path} and rerun this script. Otherwise continue to mel_dump.py"
        )
    else:
        os.makedirs(wavs_path)
        if dataset_config[dataset_name]['mel_all_wavs']:
            for speaker in lst_speakers:
                # if speaker in ex_speakers: # already done
                #     print(f'speaker {speaker} excluded')
                #     continue
                speaker_path = os.path.join(flacs_path, speaker)
                if os.path.isdir(speaker_path):
                    flacs = os.listdir(speaker_path)
                    for flac in flacs:
                        flac_name = flac.split('.')[0]
                        if flac_name.endswith('_mic1'):
                            flac_name_mic1 = flac_name[:8]
                            flac_path = os.path.join(speaker_path, flac)
                            flac_audio, sr = sf.read(flac_path)
                            flac_audio, sr = librosa.load(flac_path,
                                                          sr=wavs_sr)
                            sf.write(
                                os.path.join(wavs_path,
                                             flac_name_mic1 + '.wav'),
                                flac_audio, wavs_sr
                            )  # sr = 22050 (to match LJSpeech sampling rate)
        else:
            print("Saving training wavs with sr = 22050")
            with open(training_txt, 'r') as file:
                lines = file.readlines()
                for line in tqdm(lines):
                    line = line.strip()
                    speaker_flac, _ = line.split('|')
                    speaker, _ = speaker_flac.split('_')
                    speaker_path = os.path.join(flacs_path, speaker)
                    flac_path = os.path.join(speaker_path, speaker_flac)
                    # flac_audio, sr = sf.read(flac_path+'_mic1.flac')
                    flac_audio, sr = librosa.load(flac_path + '_mic1.flac',
                                                  sr=wavs_sr)
                    sf.write(os.path.join(wavs_path, speaker_flac + '.wav'),
                             flac_audio, wavs_sr)
            print("Saving validation wavs with sr = 22050")
            with open(validation_txt, 'r') as file:
                lines = file.readlines()
                for line in tqdm(lines):
                    line = line.strip()
                    speaker_flac, _ = line.split('|')
                    speaker, _ = speaker_flac.split('_')
                    speaker_path = os.path.join(flacs_path, speaker)
                    flac_path = os.path.join(speaker_path, speaker_flac)
                    # flac_audio, sr = sf.read(flac_path+'_mic1.flac', sr = wavs_sr)
                    flac_audio, sr = librosa.load(flac_path + '_mic1.flac',
                                                  sr=wavs_sr)
                    sf.write(os.path.join(wavs_path, speaker_flac + '.wav'),
                             flac_audio, wavs_sr)
                    # librosa.output.write_wav(os.path.join(wavs_path, speaker_flac+'.wav'), flac_audio, sr=wavs_sr)
