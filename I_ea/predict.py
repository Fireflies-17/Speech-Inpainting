import os
import json
import yaml
import torch
import librosa
from librosa.util import normalize
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.io.wavfile import read
from transformers import AutoProcessor
from dataset.mel_dump import MAX_WAV_VALUE, get_mel
from Inpainting.hifi_gan.inference_modified import extend_mel, load_checkpoint
from hifi_gan.models import Generator
from hifi_gan.env import AttrDict
from Inpainting.model import CustomModel
from main import choose_device
from loss_fn import LossFunction
from metrics import Metrics


def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate


def save_fig(tensor, path, fig_name='orig'):
    fig, ax_image = plt.subplots(1, 1, figsize=(8, 4))

    image = ax_image.imshow(np.array(tensor))
    fig.colorbar(image, ax=ax_image)

    # Output file path and extension (e.g., PNG, JPEG, etc.)
    output_path = os.path.join(path, fig_name+'.png')

    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)


def plot_wave(waveform, sr):
    n = len(waveform)
    duration = len(waveform) / sr

    time = np.linspace(0, duration, num=n)
    random_values = np.random.uniform(low=-0.2, high=0.2, size=n)
    waveform[n//4:n*3//4] = 0
    waveform[n//4:n*3//4] = random_values[n//4:n*3//4]
    # Plot the waveform
    plt.plot(time, waveform)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Waveform')
    plt.ylim([-1, 1])
    # plt.grid(True)
    plt.plot(time[n//4: n*3//4+1], waveform[n//4:n*3//4+1], color='red')
    plt.savefig('waveform_plot.png')


if __name__ == "__main__":
    # Load .yaml file and read the wave:
    file_path = 'predict.yaml'
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

    dataset_name = data['training_config']['dataset']
    wave_path = data['wave'][dataset_name]['wave_path']
    save_pred = data['wave'][dataset_name]['save_pred']
    n_clusters = data['km_model']['n_clusters']
    km_model_path = os.path.join(
        data['km_model'][dataset_name]['km_model_path'], f'km_model_{n_clusters}/model.km')
    device = choose_device(data['device']['index'])
    loss_instance = LossFunction(km_model_path, device=device)
    metrics_instance = Metrics(data['ASR_model']['model_name'],
                               data['ASR_model']['cache_dir'], loss_instance.center_, device)
    print("Current device:", device)
    wave_name = wave_path.split('/')[-1].split('.')[0]
    save_pred = os.path.join(save_pred, wave_name)
    if not os.path.exists(save_pred):
        os.makedirs(save_pred)
    wave_22, sr_22 = librosa.load(wave_path, sr=22050)
    wave_16, sr_16 = librosa.load(wave_path, sr=16000)
    assert sr_22 == 22050
    assert sr_16 == 16000

    sf.write(os.path.join(save_pred, 'orig'+'.wav'), wave_16, sr_16)
    mask_ms = int((data['mask']['end_pos_in_sec'] -
                  data['mask']['start_pos_in_sec'])*1000)
    mask_50ms = mask_ms//20  # FIXME: get the hop-size of the STFT analysis
    start_mask = int(data['mask']['start_pos_in_sec']*16000)
    end_mask = int(data['mask']['end_pos_in_sec']*16000)
    mask_pos = start_mask//320

    # check the generated waveform using hifi-gan, and save it:
    wave_22_orig = wave_22.copy()
    wave_22_orig = normalize(wave_22_orig) * 0.95
    norm_wave_22 = torch.FloatTensor(wave_22_orig)
    mel_feats_orig = get_mel(norm_wave_22.unsqueeze(0))
    save_fig(mel_feats_orig.squeeze(0), save_pred, fig_name='orig')

    start_mask_22 = start_mask*sr_22//sr_16
    end_mask_22 = end_mask*sr_22//sr_16
    wave_22_masked = wave_22.copy()
    # wave_22_masked = wave_22_masked / MAX_WAV_VALUE
    wave_22_masked[start_mask_22:end_mask_22] = 0
    wave_22_masked = normalize(wave_22_masked) * 0.95
    norm_wave_22 = torch.FloatTensor(wave_22_masked)
    mel_feats = get_mel(norm_wave_22.unsqueeze(0))
    save_fig(mel_feats.squeeze(0), save_pred, fig_name='masked')
    feats = extend_mel(mel_feats)
    checkpoint_file = data['hifi_gan']['checkpoint_file']
    config_file = os.path.join(os.path.split(
        checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        hifi_gan_data = f.read()
    json_config = json.loads(hifi_gan_data)
    h = AttrDict(json_config)
    torch.manual_seed(h.seed)
    generator = Generator(h).to(device)
    state_dict_g = load_checkpoint(checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])

    generator.eval()
    generator.remove_weight_norm()
    with torch.no_grad():
        y_g_hat = generator(feats.to(device))
        audio = y_g_hat.squeeze()
        audio = audio * MAX_WAV_VALUE
        audio = audio.cpu().numpy().astype('int16')
        sf.write(os.path.join(save_pred, 'hifi_masked'+'.wav'), audio, sr_22)

    # HuBERT
    # preprocessing (masking + normalizing):
    masked_wave_16 = wave_16.copy()
    masked_wave_16[mask_pos*320+80:(mask_pos+mask_50ms)*320+79-80] = 0
    sf.write(os.path.join(save_pred, 'masked'+'.wav'), masked_wave_16, sr_16)

    tokenizer = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
    tokenized_values = tokenizer(
        masked_wave_16, sampling_rate=sr_16, return_attention_mask=True, return_tensors='pt')
    input_values, attention_mask = tokenized_values.input_values, tokenized_values.attention_mask
    input_values = input_values.squeeze(0)
    attention_mask = attention_mask.squeeze(0)

    # loading a trained hubert:
    model_checkpoint = data['hubert_model'][dataset_name]['model_checkpoint']
    model = CustomModel(
        codebook_dim=80, type=data['hubert_model']['type'], load_pretrained=False)  # 'base'

    model.to(device)
    model.load_state_dict(torch.load(model_checkpoint, map_location='cuda'))
    model.eval()
    with torch.no_grad():
        inputs = input_values.unsqueeze(0).to(device)
        attention_masks = attention_mask.unsqueeze(0).to(device)
        mask_pos = torch.tensor(
            [mask_pos], dtype=torch.int).unsqueeze(0).to(device)
        mask_len = torch.tensor(
            [mask_50ms], dtype=torch.int).unsqueeze(0).to(device)
        path2centroids = os.path.join(
            data['km_model'][dataset_name]['path2centroids'], f'km_model_{n_clusters}/label_dir/validation')
        labels = torch.load(os.path.join(
            path2centroids, wave_name+'_labels.pt')).t()
        labels = labels[mask_pos:mask_pos+mask_50ms].unsqueeze(0).to(device)
        outputs = model(inputs, attention_masks)
        values = torch.zeros(
            (mask_pos.shape[0], mask_len[0], outputs.shape[-1])).to(device)
        for i in range(mask_pos.shape[0]):  # considre flatten the tensor
            values[i, :, :] = outputs[i, mask_pos[i]
                :mask_pos[i]+mask_len[i], :]

        # Compute loss
        loss, pred_labels = loss_instance.cos_sim(values, labels)
        cos_sim_pred_target = loss_instance.cos_sim_target_labels(
            pred_labels, labels)
        print("Loss:", loss.item())
        mel_feats = mel_feats.to(device)
        expected_mel = mel_feats.clone()
        exp_mask_mel = loss_instance.all_embeds_t_c[0,
                                                    labels[0, :], :] + loss_instance.center_
        expected_mel[0, :, mask_pos[0]:mask_pos[0] +
                     mask_len[0]] = exp_mask_mel.T
        save_fig(expected_mel.cpu().squeeze(0), save_pred, fig_name='expected')
        exp_feats = extend_mel(expected_mel)

        pred_mels = loss_instance.all_embeds_t_c[0,
                                                 pred_labels[0, :], :] + loss_instance.center_
        # new_mels = values[0,:,:] +center_ 
        mel_feats[0, :, mask_pos[0]:mask_pos[0]+mask_len[0]] = pred_mels.T
        save_fig(mel_feats.cpu().squeeze(0), save_pred, fig_name='inpainted')
        feats = extend_mel(mel_feats)
        print("Target codewords: ", labels)
        print("Predicted codewords: ", pred_labels)
        print(cos_sim_pred_target.shape, cos_sim_pred_target)
        print("Average Cosine Similarity: ", cos_sim_pred_target.mean())

    # Generate the expected inpaiting and the actual inpaiting for comparison:
    with torch.no_grad():
        y_g_hat = generator(exp_feats.to(device))
        audio = y_g_hat.squeeze()
        audio = audio * MAX_WAV_VALUE
        audio = audio.cpu().numpy().astype('int16')
        sf.write(os.path.join(save_pred, 'expected_inpaint'+'.wav'), audio, sr_22)

        y_g_hat = generator(feats.to(device))
        audio = y_g_hat.squeeze()
        audio = audio * MAX_WAV_VALUE
        audio = audio.cpu().numpy().astype('int16')
        sf.write(os.path.join(save_pred, 'inpainted'+'.wav'), audio, sr_22)
