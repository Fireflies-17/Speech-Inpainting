import re
import torch
import librosa
import numpy as np
import speech_recognition as sr
from torchmetrics.functional import word_error_rate, char_error_rate
from torch.nn.functional import F
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from pystoi import stoi
from pesq import pesq


class Metrics:
    def __init__(self, asr_model_name, asr_cache_dir, centroids, device):
        self.asr_model_name = asr_model_name
        self.cache_dir = asr_cache_dir
        self.device = device
        self.processor = WhisperProcessor.from_pretrained(
            self.asr_model_name, cache_dir=self.cache_dir)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.asr_model_name, cache_dir=self.cache_dir)
        self.model.config.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language="english", task="transcribe")
        self.model.to(self.device)

        self.center = centroids.unsqueeze(1).cpu()

    def preprocess_text(self, text):
        # Convert to lowercase
        text = text.lower()

        # Remove punctuation using regular expressions
        text = re.sub(r'[^\w\s]', '', text)
        text = text.strip()
        return text

    # Average Cosine Similarity:
    def avg_cosine_sim(self, tensor1, tensor2):
        cosine_sim_ = F.cosine_similarity(
            tensor1-self.center, tensor2-self.center, dim=0)
        avg_cos_sim = cosine_sim_.mean()
        return avg_cos_sim

    def avg_d2_dist(self, tensor1, tensor2):
        log_scale = 20/torch.log(torch.tensor(10))
        tensor1_mean = torch.mean(tensor1, dim=0)
        tensor2_mean = torch.mean(tensor2, dim=0)
        tensor1 = tensor1-tensor1_mean
        tensor2 = tensor2-tensor2_mean
        dists = log_scale * \
            torch.sqrt(torch.mean((tensor1 - tensor2) ** 2, dim=0))
        avg_dist = dists.mean()
        return avg_dist

    def rmse(self, tensor1, tensor2):
        log_scale = 20/torch.log(torch.tensor(10))
        tensor1_mean = torch.mean(tensor1, dim=0)
        tensor2_mean = torch.mean(tensor2, dim=0)
        tensor1 = tensor1-tensor1_mean
        tensor2 = tensor2-tensor2_mean
        rmse_ = log_scale*torch.sqrt(torch.mean((tensor1 - tensor2) ** 2))
        return rmse_

    def recognize_speech(self, audio_path, asr_engine):
        r = sr.Recognizer()
        audio_file = sr.AudioFile(audio_path)

        with audio_file as source:
            audio = r.record(source)

        # Use the ASR engine to recognize speech
        if asr_engine == "google":
            text = r.recognize_google(audio)
        elif asr_engine == "sphinx":
            text = r.recognize_sphinx(audio)
        # Add more ASR engines as per your choice

        return text

    def wer_cer(self, audio_arr, target_text):
        # using whisper small model:
        audio_input = librosa.resample(
            audio_arr, orig_sr=22050, target_sr=16000)
        input_features = self.processor(
            audio_input, sampling_rate=16000, return_tensors="pt").input_features
        # generate token ids
        predicted_ids = self.model.generate(input_features.to(self.device))
        # decode token ids to text
        generated_text = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=False)
        generated_text = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True)[0]

        generated_text = self.preprocess_text(generated_text)
        target_text = self.preprocess_text(target_text)
        num_words = len(target_text.split(' '))
        num_chars = len(target_text)
        wpa = word_error_rate(preds=generated_text, target=target_text)
        cpa = char_error_rate(preds=generated_text, target=target_text)
        if num_chars*cpa > 30:
            print(cpa, generated_text, target_text, num_chars)
        wpa_torch = num_words*wpa
        cpa_torch = num_chars*cpa
        return wpa_torch, cpa_torch, generated_text

    def get_text(self, audio):
        audio_input = librosa.resample(audio, orig_sr=22050, target_sr=16000)
        input_features = self.processor(
            audio_input, sampling_rate=16000, return_tensors="pt").input_features
        # generate token ids
        predicted_ids = self.model.generate(input_features)
        # decode token ids to text
        generated_text = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=False)
        generated_text = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True)[0]

        generated_text = self.preprocess_text(generated_text)
        return generated_text

    def stoi(self, target_wav, pred_wave, sr):
        return stoi(target_wav, pred_wave, sr, extended=True)

    def pesq(self, target_wav, pred_wav, sr):
        return pesq(target_wav, pred_wav, sr)

    def sisdr(self, x_est, x_ref):

        eps = np.finfo(x_est.dtype).eps
        reference = x_ref.reshape(x_ref.size, 1)
        estimate = x_est.reshape(x_est.size, 1)
        Rss = np.dot(reference.T, reference)
        # get the scaling factor for clean sources
        a = (eps + np.dot(reference.T, estimate)) / (Rss + eps)

        e_true = a * reference
        e_res = estimate - e_true

        Sss = (e_true**2).sum()
        Snn = (e_res**2).sum()

        return 10 * np.log10((eps + Sss)/(eps + Snn))