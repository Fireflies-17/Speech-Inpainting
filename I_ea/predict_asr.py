####################################
#  Speech Inpainting - ASR-TTS baseline 
# T. Hueber - 2024
###################################
import numpy as np 
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import Audio, load_dataset
import soundfile as sf
import librosa
import librosa.display
import pdb
import torch
from TTS.api import TTS
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import matplotlib
matplotlib.use('Qt5Agg')
import pytsmod as tsm # for wsola, and phase vocoder
from vad import EnergyVAD
import os  

# CONFIG 
########
input_filename = "./data/evaluation_200/p310_026/masked.wav" 
speaker_wav_filenames= ["/Users/huebert/data/VCTK/wav48/p310/p310_040.wav","/Users/huebert/data/VCTK/wav48/p310/p310_041.wav","/Users/huebert/data/VCTK/wav48/p310/p310_042.wav","/Users/huebert/data/VCTK/wav48/p310/p310_043.wav","/Users/huebert/data/VCTK/wav48/p310/p310_044.wav"] # for conditioning the TTS with a long audio 
output_dir = './p310_026_200_asr'
#output_dir = 'p244_097_200_asr'
target_sentence =  'We should not be suprised.' # text for p310_355 (target text if not using the ASR output)
mask_pos = np.array([1.5, 1.7])# position of the mask for informed inpainting

# ASR 
asr_model = "openai/whisper-large"

# TTS   
#tts_model = "tts_models/multilingual/multi-dataset/xtts_v2"
tts_model = "tts_models/multilingual/multi-dataset/your_tts"
tts_sr = 16000 #24000 for xtts_v2 

# DTW alignement 
audio_sr = 16000
win_length = 400 
hop_length = 160 
window = 'hann'
n_fft = 512
fmin = 20
fmax = 8000
n_mels = 40

# Crossfade
crossfade = 0.01 # crossfade length in second

# Steps 
step_asr = 1 # if 0 use target_sentence as input to the TTS
step_tts = 1 # if 0 load output_dir/output_tts.wav 
step_build_speaker_wav = 1 # if 1 concat all files listed in speaker_wav_filenames to build the audio signal used to condition the 0-shot TTS, if 0 use the audio to inpaint instead
step_crop_audio = 1 # remove pre-silence to help DTW alignment
step_crossfade = 1 #apply crossfade when pasting the synthetic signal 
step_display = 1 # for debug

#########################################
# RUN 
#####
os.makedirs(output_dir, exist_ok=True)
vad = EnergyVAD()

# Load input file
y_orig, sr_orig = librosa.load(input_filename,sr=audio_sr)
sf.write(output_dir + '/orig.wav', y_orig, audio_sr)

if step_asr: 
    # load model and processor
    print("Initializing ASR ...\t")
    processor = WhisperProcessor.from_pretrained(asr_model)
    model = WhisperForConditionalGeneration.from_pretrained(asr_model)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")

    # Extract features
    input_features = processor(y_orig, sampling_rate=audio_sr, return_tensors="pt").input_features

    # generate token ids
    predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
    transcription = processor.batch_decode(predicted_ids)
    
    # Parse asr output
    print('ASR output %s' % transcription)
    transcription2= ' '.join(str(transcription).split()[1:-1]);
    transcription2 = transcription2 + ' ' + str(transcription).split()[-1];
    transcription2 = transcription2[:-2];
    print('ASR parsed output  %s' % transcription2)
else:
    transcription2 = target_sentence



if step_tts: 
    # List available TTS models
    #print(TTS().list_models())

    # Init TTS
    print("Initializing TTS ...\t")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS(tts_model).to(device)

    if step_build_speaker_wav:
        # Build audio conditioning signal 
        audio_data = []

        # Load, resample, and concatenate audio files
        for filename in speaker_wav_filenames:
            y, sr = librosa.load(filename, sr=None)  # Load audio with its original sample rate
            y_resampled = librosa.resample(y, orig_sr=sr, target_sr=audio_sr) 
            audio_data.append(y_resampled)  # Add to the list
            
        concatenated_audio = np.concatenate(audio_data) # Concatenate all audio segments
            
        speaker_wav_filename = output_dir + "/speaker_wav.wav"
        sf.write(speaker_wav_filename, concatenated_audio, audio_sr)
        print(f"Speaker wav filename saved to {speaker_wav_filename}")
    else: 
        speaker_wav_filename = input_filename # use the audio file to inpaint as the speaker_wav for zero-shot TTS
    
    # Run TTS
    print('Synthesizing %s' % transcription2)
    tts.tts_to_file(text=transcription2, speaker_wav=speaker_wav_filename, language="en", file_path=output_dir+"/output_tts.wav")

# Load synthesis
y_synth, sr_synth = librosa.load(output_dir + "/output_tts.wav",sr=tts_sr)
y_synth_resampled = librosa.resample(y_synth, orig_sr=tts_sr, target_sr=audio_sr)
sf.write(output_dir + '/output_tts_resampled.wav', y_synth_resampled, audio_sr)

y_orig_with_silence = np.copy(y_orig)
if step_crop_audio:
    # remove pre-silence      
    voice_activity = vad(y_orig) 
    index_first_speech_frame = np.argmax(voice_activity)
    index_first_speech_sample = index_first_speech_frame*0.02*audio_sr # todo: find why frame_shift can be set in the EnergyVAD object 
    
    pattern_indices = np.where((voice_activity[:-1] == 1) & (voice_activity[1:] == 0))[0]
    if len(pattern_indices) > 0:
        index_last_speech_frame = pattern_indices[-1] 
    else:
        index_last_speech_frame = len(voice_activity)
    index_last_speech_sample = index_last_speech_frame*0.02*audio_sr
    
    y_orig = y_orig[int(index_first_speech_sample):int(index_last_speech_sample)] # overwrite y_orig
    mask_pos -= index_first_speech_frame*0.02 # adjust the position of the mask given the crop 
    sf.write(output_dir + '/orig_cropped.wav', y_orig, audio_sr)

    voice_activity = vad(y_synth_resampled) 
    index_first_speech_frame_tts = np.argmax(voice_activity)
    index_first_speech_sample_tts = index_first_speech_frame_tts*0.02*audio_sr # todo: find why frame_shift can be set in the EnergyVAD object 
    
    pattern_indices = np.where((voice_activity[:-1] == 1) & (voice_activity[1:] == 0))[0]
    if len(pattern_indices) > 0:
        index_last_speech_frame_tts = pattern_indices[-1] 
    else:
        index_last_speech_frame_tts = len(voice_activity)
    index_last_speech_sample_tts = index_last_speech_frame_tts*0.02*audio_sr
    #pdb.set_trace()

   
    y_synth_resampled = y_synth_resampled[int(index_first_speech_sample_tts):int(index_last_speech_sample_tts)] # overwrite y_synth_resampled
    sf.write(output_dir + '/output_tts_resampled.wav', y_synth_resampled, audio_sr)

# generate the code that estimates the signal-to-noise ration (SNR) of y_orig and normalize y_synth_resampled to have the same SNR
#scaling_factor = np.sqrt(np.mean(y_orig ** 2) / np.mean(y_synth_resampled ** 2))
#y_synth_resampled = y_synth_resampled * scaling_factor
#sf.write(output_dir + '/output_tts_resampled_normalized.wav', y_synth_resampled, audio_sr)

# DTW-based alignement
D_orig = np.abs(librosa.stft(y_orig, n_fft=n_fft, win_length=win_length, hop_length=hop_length))**2
#S_orig = librosa.feature.melspectrogram(S=D_orig, y=y_orig, sr=sr_orig, n_mels=n_mels, fmin=fmin, fmax=fmax)
S_orig = librosa.feature.mfcc(y=y_orig, sr=audio_sr, hop_length=hop_length, htk=True)

D_synth = np.abs(librosa.stft(y_synth_resampled, n_fft=n_fft, win_length=win_length, hop_length=hop_length))**2
#S_synth = librosa.feature.melspectrogram(S=D_synth, y=y_synth_resampled, sr=audio_sr, n_mels=n_mels, fmin=fmin, fmax=fmax)
S_synth = librosa.feature.mfcc(y=y_synth_resampled, sr=audio_sr, hop_length=hop_length, htk=True)

# DTW-based time alignment
sss = np.array([[1, 1], [2, 1], [1, 2]])
#sss = np.array([[1, 1], [1, 0], [0, 1]])
#pdb.set_trace()
D, wp = librosa.sequence.dtw(S_orig, S_synth, subseq=False, backtrack=True, global_constraints=False, band_rad=0.75,step_sizes_sigma=sss) 
#D, wp = librosa.sequence.dtw(S_orig, S_synth, subseq=False, backtrack=True, global_constraints=False, band_rad=0.25,step_sizes_sigma=sss)       
wp_s = librosa.frames_to_time(wp, sr=audio_sr, hop_length=hop_length)

# Inpainting (informed)
target_pos= np.array([wp_s[np.abs(wp_s[:, 0] - mask_pos[0]).argmin(),1], wp_s[np.abs(wp_s[:, 0] - mask_pos[1]).argmin(),1] ])

# get the corresponding portion in the synthetic signal
target_signal = y_synth_resampled[int(target_pos[0]*audio_sr):int(target_pos[1]*audio_sr)]
sf.write(output_dir + '/mask_synth.wav', target_signal, audio_sr)

# Adapt the length of this portion using phase vocoder
safe_margin = 1.2 # add safe margin to deal with null signal after WSOLA
alpha = (mask_pos[1]-mask_pos[0])/(target_pos[1]-target_pos[0]) * safe_margin
print("Stretching factor: %f" % alpha)
tmp = tsm.wsola(target_signal, alpha)
target_signal_stretched = tmp[0:int((mask_pos[1]-mask_pos[0])*audio_sr)]
sf.write(output_dir + '/mask_synth_stretched.wav', target_signal_stretched, audio_sr)

# Paste it onto the original signal to inpaint (with crossfade)
mask_pos_sample = (mask_pos*audio_sr).astype(int)
y_orig_inpainted = np.copy(y_orig)


if step_crossfade:
    signal_blank = np.zeros(np.shape(y_orig_inpainted))
    signal_blank[int(mask_pos_sample[0]):int(mask_pos_sample[0])+np.shape(target_signal_stretched)[0]] = target_signal_stretched
    #fade out on the left size of the original signal
    y_orig_inpainted[int(mask_pos_sample[0]-int(crossfade*audio_sr/2)):int(mask_pos_sample[0]+int(crossfade*audio_sr/2))]*=np.linspace(1,0,int(crossfade*audio_sr)) #fadeout
    #fade in on the left size of the original signal
    signal_blank[int(mask_pos_sample[0]-int(crossfade*audio_sr/2)):int(mask_pos_sample[0]+int(crossfade*audio_sr/2))]*=np.linspace(0,1,int(crossfade*audio_sr)) #fadein
    #fade in on the right size of the original signal
    y_orig_inpainted[int(mask_pos_sample[1]-int(crossfade*audio_sr/2)):int(mask_pos_sample[1]+int(crossfade*audio_sr/2))]*=np.linspace(0,1,int(crossfade*audio_sr)) #fadein
    #fade out on the right size of the original signal
    signal_blank[int(mask_pos_sample[1]-int(crossfade*audio_sr/2)):int(mask_pos_sample[1]+int(crossfade*audio_sr/2))]*=np.linspace(1,0,int(crossfade*audio_sr)) #fadeout
    # overlapp-add
    y_orig_inpainted+=signal_blank
else:
    y_orig_inpainted[int(mask_pos_sample[0]):int(mask_pos_sample[0])+np.shape(target_signal_stretched)[0]] = target_signal_stretched

sf.write(output_dir + '/orig_inpainted.wav', y_orig_inpainted, audio_sr)


# generate also inpainted signal with original silence 
y_orig_inpainted_with_silence = y_orig_with_silence 
y_orig_inpainted_with_silence[int(index_first_speech_sample):int(index_last_speech_sample)] = y_orig_inpainted
sf.write(output_dir + '/orig_inpainted_with_silence.wav', y_orig_inpainted_with_silence, audio_sr)

if step_display:
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True, sharey=True, figsize=(8, 4))

    librosa.display.waveshow(y_orig,sr=audio_sr, x_axis='time',ax=ax1)
    ax1.set(title="Original")
    ax1.plot([mask_pos[0],mask_pos[0]],[-1,1],color='r')
    ax1.plot([mask_pos[1],mask_pos[1]],[-1,1],color='r')

    librosa.display.waveshow(y_synth_resampled, sr=audio_sr, x_axis='time',ax=ax2)
    ax2.set(title="TTS: " + transcription2)
    ax2.plot([target_pos[0],target_pos[0]],[-1,1],color='r')
    ax2.plot([target_pos[1],target_pos[1]],[-1,1],color='r')
    ax2.label_outer()

    n_arrows = 50
    for tp1, tp2 in wp_s[::len(wp_s)//n_arrows]:

        con = ConnectionPatch(xyA=(tp1, 0), xyB=(tp2, 0),
                                    axesA=ax1, axesB=ax2,
                                    coordsA='data', coordsB='data',
                                    color='r', linestyle='--',
                                    alpha=0.5)
        ax2.add_artist(con)

    
    librosa.display.waveshow(y_orig_inpainted, sr=audio_sr, x_axis='time',ax=ax3)
    ax3.plot([mask_pos[0],mask_pos[0]],[-1,1],color='r')
    ax3.plot([mask_pos[1],mask_pos[1]],[-1,1],color='r')

    ax3.set(title="Original inpainted")
    ax3.label_outer()
    plt.savefig(output_dir + "/dtw.png", dpi=300, bbox_inches='tight')

    plt.show()

#pdb.set_trace()
###############
