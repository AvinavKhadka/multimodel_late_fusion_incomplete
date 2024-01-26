import numpy as np
import librosa
import os
import soundfile
import pandas as pd

sample_rate = 8000
window_size = 512
hop_size = 250      # So that there are 64 frames per second

mel_bins = 32
fmin = 20       # Hz
fmax = 14000    # Hz

frames_per_second = sample_rate // hop_size #64 frames per second
audio_duration = 10     
frames_num = frames_per_second * audio_duration #Total temporal frames = 64*10 =640
total_samples = sample_rate * audio_duration


def read_audio(audio_path, target_fs=None):
    (audio, fs) = soundfile.read(audio_path)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
        
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
        
    return audio, fs


def read_metadata(meta_csv):
    df = pd.read_csv(meta_csv)
    df = pd.DataFrame(df)
    
    audio_names = []
    for row in df.iterrows():
        audio_name = row[1]['filename']
        audio_names.append(audio_name)
        
    return audio_names


def pad_truncate_sequence(x, max_len):
    if len(x) < max_len:
        return np.concatenate((x, np.zeros(max_len - len(x))))
    else:
        return x[0 : max_len]


class LogMelExtractor(object):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax):
        
        self.window_size = window_size
        self.hop_size = hop_size
        self.window_func = np.hanning(window_size)
        
        self.melW = librosa.filters.mel(
            sr=sample_rate, 
            n_fft=window_size, 
            n_mels=mel_bins, 
            fmin=fmin, 
            fmax=fmax).T
        '''(n_fft // 2 + 1, mel_bins)'''

    def transform(self, audio):    
        window_size = self.window_size
        hop_size = self.hop_size
        window_func = self.window_func
        
        # Compute short-time Fourier transform
        stft_matrix = librosa.core.stft(
            y=audio, 
            n_fft=window_size, 
            hop_length=hop_size, 
            window=window_func, 
            center=True, 
            dtype=np.complex64, 
            pad_mode='reflect').T
        '''(N, n_fft // 2 + 1)'''
    
        # Mel spectrogram
        mel_spectrogram = np.dot(np.abs(stft_matrix) ** 2, self.melW)
        
        # Log mel spectrogram
        logmel_spectrogram = librosa.core.power_to_db(mel_spectrogram, ref=1.0, amin=1e-10, top_db=None)
        logmel_spectrogram = logmel_spectrogram.astype(np.float32)
               
        return logmel_spectrogram


dataset_dir = 'D:\\my\\audio\\train\\'
audios_dir = os.path.join(dataset_dir, 'synthetic')

metadata_path = 'D:\\my\\metadata\\train\\synthetic.csv'

IMG_DIR = 'D:\\my\\'

audio_names = read_metadata(metadata_path)
#print("The row value:", audio_names)


# Feature extractor
feature_extractor = LogMelExtractor(sample_rate=sample_rate, window_size=window_size, 
                                    hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax)


import pylab
import os
from librosa import display
from matplotlib import cm
import matplotlib.pyplot as plt

for (n, audio_name) in enumerate(audio_names):
    audio_path = os.path.join(audios_dir, audio_name)
    print(n, audio_path)
    
    # Read audio
    (audio, _) = read_audio(audio_path=audio_path, target_fs=sample_rate)
    
    # Pad or truncate audio recording
    audio = pad_truncate_sequence(audio, total_samples)
    
    # Extract feature
    feature = feature_extractor.transform(audio)
    
    # Remove the extra frames caused by padding zero
    feature = feature[0 : frames_num]
    
    # Plotting the wav_to_png_Spectrogram and save as JPG without axes (just the image)
    pylab.figure(figsize=(32,320))
    pylab.axis('off') 
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
    librosa.display.specshow(feature, cmap=cm.jet)
    #pylab.savefig(IMG_DIR + n[:-4]+'.jpg', bbox_inches=None, pad_inches=0)
    pylab.savefig(IMG_DIR + audio_name+'.jpg', bbox_inches=None, pad_inches=0)
    pylab.close()
