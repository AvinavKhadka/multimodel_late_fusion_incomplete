import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], 'utils'))
import numpy as np
import argparse
import h5py
import librosa
from scipy import signal
import matplotlib.pyplot as plt
import time
import math
import pandas as pd
import random
from moviepy.editor import VideoFileClip
import cv2

from utilities import (create_folder, read_audio, calculate_scalar_of_tensor, 
    pad_truncate_sequence, get_relative_path_no_extension, read_metadata, isnan)
import config


class LogMelExtractor(object):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax):
        '''Log mel feature extractor. 
        
        Args:
          sample_rate: int
          window_size: int
          hop_size: int
          mel_bins: int
          fmin: int, minimum frequency of mel filter banks
          fmax: int, maximum frequency of mel filter banks
        '''
        
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
        '''Extract feature of a singlechannel audio file. 
        
        Args:
          audio: (samples,)
          
        Returns:
          feature: (frames_num, freq_bins)
        '''
    
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
        logmel_spectrogram = librosa.core.power_to_db(
            mel_spectrogram, ref=1.0, amin=1e-10, 
            top_db=None)
        
        logmel_spectrogram = logmel_spectrogram.astype(np.float32)
        
        return logmel_spectrogram


class VideoFeatureExtractor(object):
    def __init__(self, video_fps):
        self.video_fps = video_fps

    def video3d_frames(self, filename, width, height, depth, color=True, skip=True):
        cap = cv2.VideoCapture(filename)
        nframe = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        print("Total no of frame:", nframe)
    
        if skip:
            frames = [x * nframe / depth for x in range(depth)]
        else:
            frames = [x for x in range(depth)]
        print("The no of frame to process:", len(frames))

        framearray = []
        for i in range(depth):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frames[i])
            ret, frame = cap.read()
            frame = cv2.resize(frame, (height, width), interpolation=cv2.INTER_CUBIC)

            if color:
                framearray.append(frame)
            else:
                framearray.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

        cap.release()

        X = np.array(framearray)    #vid3d.video3d(video_dir, color=color, skip=skip)
        # print(X)
        print("--------------------------------------------------")
        if color:
            return np.array(X).transpose((1, 2, 0, 3))
        else:
            return np.array(X).transpose((1, 2, 0, 0))

def labels_to_target(labels, classes_num, lb_to_idx):
    '''Convert labels to target array. 
    E.g., ['Dog', 'Blender'] -> np.array([0, 1, 0, 0, ...])
    
    returns:
      target: (classes_num,)
    '''
    target = np.zeros(classes_num, dtype=np.bool)
    
    for label in labels:
        if not isnan(label):
            classes_id = lb_to_idx[label]
            target[classes_id] = 1
        
    return target


def events_to_target(events, frames_num, classes_num, frames_per_second, lb_to_idx):
    '''Convert events to strongly labelled matrix: (frames_num, classes_num)
    E.g., ['Dog', 'Blender'] -> np.array(
        [[0, 0, ..., 0], 
         [0, 1, ..., 0], 
         ...
         [0, 0, ..., 0]]
         
    Returns:
      target: (frames_num, classes_num)
    '''
    target = np.zeros((frames_num, classes_num), dtype=np.bool)
    
    for event_dict in events:
        if not isnan(event_dict['event']):
            class_id = lb_to_idx[event_dict['event']]
            onset_frame = int(round(event_dict['onset'] * frames_per_second))
            offset_frame = int(round(event_dict['offset'] * frames_per_second)) +1
            target[onset_frame : offset_frame, class_id] = 1
        
    return target

 
def calculate_feature_for_all_audio_files():
    '''Calculate feature of audio files and write out features to a single hdf5 
    file. '''
    
    '''Args:
     dataset_dir: string
     workspace: string
     data_type: 'development' | 'evaluation'
     mini_data: bool, set True for debugging on a small part of data'''
    

    # Arguments & parameters
    dataset_dir = 'files\\'
    workspace = 'files\\'
    data_type = 'train_synthetic'   #choices=['train_weak', 'train_unlabel_in_domain', 'train_synthetic', 'validation']
    # data_type = 'validation'  #choices=['train_weak', 'train_unlabel_in_domain', 'train_synthetic', 'validation']
    mini_data = False
    
    
    sample_rate = config.sample_rate
    window_size = config.window_size
    hop_size = config.hop_size
    mel_bins = config.mel_bins
    fmin = config.fmin
    fmax = config.fmax
    frames_per_second = config.frames_per_second
    frames_num = config.frames_num
    total_samples = config.total_samples
    classes_num = config.classes_num
    lb_to_idx = config.lb_to_idx
    video_fps = config.video_fps
    video_feature_dim = config.video_feature_dim
    
    # Paths    
    if mini_data:
        prefix = 'minidata_'
    else:
        prefix = ''
    
    relative_name = get_relative_path_no_extension(data_type)
    audios_dir = os.path.join(dataset_dir, 'audio', relative_name)
    videos_dir = os.path.join(dataset_dir, 'video', relative_name)
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(videos_dir)


    if data_type == 'validation':
        metadata_path = os.path.join(dataset_dir, 'metadata', 'validation', 
            '{}.csv'.format(relative_name))
    else:
        metadata_path = os.path.join(dataset_dir, 'metadata', 
            '{}.csv'.format(relative_name))
    
    feature_path = os.path.join(workspace, 'features', 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        '{}.h5'.format(relative_name))
    create_folder(os.path.dirname(feature_path))
    
    # Feature extractor
    feature_extractor = LogMelExtractor(
        sample_rate=sample_rate, 
        window_size=window_size, 
        hop_size=hop_size, 
        mel_bins=mel_bins, 
        fmin=fmin, 
        fmax=fmax)
    
    
    video_feature_extractor = VideoFeatureExtractor(video_fps=video_fps)

    # Read metadata
    (audio_dict, has_weak_labels, has_strong_labels) = read_metadata(metadata_path)

    # Extract features and targets
    audio_names = sorted([*audio_dict.keys()])
    # video_names = sorted([*video_dict.keys().replace('.wav', '.avi')])
    video_names = sorted([key.replace('.wav', '.avi') for key in audio_dict.keys()])


    if mini_data:
        random_state = np.random.RandomState(1234)
        random_state.shuffle(audio_names)
        audio_names = audio_names[0 : 10]
    
    print('Extracting features of all audio and video files ...')
    extract_time = time.time()
    
    # Hdf5 file for storing features and targets
    hf = h5py.File(feature_path, 'w')

    hf.create_dataset(
        name='audio_name', 
        data=[audio_name.encode() for audio_name in audio_names], 
        dtype='S64')

    hf.create_dataset(
        name='feature', 
        shape=(0, frames_num, mel_bins), 
        maxshape=(None, frames_num, mel_bins), 
        dtype=np.float32,
        compression='lzf')
    
    hf.create_dataset(
        name='video_name', 
        data=[video_name.encode() for video_name in video_names], 
        dtype='S64')
    
    hf.create_dataset(
        name='video_feature', 
        shape=(0, video_feature_dim, video_feature_dim, video_fps, 3), 
        maxshape=(None, video_feature_dim, video_feature_dim, video_fps, 3), 
        dtype=np.float32,
        compression='lzf')

    if has_weak_labels:
        hf.create_dataset(
            name='weak_target', 
            shape=(0, classes_num), 
            maxshape=(None, classes_num), 
            dtype=np.bool)
            
    if has_strong_labels:
        hf.create_dataset(
            name='strong_target', 
            shape=(0, frames_num, classes_num), 
            maxshape=(None, frames_num, classes_num),
            dtype=np.bool)


    for (n, audio_name) in enumerate(audio_names):
        audio_path = os.path.join(audios_dir, audio_name)
        video_path = os.path.join(videos_dir, audio_name.replace('.wav', '.avi'))
        print(n, audio_path, video_path)

        
        # Read audio
        (audio, _) = read_audio(
            audio_path=audio_path, 
            target_fs=sample_rate)
        
        # Pad or truncate audio recording
        audio = pad_truncate_sequence(audio, total_samples)
        
        # Extract feature
        feature = feature_extractor.transform(audio)
        
        # Remove the extra frames caused by padding zero
        feature = feature[0 : frames_num]

        video_feature = video_feature_extractor.video3d_frames(video_path, video_feature_dim, video_feature_dim, video_fps, color=True, skip=True)

        hf['feature'].resize((n + 1, frames_num, mel_bins))
        hf['feature'][n] = feature

        # hf.create_dataset('video_feature', data=video_feature, dtype=np.float32)
        hf['video_feature'].resize((n + 1, video_feature_dim, video_feature_dim, video_fps, 3))
        hf['video_feature'][n] = video_feature
        
        if has_weak_labels:
            weak_labels = audio_dict[audio_name]['weak_labels']
            hf['weak_target'].resize((n + 1, classes_num))
            hf['weak_target'][n] = labels_to_target(
                weak_labels, classes_num, lb_to_idx)
    
        if has_strong_labels:
            events = audio_dict[audio_name]['strong_labels']
            hf['strong_target'].resize((n + 1, frames_num, classes_num))
            hf['strong_target'][n] = events_to_target(
                events=events, 
                frames_num=frames_num, 
                classes_num=classes_num, 
                frames_per_second=frames_per_second, 
                lb_to_idx=lb_to_idx)
            
    hf.close()
        
    print('Write hdf5 file to {} using {:.3f} s'.format(feature_path, time.time() - extract_time))
    
    
def calculate_scalar():
    
    # Arguments & parameters
    workspace = "files\\"
    mini_data = False   #Set True for debugging on a small part of data.
    data_type = 'train_synthetic'    #choices=['train_weak', 'train_unlabel_in_domain', 'train_synthetic', 'validation']
    assert data_type == 'train_synthetic', 'We only support using train_weak data to calculate scalar. '
       
    # data_type = 'train_weak'  
    # assert data_type == 'train_weak', 'We only support using train_weak data to calculate scalar. '
    
    # data_type = 'validation'  
    # assert data_type == 'validation', 'We only support using train_weak data to calculate scalar. '
    
    mel_bins = config.mel_bins
    frames_per_second = config.frames_per_second
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    # Paths
    if mini_data:
        prefix = 'minidata_'
    else:
        prefix = ''
    
    relative_name = get_relative_path_no_extension(data_type)
    
    feature_path = os.path.join(workspace, 'features', 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        '{}.h5'.format(relative_name))
        
    scalar_path = os.path.join(workspace, 'scalars', 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        '{}.h5'.format(relative_name))    
    create_folder(os.path.dirname(scalar_path))
    
    print('Calculate scalar of all audio and video files in {}'.format(feature_path))
    # Load data
    load_time = time.time()
    
    with h5py.File(feature_path, 'r') as hf:
        audio_features = hf['feature'][:]
        video_features = hf['video_feature'][:] 

        # feature_data = np.array(feature_dataset)
        # print("Shape of 'feature' dataset:", feature_data.shape)

        # Calculate scalars for audio and video features separately
    # features = np.concatenate(features, axis=0)
    (audio_mean, audio_std) = calculate_scalar_of_tensor(audio_features)
    # video_features = np.concatenate(video_features, axis=0)
    (video_mean, video_std) = calculate_scalar_of_tensor(video_features)


    print('Load feature from {}, using {:.3f} s'.format(feature_path, time.time() - load_time))
    # Create the H5 file with the desired structure
    with h5py.File(scalar_path, 'w') as hf:
        audio_group = hf.create_group('audio')
        audio_group.create_dataset('mean', data=audio_mean, dtype=np.float32)
        audio_group.create_dataset('std', data=audio_std, dtype=np.float32)
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        video_group = hf.create_group('video')
        video_group.create_dataset('mean', data=video_mean, dtype=np.float32)
        video_group.create_dataset('std', data=video_std, dtype=np.float32)

    # # Calculate scalar
        
    # features = np.concatenate(features, axis=0)
    # (mean, std) = calculate_scalar_of_tensor(features)
    
    # with h5py.File(scalar_path, 'w') as hf:
    #     hf.create_dataset('mean', data=mean, dtype=np.float32)
    #     hf.create_dataset('std', data=std, dtype=np.float32)
    
    # print('audio features: {}'.format(features.shape))
    print('audio_mean: {}'.format(audio_mean))
    print('audio_std: {}'.format(audio_std))
    
    # print('video features: {}'.format(video_features.shape))
    print('video_mean: {}'.format(video_mean))
    print('video_std: {}'.format(video_std))

    print('Write out scalar to {}'.format(scalar_path))
            

if __name__ == '__main__':
    
    # mode = 'calculate_feature_for_all_audio_and_video_files'
    mode = 'calculate_scalar'
    
    if mode == 'calculate_feature_for_all_audio_and_video_files':
        calculate_feature_for_all_audio_files()
        
    elif mode == 'calculate_scalar':
        calculate_scalar()
        
    else:
        raise Exception('Incorrect arguments!')