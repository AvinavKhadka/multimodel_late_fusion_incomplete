import os
#import sys
import numpy as np
import soundfile
import librosa
import h5py
import math
import pandas as pd
#from sklearn import metrics
import logging
#import matplotlib.pyplot as plt

from vad import activity_detection
import config


def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)


def get_filename(path):
    path = os.path.realpath(path)
    name_ext = path.split('/')[-1]
    name = os.path.splitext(name_ext)[0]
    return name


def create_logging(log_dir, filemode):
    create_folder(log_dir)
    i1 = 0

    while os.path.isfile(os.path.join(log_dir, '{:04d}.log'.format(i1))):
        i1 += 1
        
    log_path = os.path.join(log_dir, '{:04d}.log'.format(i1))
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S',
        filename=log_path,
        filemode=filemode)

    # Print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    return logging
    
    
def read_audio(audio_path, target_fs=None):
    (audio, fs) = soundfile.read(audio_path)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
        
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
        
    return audio, fs
    
    
def get_relative_path_no_extension(data_type):
    if data_type == 'train_weak':
        return os.path.join('train', 'weak')
        
    elif data_type == 'train_unlabel_in_domain':
        return os.path.join('train', 'unlabel_in_domain')
        
    elif data_type == 'train_synthetic':
        return os.path.join('train', 'synthetic')
        
    elif data_type == 'validation':
        return os.path.join('validation')
        
    else:
        raise Exception('Incorrect data_type!')
    
    
def pad_truncate_sequence(x, max_len):
    if len(x) < max_len:
        return np.concatenate((x, np.zeros(max_len - len(x))))
    else:
        return x[0 : max_len]
    
    
def calculate_scalar_of_tensor(x):
    if x.ndim == 2:
        axis = 0
    elif x.ndim == 3:
        axis = (0, 1)

    mean = np.mean(x, axis=axis)
    std = np.std(x, axis=axis)

    return mean, std


def load_scalar(scalar_path):
    with h5py.File(scalar_path, 'r') as hf:
        print("scalar_path:", scalar_path)
        audio_mean = hf['audio_mean'][:]
        audio_std = hf['audio_std'][:]
        video_mean = hf['video_mean'][:]
        video_std = hf['video_std'][:]
        
    scalar = {'audio_mean': audio_mean, 'audio_std': audio_std,'video_mean': video_mean, 'video_std': video_std }
    return scalar
    

# def load_scalar(scalar_path):
#     with h5py.File(scalar_path, 'r') as hf:
#         audio_group = hf['audio']
#         video_group = hf['video']

#         audio_scalar = {
#             'mean': audio_group['mean'][:],
#             'std': audio_group['std'][:]
#         }

#         video_scalar = {
#             'mean': video_group['mean'][:],
#             'std': video_group['std'][:]
#         }

#     return audio_scalar, video_scalar


   
def scale(x, mean, std):
    return (x - mean) / std
    
    
def inverse_scale(x, mean, std):
    return x * std + mean


def read_metadata(metadata_path):
#     '''Read metadata csv file. 
#     
#     Returns:
#       data_dict: {audio_name: {'weak_labels': [str, str, ...], 
#                                'strong_labels': [{'event': str, 'onset': float, 'offset': float}, 
#                                                  ...
#                                                  ]}, 
#                   ...
#                   }
#     '''
    df = pd.read_csv(metadata_path)
    #df = pd.read_csv(metadata_path, header=0)
    # print("The row value:", df)

    # Strongly labelled data
    if 'onset' in df.keys():
        onset_array = df['onset']
        offset_array = df['offset']
        event_array = df['event_label']
        
        data_dict = {}
        
        for n, audio_name in enumerate(df['filename']):
            if audio_name in data_dict:                
                data = data_dict[audio_name]                
                
                data['weak_labels'] = list(set(data['weak_labels'] + [event_array[n]]))
                data['strong_labels'].append({
                    'event': event_array[n], 
                    'onset': onset_array[n], 
                    'offset': offset_array[n]}
                    )
                
            else:
                data = {
                    'weak_labels': [event_array[n]], 
                    'strong_labels': [{
                        'event': event_array[n], 
                        'onset': onset_array[n], 
                        'offset': offset_array[n]}]
                    }
                
            data_dict[audio_name] = data
            has_weak_labels = True
            has_strong_labels = True
        
    # Weakly labelled or unlabelled data
    else:
        data_dict = {}
        
        for n, audio_name in enumerate(df['filename']):
        
            data = {}
        
            if 'event_labels' in df.keys():
                data['weak_labels'] = df['event_labels'][n].split(',')
                has_weak_labels = True
            else:
                has_weak_labels = False
            
            data_dict[audio_name] = data
            has_strong_labels = False
            
    return data_dict, has_weak_labels, has_strong_labels
    




    
'''    
def read_metadata(metadata_path):
#     Read metadata csv file. 
#     
#     Returns:
#       data_dict: {audio_name: {'weak_labels': [str, str, ...], 
#                                'strong_labels': [{'event': str, 'onset': float, 'offset': float}, 
#                                                  ...
#                                                  ]}, 
#                   ...
#                   }
#     
    df = pd.read_csv(metadata_path, sep='\t')

    # Strongly labelled data
    if 'event_label' in df.keys():
        onset_array = df['onset']
        offset_array = df['offset']
        event_array = df['event_label']
        
        data_dict = {}
        
        for n, audio_name in enumerate(df['filename']):
            if audio_name in data_dict:                
                data = data_dict[audio_name]                
                
                data['weak_labels'] = list(set(data['weak_labels'] + [event_array[n]]))
                data['strong_labels'].append({
                    'event': event_array[n], 
                    'onset': onset_array[n], 
                    'offset': offset_array[n]}
                    )
                
            else:
                data = {
                    'weak_labels': [event_array[n]], 
                    'strong_labels': [{
                        'event': event_array[n], 
                        'onset': onset_array[n], 
                        'offset': offset_array[n]}]
                    }
                
            data_dict[audio_name] = data
            has_weak_labels = True
            has_strong_labels = True
        
    # Weakly labelled or unlabelled data
    else:
        data_dict = {}
        
        for n, audio_name in enumerate(df['filename']):
        
            data = {}
        
            if 'event_labels' in df.keys():
                data['weak_labels'] = df['event_labels'][n].split(',')
                has_weak_labels = True
            else:
                has_weak_labels = False
            
            data_dict[audio_name] = data
            has_strong_labels = False
            
    return data_dict, has_weak_labels, has_strong_labels
'''   
    
def isnan(x):
    if isinstance(x, float) and math.isnan(x):
        return True
    else:
        return False
        
        
def write_submission(output_dict, sed_params_dict, submission_path):
    '''Write output to submission file. 
    
    Args:
      output_dict: {
          'audio_name': (audios_num), 
          'clipwise_output': (audios_num, classes_num), 
          'framewise_output': (audios_num, frames_num, classes_num)}
      sed_params_dict: {
          'audio_tagging_threshold': float between 0 and 1, 
          'sed_high_threshold': : float between 0 and 1, 
          'sed_low_threshold': : float between 0 and 1, 
          'n_smooth': int, silence between the same sound event shorter than 
              this number will be filled with the sound event
          'n_salt': int, sound event shorter than this number will be removed}
      submission_path: string, path to write out submission
    '''
    (audios_num, frames_num, classes_num) = output_dict['framewise_output'].shape
    # print("audios_num:", audios_num)
    # print("frames_num:", frames_num)
    # print("classes_num:", classes_num)
    frames_per_second = config.frames_per_second
    labels = config.labels
    
    f = open(submission_path, 'w')
    f.write('{}\t{}\t{}\t{}\n'.format(
        'filename', 'onset', 'offset', 'event_label'))
    ####################################### Actual problem ##########################################
    # print("output_dict['clipwise_output']:", output_dict['clipwise_output'][0,0])  
    for n in range(audios_num):
        for k in range(classes_num):
            # print("output dict: 6666666666666", output_dict['clipwise_output'][n, k])
            # print("sed_params_dict['sed_high_threshold']:", sed_params_dict['sed_high_threshold'])
            if output_dict['clipwise_output'][n, k] \
                > sed_params_dict['sed_high_threshold']:
                # print("Condition met for n={}, k={}".format(n, k))
                # print("clipwise_output value:", output_dict['clipwise_output'][n, k])
                # print("sed_high_threshold value:", sed_params_dict['sed_high_threshold'])
            
                bgn_fin_pairs = activity_detection(
                    x=output_dict['framewise_output'][n, :, k], 
                    thres=sed_params_dict['sed_high_threshold'], 
                    low_thres=sed_params_dict['sed_low_threshold'], 
                    n_smooth=sed_params_dict['n_smooth'], 
                    n_salt=sed_params_dict['n_salt'])
                
                # print("bgn_fin_pairs:", bgn_fin_pairs)

                for pair in bgn_fin_pairs:
                    bgn_time = pair[0] / float(frames_per_second)
                    fin_time = pair[1] / float(frames_per_second)
                    f.write('{}\t{}\t{}\t{}\n'.format(
                        output_dict['audio_name'][n], bgn_time, fin_time, 
                        labels[k]))
    f.close()
                
    logging.info('    Write submission file to {}'
        ''.format(submission_path))


#It read the annoted validation file used in training
def read_csv_file_for_sed_eval_tool(path):
    '''Read metadata or submission csv file to list of dict for using sed_eval
    tool to evaluate metrics. 
    
    Returns:
      dict: {audio_name: [{'file': str, 'event_label': str, 'onset': float, 'offset': float}, 
                          ...]
             ...
            }
    '''
    #df = pd.read_csv(path)
    #df = pd.read_csv(path, header=0)
    df = pd.read_csv(path)
    dict = {}
    
    for (n, row) in df.iterrows():
        audio_name = row['filename']
        
        if not isnan(row['event_label']):
            event = {
                'file': row['filename'], 
                'event_label': row['event_label'], 
                'onset': row['onset'], 
                'offset': row['offset']
                }
                
            if audio_name not in dict:
                dict[audio_name] = [event]
                    
            else:
                dict[audio_name].append(event)
        
    return dict


def read_csv_file_for_sed_eval_tool_submission(path):
    '''Read metadata or submission csv file to list of dict for using sed_eval
    tool to evaluate metrics. 
    
    Returns:
      dict: {audio_name: [{'file': str, 'event_label': str, 'onset': float, 'offset': float}, 
                          ...]
             ...
            }
    '''
    #df = pd.read_csv(path)
    #df = pd.read_csv(path, header=0)
    df = pd.read_csv(path, sep='\t')
    dict = {}
    
    for (n, row) in df.iterrows():
        audio_name = row['filename']
        
        if not isnan(row['event_label']):
            event = {
                'file': row['filename'], 
                'event_label': row['event_label'], 
                'onset': row['onset'], 
                'offset': row['offset']
                }
                
            if audio_name not in dict:
                dict[audio_name] = [event]
                    
            else:
                dict[audio_name].append(event)
        
    return dict