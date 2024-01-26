import h5py
import numpy as np

def read_feature_from_h5(file_path):
    with h5py.File(file_path, 'r') as hf:
        print("Groups in the H5 file:")
        print(list(hf.keys()))

        # if 'audio_name' not in hf:
        #     print("Error: 'feature' dataset not found in the H5 file.")
#         #     return None
############## see the shape in the h5 file ############################
        feature_dataset = hf['audio_name']
        feature_data = np.array(feature_dataset)
        print("Shape of 'audio_name' dataset:", feature_data.shape)

        feature_dataset = hf['feature']
        feature_data = np.array(feature_dataset)
        print("Shape of 'feature' dataset:", feature_data.shape)
       
        # feature_dataset = hf['strong_target']
        # feature_data = np.array(feature_dataset)
        # print("Shape of 'strong_target' dataset:", feature_data.shape)

        # feature_dataset = hf['video_name']
        # feature_data = np.array(feature_dataset)
        # print("Shape of 'video_name' dataset:", feature_data.shape)
        
        # feature_dataset = hf['video_feature']
        # feature_data = np.array(feature_dataset)
        # print("Shape of 'video_feature' dataset:", feature_data.shape)
      
        # feature_dataset = hf['weak_target']
        # feature_data = np.array(feature_dataset)
        # print("Shape of 'weak_target' dataset:", feature_data.shape)

#  ################# see the dataset in the h5 file ############################

        # audio_names = hf['strong_target'][:1]
        # # Print the first 10 entries
        # print("First 10 audio names:", audio_names[:1])

#         audio_names = hf['video_name'][:10]
#         # Print the first 10 entries
#         print("First 10 video names:", audio_names[:10])

if __name__ == '__main__':
    # h5_file_path = r'D:\features_20\files\scalars\logmel_32frames_32melbins\train\synthetic.h5'
    # h5_file_path = r'files\audio\scalars\logmel_32frames_32melbins\train\synthetic.h5'
    # h5_file_path = r'files\audio\scalars\logmel_32frames_32melbins\validation.h5'
    # h5_file_path = 'files\\scalars\\logmel_32frames_32melbins\\validation.h5'
    # h5_file_path = 'files\\features\\logmel_32frames_32melbins\\validation.h5'
    h5_file_path = r'files\features\logmel_32frames_32melbins\train\synthetic.h5'
    
    # h5_file_path = 'files\\audio\\features\\logmel_32frames_32melbins\\train\\synthetic.h5'
    read_feature_from_h5(h5_file_path)
