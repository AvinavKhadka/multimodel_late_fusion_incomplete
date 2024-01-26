import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import argparse
import time
import logging
import torch
import torch.optim as optim

from utilities import (create_folder, get_filename, get_relative_path_no_extension, create_logging, load_scalar)
from data_generator import DataGenerator

from models import Cnn_5layers_AvgPooling ,ConvBlock, Cnn_9layers_AvgPooling,Cnn_9layers_MaxPooling,Cnn_13layers_AvgPooling, C3D_Max,C3D2, LateFusion
import numpy as np
from losses import clipwise_binary_crossentropy, framewise_binary_crossentropy
from evaluate import Evaluator, StatisticsContainer
from pytorch_utils import move_data_to_gpu
import config
import h5py

#Use for GPU selection on server
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def train(args):
    # Arugments & parameters    
    dataset_dir = 'files\\'
    dataset_dir_audio = os.path.join(dataset_dir, 'audio')
    dataset_dir_video = os.path.join(dataset_dir, 'video')
    workspace = 'files\\'
    data_type = 'train_synthetic'  #choices=['train_weak', 'train_unlabel_in_domain', 'train_synthetic', 'validation']
    holdout_fold = 1    #choices=['1', 'none'], help='Set 1 for development and none for training on all data without validation.'
    model_type =  'LateFusion'     #'Cnn_5layers_AvgPooling' #ConvBlock, Cnn_9layers_AvgPooling,Cnn_9layers_MaxPooling,Cnn_13layers_AvgPooling
    loss_type = 'framewise_binary_crossentropy'  #choices=['clipwise_binary_crossentropy', 'framewise_binary_crossentropy']
    batch_size = 20
    cuda = True
    mini_data = False
    
    # for audio
    mel_bins = config.mel_bins
    frames_per_second = config.frames_per_second
    classes_num = config.classes_num
    max_iteration = None      # Number of mini-batches to evaluate on training data
    reduce_lr = True

    # for video
    frames_per_second_video = config.frames_num_video  

    # model_type_a = Cnn_9layers_MaxPooling(classes_num, strong_target_training=False) 
    # model_type_v = C3D2(classes_num) # for video
    # model_type = 'LateFusion'
    
    # Paths
    if mini_data:
        prefix = 'minidata_'
    else:
        prefix = ''
        
    if loss_type == 'clipwise_binary_crossentropy':
        strong_target_training = False
    elif loss_type == 'framewise_binary_crossentropy':
        strong_target_training = True
    else:
        raise Exception('Incorrect argument!')
        
    train_relative_name = get_relative_path_no_extension(data_type)   # join(train,synthetic)
    validate_relative_name = get_relative_path_no_extension('validation')
        
    # audio
    train_hdf5_path = os.path.join(dataset_dir_audio, 'features', 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        '{}.h5'.format(train_relative_name))
    
    # train_hdf5_audio = np.array(train_hdf5_path['audio_data'])
    # train_hdf5_video = np.array(train_hdf5_path['video_data'])
     # video
    train_hdf5_path_video = os.path.join(dataset_dir_video, 'features', 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second_video, mel_bins), 
        '{}.h5'.format(train_relative_name))

    # audio   
    validate_hdf5_path = os.path.join(dataset_dir_audio, 'features', 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        '{}.h5'.format(validate_relative_name))
    
    # video
    validate_hdf5_path_video = os.path.join(dataset_dir_video, 'features', 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second_video, mel_bins), 
        '{}.h5'.format(validate_relative_name))
    
    # audio
    scalar_path = os.path.join(dataset_dir_audio, 'scalars', 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        '{}.h5'.format(train_relative_name))
    
    # video
    scalar_path_video = os.path.join(dataset_dir_video, 'scalars', 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second_video, mel_bins), 
        '{}.h5'.format(train_relative_name))
    
    # audio    
    train_metadata_path = os.path.join(dataset_dir, 'metadata', 
        '{}.csv'.format(train_relative_name))
    
    # # video
    # train_metadata_path_video = os.path.join(dataset_dir_video, 'metadata', 
    #     '{}.csv'.format(train_relative_name))
        
    # audio
    validate_metadata_path = os.path.join(dataset_dir, 'metadata', 'validation', 
        '{}.csv'.format(validate_relative_name))
    
    # video
    # validate_metadata_path_video = os.path.join(dataset_dir_video, 'metadata', 'validation', 
    #     '{}.csv'.format(validate_relative_name))
    
    checkpoints_dir = os.path.join(workspace, 'checkpoints', 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        '{}'.format(train_relative_name), 'holdout_fold={}'.format(holdout_fold), 
        model_type, 'loss_type={}'.format(loss_type))
    create_folder(checkpoints_dir)
    
    temp_submission_path = os.path.join(workspace, '_temp', 'submissions', 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        '{}'.format(train_relative_name), 'holdout_fold={}'.format(holdout_fold), 
        model_type, 'loss_type={}'.format(loss_type), '_temp_submission.csv')
    create_folder(os.path.dirname(temp_submission_path))
    
    validate_statistics_path = os.path.join(workspace, 'statistics', 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        '{}'.format(train_relative_name), 'holdout_fold={}'.format(holdout_fold), 
        model_type, 'loss_type={}'.format(loss_type), 'validate_statistics.pickle')
    create_folder(os.path.dirname(validate_statistics_path))

    logs_dir = os.path.join(workspace, 'logs', mode, 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        '{}'.format(train_relative_name), 'holdout_fold={}'.format(holdout_fold), 
        model_type, 'loss_type={}'.format(loss_type))
    
    create_logging(logs_dir, filemode='w')
    
    logging.info(args)
    
    if cuda:
        logging.info('Using GPU.')
    else:
        logging.info('Using CPU. Set --cuda flag to use GPU.')
    
    # Load scalar
    # audio
    scalar = load_scalar(scalar_path)
    
    # video
    # scalar_video = load_scalar(scalar_path_video)
    
    # Model
    # audio
    Model = eval(model_type)
    model = Model(classes_num)

    # # video
    # Model_v = eval(model_type_v)
    # model_v = Model_v(classes_num, strong_target_training)
    
    if cuda:
        model.cuda()
        
    loss_func = eval(loss_type)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999),
        eps=1e-08, weight_decay=0., amsgrad=True)
    # optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    # Data generator
    data_generator = DataGenerator(
        train_hdf5_path=train_hdf5_path, 
        validate_hdf5_path=validate_hdf5_path,
        holdout_fold=holdout_fold, 
        scalar=scalar, 
        batch_size=batch_size)
    
    # print('data_generator',data_generator.validate_video_indexes)
    # Evaluator
    evaluator = Evaluator(
        model=model, 
        data_generator=data_generator, 
        cuda=cuda, 
        verbose=False)
        
    # Statistics
    validate_statistics_container = StatisticsContainer(validate_statistics_path)
    
    train_bgn_time = time.time()
    iteration = 0

    # Train on mini batches
    for batch_data_dict in data_generator.generate_train():
        
        # Evaluate
        # Every 200 iterations, it evaluates the model's performance on the training set.
        if iteration % 5 == 0:
            logging.info('------------------------------------')
            logging.info('Iteration: {}'.format(iteration))

            train_fin_time = time.time()

            ######################### 1   step############################################
            print('train_fin_time',train_fin_time)
            train_statistics = evaluator.evaluate(
                data_type='train', 
                metadata_path=train_metadata_path, 
                submission_path=temp_submission_path, 
                max_iteration=max_iteration)
            # print("33333333333333333333333333333333333333333333333333333333333333333333333333333333333333333")
            # print('train_statistics',train_statistics)
            
            if holdout_fold != 'none':
                validate_statistics = evaluator.evaluate(
                    data_type='validate', 
                    metadata_path=validate_metadata_path, 
                    submission_path=temp_submission_path, 
                    max_iteration=max_iteration)
                
                validate_statistics_container.append_and_dump(
                    iteration, validate_statistics)


            train_time = train_fin_time - train_bgn_time
            validate_time = time.time() - train_fin_time

            logging.info(
                'Train time: {:.3f} s, validate time: {:.3f} s'
                ''.format(train_time, validate_time))

            train_bgn_time = time.time()

        # Save model
        # Every 1000 iterations (after the initial 1000), it saves the model's checkpoint (state, optimizer, etc.) to a file.
        if iteration % 10 == 0 and iteration > 0:
            checkpoint = {
                'iteration': iteration, 
                'model': model.state_dict(), 
                'optimizer': optimizer.state_dict()}

            checkpoint_path = os.path.join(
                checkpoints_dir, '{}_iterations.pth'.format(iteration))
                
            torch.save(checkpoint, checkpoint_path)
            logging.info('Model saved to {}'.format(checkpoint_path))
            
        # Reduce learning rate
        # Every 200 iterations, it reduces the learning rate by a factor of 0.9.
        if reduce_lr and iteration % 4 == 0 and iteration > 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.4
        
        # Move data to GPU
        for key in batch_data_dict.keys():
            if key in ['feature', 'weak_target', 'strong_target', 'video_feature']:
                batch_data_dict[key] = move_data_to_gpu(batch_data_dict[key], cuda)

        # Train
        # The model is put into training mode (model.train()) and performs a forward pass on the input data.
        model.train()
        batch_output_dict = model({key: batch_data_dict[key] for key in ['feature', 'video_feature']})
  
        # loss
        # loss
        # It calculates the loss based on the model's output and the ground truth data.
        loss = loss_func(batch_output_dict, batch_data_dict)
        

        # Backward
        # The gradients are calculated, and the optimizer updates the model's parameters.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Stop learning
        if iteration == 40:
            break
            
        iteration += 1
        

def inference_validation(args):
    '''Training. Model will be saved after several iterations. 
    
    Args: 
      dataset_dir: string, directory of dataset
      workspace: string, directory of workspace
      data_type: 'train_weak' | 'train_synthetic'
      holdout_fold: '1'
      model_type: string, e.g. 'Cnn_9layers_AvgPooling'
      loss_type: 'clipwise_binary_crossentropy' | 'framewise_binary_crossentropy'
      batch_size: int
      cuda: bool
      visualize: bool
      mini_data: bool, set True for debugging on a small part of data
    '''
    # Arugments & parameters    
    dataset_dir = 'files\\'
    workspace = 'files\\'
    data_type = 'train_synthetic'  #choices=['train_weak', 'train_unlabel_in_domain', 'train_synthetic', 'validation']
    holdout_fold = 1    #choices=['1', 'none'], help='Set 1 for development and none for training on all data without validation.'
    model_type = 'LateFusion' #ConvBlock, Cnn_9layers_AvgPooling,Cnn_9layers_MaxPooling,Cnn_13layers_AvgPooling 
    # model_type = 'Cnn_13layers_AvgPooling' 
    loss_type = 'framewise_binary_crossentropy'  #choices=['clipwise_binary_crossentropy', 'framewise_binary_crossentropy']
    iteration = 40
    batch_size = 20
    cuda = True
    visualize = True
    mini_data = False
    
    mel_bins = config.mel_bins
    frames_per_second = config.frames_per_second
    classes_num = config.classes_num
    
    # Paths
    if mini_data:
        prefix = 'minidata_'
    else:
        prefix = ''
        
    if loss_type == 'clipwise_binary_crossentropy':
        strong_target_training = False
    elif loss_type == 'framewise_binary_crossentropy':
        strong_target_training = True
    else:
        raise Exception('Incorrect argument!')
        
    train_relative_name = get_relative_path_no_extension(data_type)
    validate_relative_name = get_relative_path_no_extension('validation')
    
    validate_metadata_path = os.path.join(dataset_dir, 'metadata', 'validation', 
        '{}.csv'.format(validate_relative_name))
    
    train_hdf5_path = os.path.join(dataset_dir, 'features', 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        '{}.h5'.format(train_relative_name))
    
    validate_hdf5_path = os.path.join(dataset_dir, 'features', 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        '{}.h5'.format(validate_relative_name))
    
    scalar_path = os.path.join(dataset_dir, 'scalars', 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        '{}.h5'.format(train_relative_name))
        
    checkoutpoint_path = os.path.join(workspace, 'checkpoints', 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        '{}'.format(train_relative_name), 'holdout_fold={}'.format(holdout_fold), 
        model_type, 'loss_type={}'.format(loss_type), 
        '{}_iterations.pth'.format(iteration))
    
    submission_path = os.path.join(workspace, 'submissions', 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        '{}'.format(train_relative_name), 'holdout_fold={}'.format(holdout_fold), 
        model_type, 'loss_type={}'.format(loss_type), 'validation_submission.csv')
    create_folder(os.path.dirname(submission_path))
    
    logs_dir = os.path.join(workspace, 'logs',mode, 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        '{}'.format(train_relative_name), 'holdout_fold={}'.format(holdout_fold), 
        model_type, 'loss_type={}'.format(loss_type))
    create_logging(logs_dir, filemode='w')
    logging.info(args)
        
    # Load scalar
    scalar = load_scalar(scalar_path)

    # Load model
    Model = eval(model_type)
    model = Model(classes_num, strong_target_training)
    checkpoint = torch.load(checkoutpoint_path)
    model.load_state_dict(checkpoint['model'])
    
    if cuda:
        model.cuda()
        
    # Data generator
    data_generator = DataGenerator(
        train_hdf5_path=train_hdf5_path,
        validate_hdf5_path=validate_hdf5_path, 
        holdout_fold=holdout_fold, 
        scalar=scalar, 
        batch_size=batch_size)
        
    # Evaluator
    evaluator = Evaluator(
        model=model, 
        data_generator=data_generator, 
        cuda=cuda, 
        verbose=True)

    evaluator.evaluate(
        data_type='validate', 
        metadata_path=validate_metadata_path, 
        submission_path=submission_path)
    
    if visualize:
        evaluator.visualize(data_type='validate')
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Example of parser. ')
    args = parser.parse_args()
    #args.filename = get_filename(__file__)
    
    mode = 'train'
    # mode = 'inference_validation'
    
    if mode == 'train':
        train(args)

    elif mode == 'inference_validation':
        inference_validation(args)

    else:
        raise Exception('Error argument!')