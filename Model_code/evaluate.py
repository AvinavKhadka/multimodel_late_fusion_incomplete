import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))

import numpy as np
import time
import logging
import matplotlib.pyplot as plt
#if error in ploting in Linux Server
#matplotlib.use('Agg")
from sklearn import metrics
import datetime
import pickle as cPickle
import sed_eval

from vad import activity_detection
from utilities import (get_filename, read_csv_file_for_sed_eval_tool,read_csv_file_for_sed_eval_tool_submission, inverse_scale, write_submission)
from pytorch_utils import forward
import config
import torch

'''
#Parameter for Hysteresis thresholding
from skimage import filters

low = 0.1
high = 0.35

filters.apply_hysteresis_threshold(edges, low, high)
'''

'''
#Calculate the IOU but the output is not like this
SMOOTH = 1e-6
def calc_iou(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    #outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    
    intersection = (outputs and labels)  # Will be zero if Truth=0 or Prediction=0
    union = (outputs or labels)         # Will be zzero if both are 0
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    
    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    
    return thresholded  # Or thresholded.mean() if you are interested in average across the batch
'''


class Evaluator(object):
    def __init__(self, model, data_generator, cuda=True, verbose=False):
        '''Evaluator to write out submission and evaluate performance. 
        
        Args:
          model: object
          data_generator: object
          cuda: bool
          verbose: bool
        '''
        self.model = model
        self.data_generator = data_generator
        self.cuda = cuda
        self.verbose = verbose
        
        self.frames_per_second = config.frames_per_second
        self.labels = config.labels
        
        
        # Hyper-parameters for predicting events from framewise predictions
        self.sed_params_dict = {
            'audio_tagging_threshold': 0.5, 
            'sed_high_threshold': 0.4, 
            'sed_low_threshold': 0.5, 
            'n_smooth': self.frames_per_second // 4, 
            'n_salt': self.frames_per_second // 4}
        '''
        self.sed_params_dict = {
            'audio_tagging_threshold': 0.4, 
            'sed_high_threshold': 0.9, 
            'sed_low_threshold': 0.2, 
            'n_smooth': self.frames_per_second // 4, 
            'n_salt': self.frames_per_second // 4}
        '''
        
        
    def evaluate(self, data_type, metadata_path, submission_path, 
        max_iteration=None):
        '''Write out submission file and evaluate the performance. 
        
        Args: 
          data_type: 'train' | 'validate'
          metadata_path: string, path of reference csv
          submission_path: string, path to write out submission
          max_iteration: None | int, maximum iteration to run to speed up 
              evaluation
        '''
        # It generates data for evaluation using a data generator (generate_validate method).
        generate_func = self.data_generator.generate_validate(
            data_type=data_type, 
            max_iteration=max_iteration)
        
        # print('generate_func', generate_func)
        

        ################################# 2nd information ########################################
        # Forward
        # It then performs a forward pass using a model (forward method) and obtains output predictions.
        output_dict = forward(
            model=self.model, 
            generate_func=generate_func, 
            cuda=self.cuda, 
            return_target=True)
            
        # Evaluate audio tagging
        # For audio tagging, it calculates the mean average precision (mAP) using scikit-learn's average_precision_score function.

        if 'weak_target' in output_dict:
            weak_target = output_dict['weak_target']
            clipwise_output = output_dict['clipwise_output']
            #print(type(weak_target), type(clipwise_output), weak_target.shape, clipwise_output.shape) #The shape of level 
            average_precision = metrics.average_precision_score(
                weak_target, clipwise_output, average=None)
            mAP = np.mean(average_precision)
            
            logging.info('{} statistics:'.format(data_type))       
            logging.info('    Audio tagging mAP: {:.3f}'.format(mAP))
            
        statistics = {}
        statistics['average_precision'] = average_precision
        
        # If the output dictionary contains information about strong targets (likely related to SED),
        #  it writes the predictions to a submission file using the write_submission function.
        if 'strong_target' in output_dict:
            ##############5th information#####################
            # Write out submission file
            write_submission(output_dict, self.sed_params_dict, submission_path)
            # print("////////////////////////////////////////////////////////////////")
            # print('output_dict', output_dict.keys())
            # print('4444444444444444444444444444submission_path', submission_path)


            # It reads the ground truth annotations and predicted annotations from CSV files for sound event detection.
            # Evaluate SED with official tools
            reference_dict = read_csv_file_for_sed_eval_tool(metadata_path) #Ground truth value
            predict_dict = read_csv_file_for_sed_eval_tool_submission(submission_path)  #Predicted value
            # print('reference_dict', reference_dict)
            # print('predict_dict', predict_dict)  ########priimted hsfdh
        ################## the wrong is here ############################
            # It uses the sed_eval library to compute event-based and segment-based metrics 
            # such as F-score, error rate, deletion rate, and insertion rate.
            # Event & segment based metrics
            event_based_metric = sed_eval.sound_event.EventBasedMetrics(
                event_label_list=config.labels, 
                evaluate_onset=True,
                evaluate_offset=True,
                t_collar=0.200,
                percentage_of_length=0.2)
            # print('event_based_metric', event_based_metric)
            segment_based_metric = sed_eval.sound_event.SegmentBasedMetrics(
                event_label_list=config.labels, 
                time_resolution=0.2)
            
            for audio_name in output_dict['audio_name']:
                if audio_name in reference_dict.keys():
                    ref_list = reference_dict[audio_name]
                else:
                    ref_list = []
                    
                if audio_name in predict_dict.keys():
                    pred_list = predict_dict[audio_name]
                else:
                    pred_list = []
                    
                event_based_metric.evaluate(ref_list, pred_list)
                segment_based_metric.evaluate(ref_list, pred_list)
                # calc_iou(ref_list, pred_list)
            
            event_metrics = event_based_metric.results_class_wise_average_metrics()
            f_measure = event_metrics['f_measure']['f_measure']
            error_rate = event_metrics['error_rate']['error_rate']
            deletion_rate = event_metrics['error_rate']['deletion_rate']
            insertion_rate = event_metrics['error_rate']['insertion_rate']
            
            statistics['event_metrics'] = {'f_measure': f_measure, 
                'error_rate': error_rate, 'deletion_rate': deletion_rate, 
                'insertion_rate': insertion_rate}
            
            logging.info('Event-based, classwise F score: {:.3f}, ER: '
                '{:.3f}, Del: {:.3f}, Ins: {:.3f}'.format(f_measure, 
                error_rate, deletion_rate, insertion_rate))
                    
            segment_metrics = segment_based_metric.results_class_wise_average_metrics()
            f_measure = segment_metrics['f_measure']['f_measure']
            error_rate = segment_metrics['error_rate']['error_rate']
            deletion_rate = segment_metrics['error_rate']['deletion_rate']
            insertion_rate = segment_metrics['error_rate']['insertion_rate']
            
            statistics['segment_metrics'] = {'f_measure': f_measure, 
                'error_rate': error_rate, 'deletion_rate': deletion_rate, 
                'insertion_rate': insertion_rate}
            
            logging.info('Segment based, classwise F score: {:.3f}, ER: '
                '{:.3f}, Del: {:.3f}, Ins: {:.3f}'.format(f_measure, 
                error_rate, deletion_rate, insertion_rate))
            
            '''
            #Compute IOU
            iou_metrics = calc_iou.results_class_wise_average_metrics()
            f_measure = iou_metrics['f_measure']['f_measure']
            error_rate = iou_metrics['error_rate']['error_rate']
            deletion_rate = iou_metrics['error_rate']['deletion_rate']
            insertion_rate = iou_metrics['error_rate']['insertion_rate']
            
            statistics['iou_metrics'] = {'f_measure': f_measure, 
                'error_rate': error_rate, 'deletion_rate': deletion_rate, 
                'insertion_rate': insertion_rate}
            
            logging.info('IOU score: {:.3f}, ER: ''{:.3f}, Del: {:.3f}, Ins: {:.3f}'.format(f_measure, 
                error_rate, deletion_rate, insertion_rate))
            '''
                
            if self.verbose:
                logging.info(event_based_metric)
                logging.info(segment_based_metric)
                
            return statistics
    
    
    #Task-4 original
    def visualize(self, data_type, max_iteration=None):
        '''
        Visualize logmel spectrogram, reference and prediction. 
        
        Args: 
          data_type: 'train' | 'validate'
          max_iteration: None | int, maximum iteration to run to speed up 
              evaluation
        '''
        generate_func = self.data_generator.generate_validate(data_type=data_type, max_iteration=max_iteration)
        
        mel_bins = config.mel_bins
        audio_duration = config.audio_duration
        labels = config.labels
        
        # Forward
        generate_func = self.data_generator.generate_validate(data_type=data_type)
        
        # Forward
        output_dict = forward(
            model=self.model, 
            generate_func=generate_func, 
            cuda=self.cuda, 
            return_input=True, 
            return_target=True)

        (audios_num, frames_num, classes_num) = output_dict['framewise_output'].shape

        for n in range(audios_num):
            print('File: {}'.format(output_dict['audio_name'][n]))
            
            for k in range(classes_num):
                print('{:<20}{:<8}{:.3f}'.format(labels[k], 
                    output_dict['weak_target'][n, k], output_dict['clipwise_output'][n, k]))
                
            event_prediction = np.zeros((frames_num, classes_num))
                
            for k in range(classes_num):
                if output_dict['clipwise_output'][n, k] \
                    > self.sed_params_dict['sed_high_threshold']:
                        
                    bgn_fin_pairs = activity_detection(
                        x=output_dict['framewise_output'][n, :, k], 
                        thres=self.sed_params_dict['sed_high_threshold'], 
                        low_thres=self.sed_params_dict['sed_low_threshold'], 
                        n_smooth=self.sed_params_dict['n_smooth'], 
                        n_salt=self.sed_params_dict['n_salt'])
                    
                    for pair in bgn_fin_pairs:
                        event_prediction[pair[0] : pair[1], k] = 1
            
            # Plot
            fig, axs = plt.subplots(4, 1, figsize=(10, 400))
            logmel = inverse_scale(output_dict['feature'][n], 
                self.data_generator.scalar['audio_mean'], 
                self.data_generator.scalar['audio_std'])
            axs[0].matshow(logmel.T, origin='lower', aspect='auto', cmap='jet')
            if 'strong_target' in output_dict.keys():
                axs[1].matshow(output_dict['strong_target'][n].T, origin='lower', aspect='auto', cmap='jet')
            masked_framewise_output = output_dict['framewise_output'][n] * output_dict['clipwise_output'][n]
            axs[2].matshow(masked_framewise_output.T, origin='lower', aspect='auto', cmap='jet')
            axs[3].matshow(event_prediction.T, origin='lower', aspect='auto', cmap='jet')
            
            # axs[0].set_title('Log mel spectrogram', color='r')
            # axs[1].set_title('Reference sound events', color='r')
            # axs[2].set_title('Framewise prediction', color='b')
            # axs[3].set_title('Eventwise prediction', color='b')
            titles = ['Log mel spectrogram', 'Reference sound events', 'Framewise prediction', 'Eventwise prediction']

            # for i in range(1):
            #     axs[i].set_xticks([0, frames_num])
            #     axs[i].set_xticklabels(['0', '{:.1f} s'.format(audio_duration)])
            #     axs[i].xaxis.set_ticks_position('bottom')
            #     axs[i].set_yticks(np.arange(classes_num))
            #     axs[i].set_yticklabels(labels)
            #     axs[i].yaxis.grid(color='w', linestyle='solid', linewidth=0.2)


            # axs.set_xticks([0, frames_num])
            # axs.set_xticklabels(['0', '{:.1f} s'.format(audio_duration)])
            # axs.xaxis.set_ticks_position('bottom')
            # axs.set_yticks(np.arange(classes_num))
            # axs.set_yticklabels(labels)
            # axs.yaxis.grid(color='w', linestyle='solid', linewidth=0.2)
            
            for i in range(4):
                axs[i].set_title(titles[i], color='r')
                axs[i].set_xticks([0, frames_num])
                axs[i].set_xticklabels(['0', '{:.1f} s'.format(audio_duration)])
                axs[i].xaxis.set_ticks_position('bottom')
                axs[i].set_yticks(np.arange(classes_num))
                axs[i].set_yticklabels(labels)
                axs[i].yaxis.grid(color='w', linestyle='solid', linewidth=0.2)

                axs[i].set_xticks([0, frames_num])
                axs[i].set_xticklabels(['0', '{:.1f} s'.format(audio_duration)])
                axs[i].xaxis.set_ticks_position('bottom')
                axs[i].set_yticks(np.arange(classes_num))
                axs[i].set_yticklabels(labels)
                axs[i].yaxis.grid(color='w', linestyle='solid', linewidth=0.2)

            axs[0].set_ylabel('Mel bins')
            axs[0].set_yticks([0, mel_bins])
            axs[0].set_yticklabels([0, mel_bins])
            
            fig.tight_layout()
            #plt.savefig(str(n) + "fig.jpg") #For saving the figures
            plt.show()
            
            
class StatisticsContainer(object):
    def __init__(self, statistics_path):
        '''Container of statistics during training. 
        
        Args:
          statistics_path: string, path to write out
        '''
        self.statistics_path = statistics_path

        self.backup_statistics_path = '{}_{}.pickle'.format(
            os.path.splitext(self.statistics_path)[0], 
                datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

        self.statistics_list = []

    def append_and_dump(self, iteration, statistics):
        '''Append statistics to container and dump the container. 
        
        Args:
          iteration: int
          statistics: dict of statistics
        '''
        statistics['iteration'] = iteration
        self.statistics_list.append(statistics)

        cPickle.dump(self.statistics_list, open(self.statistics_path, 'wb'))
        cPickle.dump(self.statistics_list, open(self.backup_statistics_path, 'wb'))
        logging.info('    Dump statistics to {}'.format(self.statistics_path))
            