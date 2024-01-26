import argparse
import os
import matplotlib.pyplot as plt
import pickle as cPickle
import numpy as np
from utilities import get_relative_path_no_extension

import config


def plot_results():
    '''Plot statistics curve of different models. 
    
    Args:
      workspace: string, directory of workspace
      data_type: 'train_weak' | 'train_synthetic'
      loss_type: 'clipwise_binary_crossentropy' | 'framewise_binary_crossentropy'
    '''
    
    # Arugments & parameters
    workspace = 'D:\\project_wildlife2\\model_train\\files\\'
    data_type = 'train_synthetic'  #choices=['train_weak', 'train_unlabel_in_domain', 'train_synthetic', 'validation']
    loss_type = 'framewise_binary_crossentropy'  #choices=['clipwise_binary_crossentropy', 'framewise_binary_crossentropy']
    
    #filename = 'main'
    prefix = ''
    frames_per_second = config.frames_per_second
    mel_bins = config.mel_bins
    holdout_fold = 1
    max_plot_iteration = 20000
    
    iterations = np.arange(0, max_plot_iteration, 200)
    measure_keys = ['event_f_measure', 'mAP']
    
    def _load_stat(model_type):
        train_relative_name = get_relative_path_no_extension(data_type)
        
        validate_statistics_path = os.path.join(workspace, 'statistics',
            '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
            '{}'.format(train_relative_name), 'holdout_fold={}'.format(holdout_fold), 
            model_type, 'loss_type={}'.format(loss_type), 'validate_statistics.pickle')
        
        statistics_list = cPickle.load(open(validate_statistics_path, 'rb'))
        
        average_precisions = np.array([statistics['average_precision']
            for statistics in statistics_list])    # (N, classes_num)
            
        mAP = np.mean(average_precisions, axis=-1)
        
        event_f_measure = np.array([statistics['event_metrics']['f_measure'] 
            for statistics in statistics_list])
         
        legend = '{}'.format(model_type)
        
        results = {'mAP': mAP, 'event_f_measure': event_f_measure, 'legend': legend}
        
        print('Model type: {}'.format(model_type))
        print('    event_f_measure: {:.3f}'.format(event_f_measure[-1]))
        print('    mAP: {:.3f}'.format(mAP[-1]))
        
        return results
    
    # Plot
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
      
    results_dict = {}
    
    results_dict['Cnn_5layers_AvgPooling'] = _load_stat('Cnn_5layers_AvgPooling')
    # results_dict['Cnn_9layers_MaxPooling'] = _load_stat('Cnn_9layers_MaxPooling')
    # results_dict['Proposed_CNN_AvgPooling'] = _load_stat('Proposed_CNN_AvgPooling')
    # results_dict['Proposed_CNN_MaxPooling'] = _load_stat('Proposed_CNN_MaxPooling')
       
    for n, measure_key in enumerate(measure_keys):
        lines = []
        
        for model_key in results_dict.keys():
            #line, = axs[n].plot(results_dict[model_key][measure_key], label=measure_keys[n])
            line, = axs[n].plot(results_dict[model_key][measure_key], label=results_dict[model_key]['legend'])
            lines.append(line)
            
        axs[n].set_title(measure_key)    
        axs[n].legend(handles=lines, loc=4)
        axs[n].set_ylim(0, 1.0)
        axs[n].set_xlabel('Iterations')
        axs[n].grid(color='b', linestyle='solid', linewidth=0.2)
        axs[n].xaxis.set_ticks(np.arange(0, len(iterations), len(iterations) // 4))
        axs[n].xaxis.set_ticklabels(np.arange(0, max_plot_iteration, max_plot_iteration // 4))

    axs[0].set_ylabel('f_measure')
    #axs[1].set_ylabel('f_measure')
    axs[1].set_ylabel('mAP')
        
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_results()
    