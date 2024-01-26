#https://github.com/qiuqiangkong/dcase2019_task3/blob/master/utils/plot_results.py

import argparse
import os
import matplotlib.pyplot as plt
import pickle as cPickle
import numpy as np

import config
from utilities import get_relative_path_no_extension

def plot_results():
    # Arugments & parameters
    workspace = 'D:\\project_wildlife2\\model_train\\files\\'
    data_type = 'train_synthetic'  #choices=['train_weak', 'train_unlabel_in_domain', 'train_synthetic', 'validation']
    loss_type = 'framewise_binary_crossentropy'  #choices=['clipwise_binary_crossentropy', 'framewise_binary_crossentropy']

    prefix = ''
    frames_per_second = config.frames_per_second
    mel_bins = config.mel_bins
    holdout_fold = 1
    max_plot_iteration = 20000
    
    iterations = np.arange(0, max_plot_iteration, 200)
    
    def _load_stat(model_type):
        train_relative_name = get_relative_path_no_extension(data_type)
        
        validate_statistics_path = os.path.join(workspace, 'statistics',
            '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
            '{}'.format(train_relative_name), 'holdout_fold={}'.format(holdout_fold), 
            model_type, 'loss_type={}'.format(loss_type), 'validate_statistics.pickle')
        
        
        statistics_list = cPickle.load(open(validate_statistics_path, 'rb'))
        
        sed_error_rate = np.array([statistics['event_metrics']['error_rate'] 
            for statistics in statistics_list])
            
        sed_f1_score = np.array([statistics['event_metrics']['f_measure'] 
            for statistics in statistics_list])

        legend = '{}'.format(model_type)
        
        results = {'sed_error_rate': sed_error_rate, 'sed_f1_score': sed_f1_score, 'legend': legend}
        
        print('Model type: {}'.format(model_type))
        print('    sed_error_rate: {:.3f}'.format(sed_error_rate[-1]))
        print('    sed_f1_score: {:.3f}'.format(sed_f1_score[-1]))
        
        return results
    
    measure_keys = ['sed_error_rate', 'sed_f1_score']
    
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
        
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
    
    axs[0].set_ylabel('sed_error_rate')
    axs[1].set_ylabel('sed_f1_score')
        
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    
    plot_results()