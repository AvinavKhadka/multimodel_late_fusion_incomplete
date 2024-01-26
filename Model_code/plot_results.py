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
    workspace = 'D:\\project_wildlife_sound_30_sec\\model_train\\files\\'
    data_type = 'train_synthetic'  #choices=['train_weak', 'train_unlabel_in_domain', 'train_synthetic', 'validation']
    loss_type = 'framewise_binary_crossentropy'  #choices=['clipwise_binary_crossentropy', 'framewise_binary_crossentropy']
    
    #filename = 'main'
    prefix = ''
    frames_per_second = config.frames_per_second
    mel_bins = config.mel_bins
    holdout_fold = 1
    max_plot_iteration = 20000
    
    iterations = np.arange(0, max_plot_iteration, 200)
    measure_keys = ['event_f_measure', 'segment_f_measure', 'mAP']
    
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
            
        segment_f_measure = np.array([statistics['segment_metrics']['f_measure']
            for statistics in statistics_list])
            
        legend = '{}'.format(model_type)
        
        results = {'mAP': mAP, 'event_f_measure': event_f_measure, 
            'segment_f_measure': segment_f_measure, 'legend': legend}
        
        print('Model type: {}'.format(model_type))
        print('    event_f_measure: {:.3f}'.format(event_f_measure[-1]))
        print('    segment_f_measure: {:.3f}'.format(segment_f_measure[-1]))
        print('    mAP: {:.3f}'.format(mAP[-1]))
        
        return results
    
    # Plot
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
      
    results_dict = {}
    # results_dict['Cnn_5layers_AvgPooling'] = _load_stat('Cnn_5layers_r6t, klm, AvgPooling')
    results_dict['Cnn_9layers_MaxPooling'] = _load_stat('Cnn_9layers_MaxPooling')
    # results_dict['Cnn_13layers_AvgPooling'] = _load_stat('Cnn_13layers_AvgPooling')
    # results_dict['GMSE_SA_NW'] = _load_stat('GMSE_SA_NW')
    # results_dict['Cnn_9layers_AvgPooling_BOOST'] = _load_stat('Cnn_9layers_AvgPooling_BOOST')
    # results_dict['Cnn_9layers_AvgPooling_BOOST_GM_Only'] = _load_stat('Cnn_9layers_AvgPooling_BOOST_GM_Only')
    
    # results_dict['StairNet_CS_ATT_BOOST_GMSE'] = _load_stat('StairNet_CS_ATT_BOOST_GMSE')
    # results_dict['StairNet_NO_ATT_SOFTMAX_ALL'] = _load_stat('StairNet_NO_ATT_SOFTMAX_ALL')
    # results_dict['StairNet_SOLO'] = _load_stat('StairNet_SOLO')
    # #results_dict['CascadeBlock_COW'] = _load_stat('CascadeBlock_COW')
    
    '''
    results_dict = {}
    results_dict['AttentionNet'] = _load_stat('AttentionNet')
    results_dict['AttentionNet_Avg'] = _load_stat('AttentionNet_Avg')
    results_dict['AttentionNet_Max'] = _load_stat('AttentionNet_Max')
    #results_dict['CascadeBlock_COW'] = _load_stat('CascadeBlock_COW')
    
    results_dict['CascadeBlock_COW_MOD'] = _load_stat('CascadeBlock_COW_MOD')
    results_dict['Cnn_9layers_AvgPooling_BOOST_PA'] = _load_stat('Cnn_9layers_AvgPooling_BOOST_PA')
    #results_dict['Cnn_9layers_AvgPooling_BOOST_PA_NN'] = _load_stat('Cnn_9layers_AvgPooling_BOOST_PA_NN')
    results_dict['Cnn_9layers_AvgPooling_BOOST_RESIDUAL'] = _load_stat('Cnn_9layers_AvgPooling_BOOST_RESIDUAL')
    
    #results_dict['Cnn_9layers_AvgPooling_GRU'] = _load_stat('Cnn_9layers_AvgPooling_GRU')
    #results_dict['Cnn_T3ORG_GRU'] = _load_stat('Cnn_T3ORG_GRU')
    #results_dict['DualAttention'] = _load_stat('DualAttention')
    #results_dict['PyramidNet'] = _load_stat('PyramidNet')
    
    results_dict['Residual_COW'] = _load_stat('Residual_COW')
    #results_dict['Residual_COW_Edit'] = _load_stat('Residual_COW_Edit')
    results_dict['ResidualLL'] = _load_stat('ResidualLL')
    #results_dict['ResidualSlim'] = _load_stat('ResidualSlim')
    
    results_dict['Cnn_9layers_AvgPooling_BOOST'] = _load_stat('Cnn_9layers_AvgPooling_BOOST')
    results_dict['Cnn_9layers_AvgPooling'] = _load_stat('Cnn_9layers_AvgPooling')
    results_dict['Cnn_9layers_MaxPooling_BOOST'] = _load_stat('Cnn_9layers_MaxPooling_BOOST')
    results_dict['Cnn_9layers_MaxPooling'] = _load_stat('Cnn_9layers_MaxPooling')
    '''
    
    
        
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
    axs[1].set_ylabel('f_measure')
    axs[2].set_ylabel('mAP')
        
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    '''
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')
    parser.add_argument('--data_type', type=str, choices=['train_weak', 'train_synthetic'], required=True)
    parser.add_argument('--loss_type', type=str, choices=['clipwise_binary_crossentropy', 'framewise_binary_crossentropy'], required=True)

    args = parser.parse_args()
    
    plot_results(args)
    '''
    plot_results()
    