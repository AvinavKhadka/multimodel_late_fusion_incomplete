
import config
import matplotlib.pyplot as plt
from pytorch_utils import forward
import numpy as np
from vad import activity_detection
from utilities import (get_filename, read_csv_file_for_sed_eval_tool,read_csv_file_for_sed_eval_tool_submission, inverse_scale, write_submission)



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
            fig, axs = plt.subplots(4, 1, figsize=(10, 8))
            logmel = inverse_scale(output_dict['feature'][n], 
                self.data_generator.scalar['mean'], 
                self.data_generator.scalar['std'])
            axs[0].matshow(logmel.T, origin='lower', aspect='auto', cmap='jet')
            if 'strong_target' in output_dict.keys():
                axs[1].matshow(output_dict['strong_target'][n].T, origin='lower', aspect='auto', cmap='jet')
            masked_framewise_output = output_dict['framewise_output'][n] * output_dict['clipwise_output'][n]
            axs[2].matshow(masked_framewise_output.T, origin='lower', aspect='auto', cmap='jet')
            axs[3].matshow(event_prediction.T, origin='lower', aspect='auto', cmap='jet')
            
            axs[0].set_title('Log mel spectrogram', color='r')
            axs[1].set_title('Reference sound events', color='r')
            axs[2].set_title('Framewise prediction', color='b')
            axs[3].set_title('Eventwise prediction', color='b')
            
            for i in range(4):
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
            plt.savefig(str(n) + "fig.jpg") #For saving the figures
            plt.show()
            
            
if __name__ == '__main__':
    visualize()
    