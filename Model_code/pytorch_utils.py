import numpy as np
import torch


def move_data_to_gpu(x, cuda):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        raise Exception("Error!")

    if cuda:
        x = x.cuda()

    return x
    
    
def interpolate(x, ratio):
    '''Interpolate the prediction to have the same time_steps as the target. 
    The time_steps mismatch is caused by maxpooling in CNN. 
    
    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to upsample
    '''
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled
    
    
def append_to_dict(dict, key, value):
    if key in dict.keys():
        dict[key].append(value)
        
    else:
        dict[key] = [value]
    
    
def forward(model, generate_func, cuda, return_input=False, 
    return_target=False):
    '''Forward data to model in mini-batch. 
    
    Args: 
      model: object
      generate_func: function
      cuda: bool
      return_input: bool
      return_target: bool
      max_validate_num: None | int, maximum mini-batch to forward to speed up validation
    '''
    output_dict = {}
    # print('generate_func', generate_func)
    # Evaluate on mini-batch
    for (n, batch_data_dict) in enumerate(generate_func):
        

        # Predict

        feature_tensor = torch.from_numpy(batch_data_dict['feature'])
        video_feature_tensor = torch.from_numpy(batch_data_dict['video_feature'])

        # Move tensors to GPU if CUDA is available
        cuda = torch.cuda.is_available()
        if cuda:
            feature_tensor = feature_tensor.cuda()
            video_feature_tensor = video_feature_tensor.cuda()


        # Assuming audio_batch_feature and video_batch_feature have the same batch size
        batch_feature = {'feature': feature_tensor, 'video_feature': video_feature_tensor}
            # batch_feature = {'feature': feature_tensor}

        
        with torch.no_grad():
            model.eval()
            batch_output_dict = model(batch_feature)

        ##################################3rd information##############################################
        # print("batch_data_dict", batch_data_dict.keys())
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # print('batch_output_dict', batch_output_dict.keys())
        append_to_dict(output_dict, 'audio_name', batch_data_dict['audio_name'])

        
        if 'clipwise_output' in batch_output_dict.keys():
            append_to_dict(output_dict, 'clipwise_output', 
                batch_output_dict['clipwise_output'].data.cpu().numpy())
            
        if 'framewise_output' in batch_output_dict.keys():
            append_to_dict(output_dict, 'framewise_output', 
                batch_output_dict['framewise_output'].data.cpu().numpy())
        
        if return_input:
            append_to_dict(output_dict, 'feature', batch_data_dict['feature'])
            
        if return_target:
            if 'weak_target' in batch_data_dict.keys():
                append_to_dict(output_dict, 'weak_target', batch_data_dict['weak_target'])
                
            if 'strong_target' in batch_data_dict.keys():
                append_to_dict(output_dict, 'strong_target', batch_data_dict['strong_target'])

    for key in output_dict.keys():
        output_dict[key] = np.concatenate(output_dict[key], axis=0)
    # print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
    # print('output_dict', output_dict['strong_target'])
    # print('output_dict', output_dict)
    return output_dict