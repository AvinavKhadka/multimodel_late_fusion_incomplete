import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_utils import interpolate


def init_layer(layer, nonlinearity='leaky_relu'):
    """Initialize a Linear or Convolutional layer. """
    nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
    
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    
    bn.bias.data.fill_(0.)
    bn.running_mean.data.fill_(0.)
    bn.weight.data.fill_(1.)
    bn.running_var.data.fill_(1.)
    
    
class Cnn_5layers_AvgPooling(nn.Module):
    
    def __init__(self, classes_num, strong_target_training):
        super(Cnn_5layers_AvgPooling, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64,
                              kernel_size=(5, 5), stride=(1, 1),
                              padding=(2, 2), bias=False)
        
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128,
                              kernel_size=(5, 5), stride=(1, 1),
                              padding=(2, 2), bias=False)
                              
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256,
                              kernel_size=(5, 5), stride=(1, 1),
                              padding=(2, 2), bias=False)
                              
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512,
                              kernel_size=(5, 5), stride=(1, 1),
                              padding=(2, 2), bias=False)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)

        self.fc = nn.Linear(512, classes_num, bias=True)

        self.init_weights()
        
        self.strong_target_training = strong_target_training

    def init_weights(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_layer(self.conv3)
        init_layer(self.conv4)
        init_layer(self.fc)
        
        init_bn(self.bn1)
        init_bn(self.bn2)
        init_bn(self.bn3)
        init_bn(self.bn4)

    def forward(self, input):
        '''
        Input: (batch_size, times_steps, freq_bins)'''
        
        interpolate_ratio = 8
        
        x = input[:, None, :, :]
        '''(batch_size, 1, times_steps, freq_bins)'''
        
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.avg_pool2d(x, kernel_size=(2, 2))
        
        x = F.relu_(self.bn2(self.conv2(x)))
        x = F.avg_pool2d(x, kernel_size=(2, 2))
        
        x = F.relu_(self.bn3(self.conv3(x)))
        x = F.avg_pool2d(x, kernel_size=(2, 2))
        
        x = F.relu_(self.bn4(self.conv4(x)))
        tf_maps = F.avg_pool2d(x, kernel_size=(1, 1))
        '''Time-frequency maps: (batch_size, channels_num, times_steps, freq_bins)'''

        framewise_vector = torch.mean(tf_maps, dim=3)
        '''(batch_size, feature_maps, frames_num)'''
        
        output_dict = {}
        
        # Framewise prediction
        framewise_output = torch.sigmoid(self.fc(framewise_vector.transpose(1, 2)))
        framewise_output = interpolate(framewise_output, interpolate_ratio)
        '''(batch_size, frames_num, classes_num)'''
            
        output_dict['framewise_output'] = framewise_output

        # Clipwise prediction
        if self.strong_target_training:
            # Obtained by taking the maximum framewise predictions
            (output_dict['clipwise_output'], _) = torch.max(framewise_output, dim=1)
            
        else:
            # Obtained by applying fc layer on aggregated framewise_vector
            (aggregation, _) = torch.max(framewise_vector, dim=2)
            output_dict['clipwise_output'] = torch.sigmoid(self.fc(aggregation))

        return output_dict
    

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.init_weights()
        
    def init_weights(self):
        
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)
        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        else:
            raise Exception('Incorrect argument!')
        
        return x
    
    
class Cnn_9layers_AvgPooling(nn.Module):
    def __init__(self, classes_num, strong_target_training=False):
        
        super(Cnn_9layers_AvgPooling, self).__init__()

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.fc = nn.Linear(512, classes_num, bias=True)

        self.init_weights()
        
        self.strong_target_training = strong_target_training

    def init_weights(self):

        init_layer(self.fc)

    def forward(self, input):
        '''
        Input: (batch_size, times_steps, freq_bins)'''
        
        interpolate_ratio = 8
        
        x = input[:, None, :, :]
        '''(batch_size, 1, times_steps, freq_bins)'''
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        tf_maps = self.conv_block4(x, pool_size=(1, 1), pool_type='avg')
        '''Time-frequency maps: (batch_size, channels_num, times_steps, freq_bins)'''

        (framewise_vector, _) = torch.max(tf_maps, dim=3)
        '''(batch_size, feature_maps, frames_num)'''
        
        output_dict = {}

        # Framewise prediction
        framewise_output = torch.sigmoid(self.fc(framewise_vector.transpose(1, 2)))
        framewise_output = interpolate(framewise_output, interpolate_ratio)
        '''(batch_size, frames_num, classes_num)'''
            
        output_dict['framewise_output'] = framewise_output

        # Clipwise prediction
        if self.strong_target_training:
            # Obtained by taking the maximum framewise predictions
            (output_dict['clipwise_output'], _) = torch.max(framewise_output, dim=1)
            
        else:
            # Obtained by applying fc layer on aggregated framewise_vector
            (aggregation, _) = torch.max(framewise_vector, dim=2)
            output_dict['clipwise_output'] = torch.sigmoid(self.fc(aggregation))

        return output_dict
        
        
class Cnn_9layers_MaxPooling(nn.Module):
    def __init__(self, classes_num, strong_target_training=False):
        
        super(Cnn_9layers_MaxPooling, self).__init__()

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.fc = nn.Linear(512, classes_num, bias=True)

        self.init_weights()
        
        self.strong_target_training = strong_target_training

    def init_weights(self):

        init_layer(self.fc)

    def forward(self, input):
        '''
        Input: (batch_size, times_steps, freq_bins)'''
        
        interpolate_ratio = 8
        
        x = input[:, None, :, :]
        '''(batch_size, 1, times_steps, freq_bins)'''
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='max')
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='max')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='max')
        tf_maps = self.conv_block4(x, pool_size=(1, 1), pool_type='max')
        '''Time-frequency maps: (batch_size, channels_num, times_steps, freq_bins)'''

        (framewise_vector, _) = torch.max(tf_maps, dim=3)
        '''(batch_size, feature_maps, frames_num)'''
        
        output_dict = {}
        
        # Framewise prediction
        framewise_output = torch.sigmoid(self.fc(framewise_vector.transpose(1, 2)))
        framewise_output = interpolate(framewise_output, interpolate_ratio)
        '''(batch_size, frames_num, classes_num)'''
            
        output_dict['framewise_output'] = framewise_output

        # Clipwise prediction
        if self.strong_target_training:
            # Obtained by taking the maximum framewise predictions
            (output_dict['clipwise_output'], _) = torch.max(framewise_output, dim=1)
            
        else:
            # Obtained by applying fc layer on aggregated framewise_vector
            (aggregation, _) = torch.max(framewise_vector, dim=2)
            output_dict['clipwise_output'] = torch.sigmoid(self.fc(aggregation))

        return output_dict
        
        
class Cnn_13layers_AvgPooling(nn.Module):
    def __init__(self,classes_num, strong_target_training=True ):
        
        super(Cnn_13layers_AvgPooling, self).__init__()

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc = nn.Linear(2048, classes_num, bias=True)

        self.init_weights()
        
        self.strong_target_training = strong_target_training

    def init_weights(self):

        init_layer(self.fc)

    def forward(self, input):
        '''
        Input: (batch_size, times_steps, freq_bins)'''
        
        interpolate_ratio = 32
        
        x = input[:, None, :, :]
        '''(batch_size, 1, times_steps, freq_bins)'''
        # print("x.shape",x.shape)
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        # print("x.shape_conv_1",x.shape)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        # print("x.shape_conv_2",x.shape)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        # print("x.shape_conv_3",x.shape)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        # print("x.shape_conv_4",x.shape)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        # print("x.shape_conv_5",x.shape)
        tf_maps = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        # print("x.shape_conv_6",tf_maps.shape)
        # output_dict = {}
        # output_dict['tf_maps'] = tf_maps
        output_dict = tf_maps
        '''Time-frequency maps: (batch_size, channels_num, times_steps, freq_bins)'''

        # (framewise_vector, _) = torch.max(tf_maps, dim=3)
        # '''(batch_size, feature_maps, frames_num)'''
        
        # output_dict = {}
        
        # # Framewise prediction
        # framewise_output = torch.sigmoid(self.fc(framewise_vector.transpose(1, 2)))
        # framewise_output = interpolate(framewise_output, interpolate_ratio)
        # '''(batch_size, frames_num, classes_num)'''
            
        # output_dict['framewise_output'] = framewise_output
        
        # # Clipwise prediction
        # if self.strong_target_training:
        #     # Obtained by taking the maximum framewise predictions
        #     (output_dict['clipwise_output'], _) = torch.max(framewise_output, dim=1)
            
        # else:
        #     # Obtained by applying fc layer on aggregated framewise_vector
        #     (aggregation, _) = torch.max(framewise_vector, dim=2)
        #     output_dict['clipwise_output'] = torch.sigmoid(self.fc(aggregation))

        # return tf_maps
        return output_dict
    
# 3d cnn
class C3D_Max(nn.Module):
    def __init__(self, sample_size, sample_duration, num_classes=600, in_channels=1):

        super(C3D_Max, self).__init__()
        self.group1 = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 2)))
        self.group2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        self.group3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        self.group4 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        self.group5 = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1)))

        last_duration = int(math.floor(sample_duration / 16))
        last_size = int(math.ceil(sample_size / 32))
        self.fc1 = nn.Sequential(
            nn.Linear((512 * last_duration * last_size * last_size) , 4096),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.fc2 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.fc = nn.Sequential(
            nn.Linear(4096, num_classes))         

    def forward(self, x):
        interpolate_ratio = 32
        out = self.group1(x)
        out = self.group2(out)
        out = self.group3(out)
        out = self.group4(out)
        out = self.group5(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc(out)
        tf_maps = self.conv_block4(x, pool_size=(1, 1), pool_type='max')

        output_dict = {}
        output_dict['tf_maps'] = tf_maps
        '''Time-frequency maps: (batch_size, channels_num, times_steps, freq_bins)'''

        # (framewise_vector, _) = torch.max(tf_maps, dim=3)
        # '''(batch_size, feature_maps, frames_num)'''
        
        # output_dict = {}
        
        # # Framewise prediction
        # framewise_output = torch.sigmoid(self.fc(framewise_vector.transpose(1, 2)))
        # framewise_output = interpolate(framewise_output, interpolate_ratio)
        # '''(batch_size, frames_num, classes_num)'''
            
        # output_dict['framewise_output'] = framewise_output

        # # Clipwise prediction
        # if self.strong_target_training:
        #     # Obtained by taking the maximum framewise predictions
        #     (output_dict['clipwise_output'], _) = torch.max(framewise_output, dim=1)
            
        # else:
        #     # Obtained by applying fc layer on aggregated framewise_vector
        #     (aggregation, _) = torch.max(framewise_vector, dim=2)
        #     output_dict['clipwise_output'] = torch.sigmoid(self.fc(aggregation))

        return output_dict

class C3D2(nn.Module):
    def __init__(self,classes_num):
        super(C3D2, self).__init__()
        self.conv1 = nn.Conv3d(128, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4 = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5 = nn.Conv3d(512, 1024, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv6 = nn.Conv3d(1024, 2048, kernel_size=(3, 3, 3), padding=(1, 1, 1))

        # fully connected layers
        self.fc = nn.Linear(8192, classes_num)

        # Initialization
        self.init_weights()

        #self.strong_target_training = strong_target_training

    def init_weights(self):
        nn.init.kaiming_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, input):

        x = input[:, :, :]
        # print("x.shape",x.shape)
        x = self.conv1(x)
        # print("x.shape_conv1",x.shape)
        x = nn.functional.max_pool3d(x, kernel_size=(1, 2, 2), stride=(2, 2, 2),padding=(0, 1, 1))
        # print("x.shape_conv1_max",x.shape)
        x = self.conv2(x)
        # print("x.shape_conv2",x.shape)
        x = nn.functional.max_pool3d(x, kernel_size=(1, 2, 2), stride=(2, 2, 2),padding=(0, 1, 1))
        # print("x.shape_conv2_max",x.shape)
        x = self.conv3(x)
        # print("x.shape_conv3",x.shape)
        x = nn.functional.max_pool3d(x, kernel_size=(1, 2, 2), stride=(2, 2, 2))
        # print("x.shape_conv3_max",x.shape)
        x = self.conv4(x)
        # print("x.shape_conv4",x.shape)
        x = nn.functional.max_pool3d(x, kernel_size=(1, 2, 2), stride=(2, 2, 2),padding=(0, 1, 1))
        x = self.conv5(x)
        x = nn.functional.max_pool3d(x, kernel_size=(1, 2, 2), stride=(2, 2, 2),padding=(0, 1, 1))
        x = self.conv6(x)
        x = nn.functional.max_pool3d(x, kernel_size=(1, 2, 2), stride=(2, 2, 2),padding=(0, 1, 1))

     
        # Global Average Pooling
        x = nn.functional.avg_pool3d(x, kernel_size=(1, x.size(3), x.size(4)))
   

        # flatten
        x = x.view(x.size(0),x.size(1),x.size(2), -1)


        tf_maps = x
        # output_dict = {}
        # output_dict['tf_maps'] = tf_maps
        output_dict = tf_maps

        return output_dict
    
class LateFusion(nn.Module):
    def __init__(self, classes_num, strong_target_training=True):
        super(LateFusion, self).__init__()
        
        self.fused_layer = None
        
        self.model_type_a = Cnn_13layers_AvgPooling(classes_num=classes_num, strong_target_training=strong_target_training) 
        self.model_type_v = C3D2(classes_num=classes_num) # 
        
        # Define the fully connected layer for late fusion
        self.fc = nn.Linear(2048, classes_num, bias=True)

        self.init_weights()

        self.strong_target_training = strong_target_training
    
    def init_weights(self):
        init_layer(self.fc)
        
    def forward(self, input_data):
        
        interpolate_ratio = 32
        # Forward pass through the first model
        tf_maps_a = self.model_type_a(input_data['feature'])
        # Forward pass through the second model
        tf_maps_v = self.model_type_v(input_data['video_feature'])

        
        # Concatenate the two outputs
        c_tf_maps = torch.cat((tf_maps_v, tf_maps_a), dim=2)
        (framewise_vector, _) = torch.max(c_tf_maps, dim=2)
        '''(batch_size, feature_maps, frames_num)'''
        
        output_dict = {}
        
        # Framewise prediction
        framewise_output = torch.sigmoid(self.fc(framewise_vector.transpose(1, 2)))
        framewise_output = interpolate(framewise_output, interpolate_ratio)
        '''(batch_size, frames_num, classes_num)'''
            
        output_dict['framewise_output'] = framewise_output

        # Clipwise prediction
        if self.strong_target_training:
            # Obtained by taking the maximum framewise predictions
            (output_dict['clipwise_output'], _) = torch.max(framewise_output, dim=1)
            
        else:
            # Obtained by applying fc layer on aggregated framewise_vector
            (aggregation, _) = torch.max(framewise_vector, dim=2)
            output_dict['clipwise_output'] = torch.sigmoid(self.fc(aggregation))
        return output_dict


