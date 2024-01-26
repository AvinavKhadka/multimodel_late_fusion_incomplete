'''
sample_rate = 32000
window_size = 1024
hop_size = 500      # So that there are 64 frames per second
mel_bins = 64
fmin = 50       # Hz
fmax = 14000    # Hz

frames_per_second = sample_rate // hop_size
audio_duration = 10     # Audio recordings in DCASE2019 Task4 are all 
                        # approximately 10 seconds
frames_num = frames_per_second * audio_duration
total_samples = sample_rate * audio_duration

# Labels
#labels = ['Speech', 'Dog', 'Cat', 'Alarm_bell_ringing', 'Dishes', 'Frying', 'Blender', 'Running_water', 'Vacuum_cleaner', 'Electric_shaver_toothbrush']
labels = ['Ahem', 'Bull', 'Calf', 'Cow', 'Mix']

classes_num = len(labels)
lb_to_idx = {lb: idx for idx, lb in enumerate(labels)}
idx_to_lb = {idx: lb for idx, lb in enumerate(labels)}
'''
sample_rate = 8000
window_size = 512
hop_size = 250 #for 32 frames
#hop_size = 200      # So that there are 40 frames per second 

mel_bins = 32
fmin = 5       # Hz
fmax = 4660    # Hz

#for video
video_fps = 64
video_feature_dim = 128

frames_per_second = sample_rate // hop_size #64 frames per second
audio_duration = 30   # Audio recordings in DCASE2019 Task4 are all in sec     
frames_num = frames_per_second * audio_duration #Total temporal frames = 64*10 =640
total_samples = sample_rate * audio_duration

# Labels
labels = [ "Wild_Boar", "Elephant", "Ferret", "Rhino", "Wolf",
    "Coyote", "Bear", "Monkey", "Lion", "Tiger", "Squirrel", "Leopard", "Mongoose", "Peacock", "Cheetah", "Deer","Fox","Frog",
    "Gecko","Zebra"]

classes_num = len(labels)
lb_to_idx = {lb: idx for idx, lb in enumerate(labels)}
idx_to_lb = {idx: lb for idx, lb in enumerate(labels)}