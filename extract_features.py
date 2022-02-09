
"""
This script generates extracted features for each patient, which other
models make use of. 
"""

"""
Train all the basline networks
"""


import os
os.environ['CUDA_VISIBLE_DEVICES'] ='3'
###### Collect all garbage #######

import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras
sys.stderr = stderr

###################################

import tensorflow as tf
tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.2)

"""
from keras.backend.tensorflow_backend import set_session
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
sess = tf.Session(config=tf.ConfigProto(
  allow_soft_placement=True, log_device_placement=True))
"""
import numpy as np
import os.path
from utils.data import DataSet
from tqdm import tqdm
from models.models import ResearchModels


# IV3, IR2, vgg19,densenet169,resnet152,resnet50
# Set defaults.
lists='Splits.csv'     # s1_1.scv, s1_1.scv,s1_1.scv,s1_1.scv, Splits.csv
seq_length = 16                  #                     H                   
extractor_arch='IV3' #Options (ALEXNET,mobilenet,resnet50, IV3,vgg16,vgg19,IR2,resnet152,densenet169,xception,efficientnetb7)
target='action'


class_limit = None
ndir=os.path.join('data','sequences_'+extractor_arch)
if not os.path.exists(ndir):
    os.makedirs(ndir)
# Get the dataset.
data = DataSet(seq_length=seq_length, class_limit=class_limit,target=target)
# get the model.
if extractor_arch=='googlenet':
    from  extractors.extractor_googlenet import Extractor
    model = Extractor()
if extractor_arch=='IV3':
    from  extractors.extractor_iv3 import Extractor
    model = Extractor()
if extractor_arch=='resnet50':
    from  extractors.extractor_resnet50 import Extractor
    model = Extractor()    
if extractor_arch=='vgg16':
    from  extractors.extractor_vgg16 import Extractor
    model = Extractor()     
if extractor_arch=='IR2':
    from  extractors.extractor_IR2 import Extractor
    model = Extractor()    
if extractor_arch=='resnet152':
    from  extractors.extractor_resnet152 import Extractor
    model = Extractor()       
if extractor_arch=='densenet169':
    from  extractors.extractor_densenet169 import Extractor
    model = Extractor()  
if extractor_arch=='vgg19':
    from  extractors.extractor_vgg19 import Extractor
    model = Extractor()    
if extractor_arch=='xception':
    from  extractors.extractor_xception import Extractor
    model = Extractor()   
if extractor_arch=='efficientnetb7':
    from  extractors.extractor_efficientnetb7 import Extractor
    model = Extractor()    
if extractor_arch=='mobilenet':
    from  extractors.extractor_mobilenet import Extractor
    model = Extractor()
if extractor_arch=='ALEXNET':
    from  extractors.extractor_ALEXNET import Extractor
    model = Extractor()     
    

# Loop through data.
pbar = tqdm(total=len(data.data))
for video in data.data:

    #trim the file name
    tmp=video[2].split('/')
    tmp=tmp[1].split('.')
    fname=tmp[0]
    
    
    # Get the path to the sequence for this video.
    path = os.path.join('data', 'sequences_'+extractor_arch, fname + '-' + str(seq_length) + \
        '-features_'+target)  # numpy will auto-append .npy

    # Check if we already have it.
    if os.path.isfile(path + '.npy'):
        pbar.update(1)
        continue

    # Get the frames for this video.
    frames = data.get_frames_for_sample(video)
    
    # Now downsample to just the ones we need.
    frames = data.rescale_list(frames, seq_length)

    # Now loop through and extract features to build the sequence.
    sequence = []
    for image in frames:
        features = model.extract(image)
        features=np.squeeze(features)         #remove this line for python 2.7        
        sequence.append(features)

    # Save the sequence.
    np.save(path, sequence)

    pbar.update(1)

pbar.close()
