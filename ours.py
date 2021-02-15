import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.95
config.allow_soft_placement=True
config.gpu_options.allow_growth=True
config.log_device_placement=False
set_session(tf.Session(config=config))

from keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger
from models.models import ResearchModels
from utils.data import DataSet
import functools
import keras.metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import numpy as np



seq_length=60
batch_size=2

######################################################################
# Adding resnet 3D model
model_1='r3d_50'
model_file_name_1='r3d_50-images.020-4.377.hdf5'
saved_model = 'data/'+'Weights_'+model_1+'/'+model_file_name_1
image_shape_1 = (224, 224, 3)
data_type_1 = 'images'

data = DataSet(
            seq_length=seq_length,
            class_limit=None,
            image_shape=image_shape_1)
    
val_generator_1 = data.frame_generator(batch_size, 'test', data_type_1)
rm = ResearchModels(len(data.classes), model_1, seq_length, saved_model)

for X,y in data.gen_test('test', data_type_1): 
    Probs_1 = rm.model.predict(X)

######################################################################
model_2='gru'
features='IR2'
model_file_name_2='IR2_gru-features.086-0.431.hdf5'
saved_model = 'data/'+'Weights_'+features+'_'+model_2+'/'+model_file_name_2
image_shape_2 = None
data_type_2 = 'features'

data = DataSet(
            seq_length=seq_length,
            class_limit=None,
            features=features,
            image_shape=image_shape_2)
    
val_generator_2 = data.frame_generator(batch_size, 'test', data_type_2)
rm = ResearchModels(len(data.classes), model_2, seq_length, saved_model)

for X,y in data.gen_test('test', data_type_2): 
    Probs_2 = rm.model.predict(X)


######################################################################
#print('*******',Probs_2,'********')
#print('*******',y,'********')

combined_prob=Probs_1+Probs_2
predicted=np.argmax(combined_prob, axis=-1)
idx=np.where(np.array(y)==1)
true_label=idx[1]
print(classification_report(true_label, predicted))

tn, fp, fn, tp = confusion_matrix(true_label, predicted).ravel()
print('Confusion Matrix: TP=',tp,'-- TN=',tn,'-- FP=',fp,'-- FN=',fn)




   
    