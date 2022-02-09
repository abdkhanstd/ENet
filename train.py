"""
Train all the basline networks
"""


import os


###### Collect all garbage #######
import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras
sys.stderr = stderr
###################################

'''
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.15
config.allow_soft_placement=True
config.log_device_placement=False
config.gpu_options.allocator_type = 'BFC'
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
'''
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from models.models import ResearchModels
from utils.data import DataSet
import time
import os.path

def train(lists,data_type, seq_length, model, saved_model=None,target=None,
          class_limit=None, image_shape=None,
          load_to_memory=False, batch_size=32, nb_epoch=100,features=None,features_length=None):
    # Helper: Save the model.

    if data_type=='images':
        checkpoints_dir='data/Weights_'+model
        checkpointer = ModelCheckpoint(
            filepath=os.path.join(checkpoints_dir, model + '-' + data_type + '_'+target + '_'+lists+'_'+\
                '.{epoch:03d}-{val_loss:.3f}.hdf5'),
            verbose=1,
            save_best_only=True)
        # Helper: TensorBoard
        tb = TensorBoard(log_dir=os.path.join('data', model+'_logs', model))
        # Helper: Save results.
        timestamp = time.time()
        csv_logger = CSVLogger(os.path.join('data', model+'_logs', model +'_'+target + '-' + 'training-' + '_'+lists+'_'+\
            str(timestamp) + '.log'))        
    else:
    
        checkpoints_dir='data/Weights_'+features+'_'+model    
        checkpointer = ModelCheckpoint(
            filepath=os.path.join(checkpoints_dir, features+ '_'+model + '-' + data_type + '_'+ target +'_'+lists+'_'+\
                '.{epoch:03d}-{val_loss:.3f}.hdf5'),
            verbose=1,
            save_best_only=True)
        # Helper: TensorBoard
            
        
        tb = TensorBoard(log_dir=os.path.join('data', features+ '_'+model+'_'+'_'+lists+'_'+target +'_logs', model)) 
        # Helper: Save results.
        timestamp = time.time()
        csv_logger = CSVLogger(os.path.join('data', features+ '_'+model+'_'+'_'+lists+'_'+target +'_logs', model + '-' + 'training-' + \
            str(timestamp) + '.log'))        
         
     
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)   



    # Helper: Stop when we stop learning.
    early_stopper = EarlyStopping(patience=5)



    # Get the data and process it.
    if image_shape is None:
        data = DataSet(
            seq_length=seq_length, lists=lists,
            class_limit=class_limit,target=target,
            features=features
        )
    else:
        data = DataSet(
            seq_length=seq_length, lists=lists,
            class_limit=class_limit,target=target,
            image_shape=image_shape
        )

    # Get samples per epoch.
    # Multiply by 0.7 to attempt to guess how much of data.data is the train set.
    steps_per_epoch = (len(data.data) * 0.7) // batch_size

    if load_to_memory:
        # Get data.
        X, y = data.get_all_sequences_in_memory('Train', data_type)
        X_test, y_test = data.get_all_sequences_in_memory('test', data_type)
    else:
        # Get generators.
        generator = data.frame_generator(batch_size, 'Train', data_type)
        val_generator = data.frame_generator(batch_size, 'test', data_type)

    # Get the model.
    rm = ResearchModels(len(data.classes), model, seq_length, saved_model,image_shape=image_shape,batch_size=batch_size,features_length=features_length)

    # Fit!
    if load_to_memory:
        # Use standard fit.
        rm.model.fit(
            X,
            y,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=1,
            callbacks=[tb, early_stopper, csv_logger,checkpointer],

            epochs=nb_epoch)
    else:
        # Use fit generator.
        rm.model.fit_generator(
            generator=generator,
            steps_per_epoch=steps_per_epoch,
            epochs=nb_epoch,
            verbose=1,
            callbacks=[tb, early_stopper, csv_logger,checkpointer],
            validation_data=val_generator,
            validation_steps=40,
            workers=1)

def main():
    """These are the main training settings. Set each before running
    this file."""
    # model can be one of lstm, lrcn, mlp, conv_3d, c3d,r3d_18,r3d_34,r3d_50,r3d_101,r3d_152,densenetresnet_3d,densenet_3d
    #os.environ['CUDA_VISIBLE_DEVICES'] ='0' # 6,2
    target='action'  # views action situation

    os.environ['CUDA_VISIBLE_DEVICES'] ='1' # 6,2
    lists='s1_1.csv'     # s1_1.scv, s1_1.scv,s1_1.scv,s1_1.scv, Splits.csv
    model = 'r3d_101'
    features='IR2' #Options (ALEXNET,resnet50, IV3,vgg16,vgg19,IR2,resnet152,densenet169,xception)

    saved_model = None  # None or weights file
    class_limit = None  # 
    seq_length = 16
    batch_size =2
    nb_epoch = 1000
    
    



    if features in ['IV3','resnet50','resnet152','densenet169','xception']:
        features_length=2048
    if features in['vgg16','vgg19']:
        features_length=25088  
    if features=='IR2':
        features_length=1536  
    if features=='efficientnetb7':
        features_length=2560  
    if features=='densenet169':
        features_length=1664         
    if features=='mobilenet':
        features_length=1024     
    if features=='ALEXNET':
        features_length=256    
    if features=='googlenet':
        features_length=1024          # or 1024


    # Chose images or features and image shape based on network.nvidia-smi
    if model in ['conv_3d']:
        data_type = 'images'
        image_shape = (100, 100, 3)
    elif model in ['lstm', 'mlp','capsnet','gru']:
        data_type = 'features'
        image_shape = None
    elif model in ['r3d_152_BERT','i3d','r3d_18','r3d_34','r3d_50','r3d_101','r3d_152']:
        data_type = 'images'
        image_shape = (224, 224, 3)
    elif model in ['c3d', 'lrcn']:
        data_type = 'images'
        image_shape = (150, 150, 3)
    elif model in ['densenetresnet_3d','densenet_3d']:
        data_type = 'images'
        image_shape = (112, 112, 3)        
    else:
        raise ValueError("Invalid model. See train.py for options.")
        
    if data_type=='images':
        load_to_memory = False
    else:
        load_to_memory = False        

    print('Model = ',model,' Features = ',features)
    train(lists,data_type, seq_length, model, saved_model=saved_model,target=target,
          class_limit=class_limit, image_shape=image_shape,
          load_to_memory=load_to_memory, batch_size=batch_size, nb_epoch=nb_epoch,features=features,features_length=features_length)

if __name__ == '__main__':
    main()
