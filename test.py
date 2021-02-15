import os
os.environ['CUDA_VISIBLE_DEVICES'] ='2'



###### Collect all garbage #######
import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras
sys.stderr = stderr
###################################


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




def validate(data_type, model, seq_length=16, saved_model=None,
             class_limit=None, image_shape=None,batch_size = 16,features=None,target=None):
    

    # Get the data and process it.
    if image_shape is None:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit,target=target,
            features=features
        )
    else:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit,target=target,
            image_shape=image_shape
        )

    val_generator = data.frame_generator(batch_size, 'test', data_type)

    #Get the model.
    rm = ResearchModels(len(data.classes), model, seq_length, saved_model)
    scores=np.zeros([2])
    total=np.zeros([2])
    for X,y in data.gen_test('test', data_type): 
        results = rm.model.predict(X)
        print('*******',results,'********')
        predicted=np.argmax(results, axis=-1)
        print(predicted)
        idx=np.where(np.array(y)==1)
        true_label=idx[1]
        print(true_label)

        total[true_label]=total[true_label]+1
        
        if predicted[0]==true_label[0]:
            scores[true_label]=scores[true_label]+1
        

    print(classification_report(true_label, predicted))
    tn, fp, fn, tp = confusion_matrix(true_label, predicted).ravel()
    print('Confusion Matrix: TP=',tp,'-- TN=',tn,'-- FP=',fp,'-- FN=',fn)





def main():
    model = 'c3d' 
    target='views'
    features='IR2' # Choose model name (resnet50, IV3,vgg16,vgg19,IR2,resnet152,densenet169,xception,efficientnetb7)
    seq_length=60
    batch_size=2

    print('******Testing ',features,'_',model,'******')



    if model=='gru' and features=='efficientnetb7':
        model_file_name='efficientnetb7_gru-features.050-0.640.hdf5'
        image_shape = None
        data_type = 'features'

    if model=='lstm' and features=='efficientnetb7':
        model_file_name='efficientnetb7_lstm-features.009-0.533.hdf5'
        image_shape = None
        data_type = 'features'

    if model=='lstm' and features=='vgg19':
        model_file_name='vgg19_lstm-features.017-0.376.hdf5'
        image_shape = None
        data_type = 'features'

    if model=='gru' and features=='vgg19':
        model_file_name='vgg19_gru-features.027-0.531.hdf5'
        image_shape = None
        data_type = 'features'

    if model=='gru' and features=='xception':
        model_file_name='xception_gru-features.049-0.505.hdf5'
        image_shape = None
        data_type = 'features'

    if model=='lstm' and features=='xception':
        model_file_name='xception_lstm-features.018-0.471.hdf5'
        image_shape = None
        data_type = 'features'

    if model=='gru' and features=='densenet169':
        model_file_name='densenet169_gru-features.123-0.466.hdf5'
        image_shape = None
        data_type = 'features'

    if model=='lstm' and features=='densenet169':
        model_file_name='densenet169_lstm-features.030-0.490.hdf5'
        image_shape = None
        data_type = 'features'

    if model=='gru' and features=='resnet152':
        model_file_name='resnet152_gru-features.062-0.508.hdf5'
        image_shape = None
        data_type = 'features'

    if model=='lstm' and features=='resnet152':
        model_file_name='resnet152_lstm-features.028-0.432.hdf5'
        image_shape = None
        data_type = 'features'

    if model=='lstm' and features=='IR2':
        model_file_name='IR2_lstm-features.018-0.425.hdf5'
        image_shape = None
        data_type = 'features'

    if model=='gru' and features=='IR2':
        model_file_name='IR2_gru-features.086-0.431.hdf5'
        image_shape = None
        data_type = 'features'

    if model=='lstm' and features=='vgg16':
        model_file_name='vgg16_lstm-features.001-0.527.hdf5'
        image_shape = None
        data_type = 'features'


    if model=='gru' and features=='vgg16':
        model_file_name='vgg16_gru-features.017-0.357.hdf5'
        image_shape = None
        data_type = 'features'

    if model=='lrcn':
        model_file_name='lrcn-images.015-2.315.hdf5'
        image_shape = (150, 150, 3)
        data_type = 'images'   

    if model=='c3d':
        model_file_name='c3d-images.004-0.300.hdf5'
        image_shape = (150, 150, 3)
        data_type = 'images'      

    if model=='conv_3d':
        model_file_name='conv_3d-images.001-0.570.hdf5'
        image_shape = (100, 100, 3)
        data_type = 'images'    
        
    if model=='i3d':
        model_file_name='i3d-images.002-0.571.hdf5'
        image_shape = (224, 224, 3)
        data_type = 'images'   
   
    if model=='lstm' and features=='IV3':
        model_file_name='lstm-features.021-0.327.hdf5'
        image_shape = None
        data_type = 'features'  
    
    if model=='lstm' and features=='resnet50':
        model_file_name='resnet50_lstm-features.006-0.322.hdf5'
        image_shape = None
        data_type = 'features'          
        
    if model=='gru' and features=='resnet50':
        model_file_name='resnet50_gru-features.031-0.356.hdf5'
        image_shape = None
        data_type = 'features'         
    if model=='mlp' and features=='resnet50':
        model_file_name='resnet50_gru-features.031-0.356.hdf5'
        image_shape = None
        data_type = 'features'           
  
    if model=='mlp' and features=='IV3':
        model_file_name='resnet50_mlp-features.011-2.821.hdf5'
        image_shape = None
        data_type = 'features' 
   
    if model=='r3d_18':
        model_file_name='r3d_18-images.002-1.118.hdf5'
        image_shape = (224, 224, 3)
        data_type = 'images' 
        
    if model=='r3d_34':
        model_file_name='r3d_34-images.012-1.587.hdf5'
        image_shape = (224, 224, 3)
        data_type = 'images' 
        
    if model=='r3d_50':
        model_file_name='r3d_50-images.020-4.377.hdf5'
        image_shape = (224, 224, 3)
        data_type = 'images' 
        
    if model=='r3d_101':
        model_file_name='r3d_101-images.172-6.432.hdf5'
        image_shape = (224, 224, 3)
        data_type = 'images' 
        
    if model=='r3d_152':
        model_file_name='r3d_152-images.112-10.426.hdf5'
        image_shape = (224, 224, 3)
        data_type = 'images'         
    if model=='densenet_3d':
        model_file_name='densenet_3d-images.301-4.781.hdf5'
        image_shape = (112, 112, 3)
        data_type = 'images'  
    if model=='densenetresnet_3d':
        model_file_name='densenetresnet_3d-images.167-8.736.hdf5'
        image_shape = (112, 112, 3)
        data_type = 'images'          
        
    if data_type=='images':
        saved_model = 'data/'+'Weights_'+model+'/'+model_file_name
    else:
        saved_model = 'data/'+'Weights_'+features+'_'+model+'/'+model_file_name
        print('Testing ',model,' with ',features,' features');

    

    validate(data_type, model, saved_model=saved_model,seq_length=seq_length,
             image_shape=image_shape,batch_size=batch_size,features=features,target=target)

if __name__ == '__main__':
    main()
