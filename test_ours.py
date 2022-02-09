import os



###### Collect all garbage #######
import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras
sys.stderr = stderr
###################################

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger
from models.models import ResearchModels
from utils.data import DataSet
import functools
import keras.metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sns
from matplotlib import pyplot as plt

import numpy as np
from sklearn.metrics import accuracy_score




def validate(lists,data_type, model, seq_length=16, saved_model=None,
             class_limit=None, image_shape=None,batch_size = 16,features=None,target=None):
    
    # Get the data and process it.
    if image_shape is None:
        data = DataSet(
            seq_length=seq_length,lists=lists,
            class_limit=class_limit,target=target,
            features=features
        )
    else:
        data = DataSet(
            seq_length=seq_length,lists=lists,
            class_limit=class_limit,target=target,
            image_shape=image_shape
        )

    val_generator = data.frame_generator(batch_size, 'test', data_type)

    #Get the model.
    rm = ResearchModels(len(data.classes), model, seq_length, saved_model)
    num_classes=len(data.get_classes())
    scores=np.zeros([num_classes])
    total=np.zeros([num_classes])
    for X,y,z in data.gen_test('test', data_type): 
        results = rm.model.predict(X)
        #print('*******',results,'********')
        predicted=np.argmax(results, axis=-1)
        #f.write()

        idx=np.where(np.array(y)==1)
        true_label=idx[1]

      
    return true_label,predicted    

    
    
  
   

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] ='3'
  
# Branch 1 IR2
    print('Tesing Branch 1')
    model = 'lstm' 
    target='action'
    lists='s2_1.csv'     # s1_1.scv, s1_1.scv,s1_1.scv,s1_1.scv, Splits.csv
    features='IR2' # Choose model name (IR2)
    seq_length=16
    batch_size=2
    saved_model='data/Weights_IR2_lstm/IR2_lstm-features_action_s2_1.csv_.014-0.778.hdf5'
    
    image_shape = None
    data_type = 'features' 
    t2,p2=validate(lists,data_type, model, saved_model=saved_model,seq_length=seq_length,
             image_shape=image_shape,batch_size=batch_size,features=features,target=target)

# Branch 2 IR2
    print('Tesing Branch 2')
    model = 'lstm' 
    target='action'
    lists='s4_1.csv'     # s1_1.scv, s1_1.scv,s1_1.scv,s1_1.scv, Splits.csv
    features='IR2' # Choose model name (IR2)
    seq_length=16
    batch_size=2
    saved_model='data/Weights_IR2_lstm/IR2_lstm-features_action_s4_1.csv_.010-0.882.hdf5'
    
    image_shape = None
    data_type = 'features' 
    t4,p4=validate(lists,data_type, model, saved_model=saved_model,seq_length=seq_length,
             image_shape=image_shape,batch_size=batch_size,features=features,target=target)        
  
# Branch 3 ResNet3D 101  
    print('Tesing Branch 3')
    model = 'r3d_101' 
    target='action'
    lists='s1_1.csv'     # s1_1.scv, s1_1.scv,s1_1.scv,s1_1.scv, Splits.csv
    features='none'
    seq_length=16
    batch_size=2


    saved_model='data/Weights_r3d_50/r3d_50-images_action_s1_1.csv_.015-5.872.hdf5' #Best
    #saved_model='data/Weights_r3d_101/r3d_101-images_action_s1_1.csv_.018-10.237.hdf5'
    
    image_shape = (224, 224, 3)
    data_type = 'images'          
    t1,p1=validate(lists,data_type, model, saved_model=saved_model,seq_length=seq_length,
             image_shape=image_shape,batch_size=batch_size,features=features,target=target)
    
# Branch 4 ResNet3D 101  
    print('Tesing Branch 4')
    model = 'r3d_101' 
    target='action'
    lists='s3_1.csv'     # s1_1.scv, s1_1.scv,s1_1.scv,s1_1.scv, Splits.csv
    features='none' # Choose model name (IR2)
    seq_length=16
    batch_size=2
    saved_model='data/Weights_r3d_101/r3d_101-images_action_s3_1.csv_.010-10.445.hdf5' # Best
    #saved_model='data/Weights_r3d_101/r3d_101-images_action_s3_1.csv_.015-9.070.hdf5'
    Simage_shape = (224, 224, 3)
    data_type = 'images'          
    t3,p3=validate(lists,data_type, model, saved_model=saved_model,seq_length=seq_length,
             image_shape=image_shape,batch_size=batch_size,features=features,target=target)    
    
    true_label=[t1,t2,t3,t4]
    t1 = np.array(t1)
    t2 = np.array(t2)+30
    t3 = np.array(t3)+54
    t4 = np.array(t4)+70
    true_label=np.append(t1,t2)
    true_label=np.append(true_label,t3)
    true_label=np.append(true_label,t4)

    p1 = np.array(p1)
    p2 = np.array(p2)+30
    p3 = np.array(p3)+54
    p4 = np.array(p4)+70
    predicted=np.append(p1,p2)
    predicted=np.append(predicted,p3)
    predicted=np.append(predicted,p4)
                        

    print(classification_report(true_label, predicted))
    cm = confusion_matrix(true_label, predicted,labels=list(range(0,84)))
    #Accuracies of diagonal class
    # Normalise
    #cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm.diagonal())
     
    #Plot Figure 
    fig, ax = plt.subplots(figsize=(300,300))
    sns.heatmap(cm, annot=True, fmt='.0f',annot_kws={"size": 6}) #cmap='Blues'
    sns.set(font_scale=1.4) # for label size

    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    plt.savefig('result_1.png')    
    print("*************************************")
    print("S1",round(accuracy_score(t1, p1),2))
    print("S2",round(accuracy_score(t2, p2),2))
    print("S3",round(accuracy_score(t3, p3),2))
    print("S4",round(accuracy_score(t4, p4),2))
    print("Overall",round(accuracy_score(true_label, predicted),2))
if __name__ == '__main__':
    main()
