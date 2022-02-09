"""

"""
###### Collect all garbage #######
import sys
import os
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras
sys.stderr = stderr
###################################

from collections import deque
from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization
import functools
import keras.metrics
from tensorflow.keras.models import Sequential, load_model

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras import activations
from tensorflow.keras import utils
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
import time
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

from tensorflow.keras.layers import concatenate
##Additional
from models.i3d_inception import Inception_Inflated3d, conv3d_bn
from models.resnet3d import Resnet3DBuilder

from models.densenet_3d import densenet_3d
from models.denseresnet3d import dense_resnet_3d
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
import tensorflow

from tensorflow.keras.layers import (
    Conv3D,
    AveragePooling3D,
    MaxPooling3D
)

import sys
class i3d_modified:
    def __init__(self, weights = 'rgb_imagenet_and_kinetics',seq_len=30,input_shape=None,classes=2,shape_=224):
        self.model = Inception_Inflated3d(include_top = True,seq_len=seq_len, weights= weights,endpoint_logit=True,classes=classes,shape_=shape_)
        
    def i3d_flattened(self, num_classes = 2):
        i3d = Model(inputs = self.model.input, outputs = self.model.get_layer(index=-4).output)
        x = conv3d_bn(i3d.output, num_classes, 1, 1, 1, padding='same', use_bias=True, use_activation_fn=False, use_bn=False, name='Conv3d_6a_1x1')
        num_frames_remaining = int(x.shape[1])
        x = Flatten()(x)
        predictions = Dense(num_classes, activation = 'softmax', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01))(x)
        new_model  = Model(inputs = i3d.input, outputs = predictions)
        
        for layer in i3d.layers:
            layer.trainable = True
        
        return new_model
class ResearchModels():
    def __init__(self, nb_classes, model, seq_length,
                 saved_model=None, features_length=2048,image_shape=None,batch_size=None):
        # Set defaults.
        self.seq_length = seq_length
        self.load_model = load_model
        self.saved_model = saved_model
        self.batch_size=batch_size
        self.nb_classes = nb_classes
        self.feature_queue = deque()
        self.image_shape=image_shape
        top3_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=3)
        top3_acc.__name__ = 'top3_acc'
        
        top5_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=5)
        top5_acc.__name__ = 'top5_acc'


        # Set the metrics. Only use top k if there's a need.
        metrics = ['accuracy',top3_acc,top5_acc]

        # Get the appropriate model.
        if self.saved_model is not None:
            top3_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=3)
            top3_acc.__name__ = 'top3_acc'
        
            top5_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=5)
            top5_acc.__name__ = 'top5_acc'

            print("Loading model %s" % self.saved_model)
            self.model = load_model(self.saved_model,custom_objects={'top3_acc': top3_acc,'top5_acc': top5_acc})
        elif model == 'lstm':
            print("Loading LSTM model.")
            self.input_shape = (seq_length, features_length)
            self.model = self.lstm()
        elif model == 'lrcn':
            print("Loading CNN-LSTM model.")
            self.input_shape = (seq_length, image_shape[0], image_shape[1], image_shape[2])
            self.model = self.lrcn()
        elif model == 'mlp':
            print("Loading simple MLP.")
            self.input_shape = (seq_length, features_length)
            self.model = self.mlp()
        elif model == 'capsnet':
            print("Loading simple capsnet.")
            self.input_shape = (seq_length, features_length)
            self.model = self.capsnet()            
        elif model == 'conv_3d':
            print("Loading Conv3D")
            self.input_shape = (seq_length, image_shape[0], image_shape[1], image_shape[2])
            self.model = self.conv_3d()
        elif model == 'gru':
            print("Loading GRU Model")
            self.input_shape = (seq_length, features_length)
            self.model = self.gru()            
        elif model == 'c3d':
            print("Loading C3D")
            self.input_shape = (seq_length, image_shape[0], image_shape[1], image_shape[2])
            self.model = self.c3d()
        elif model == 'i3d':
            print("Loading i3D")
            self.input_shape = (seq_length, image_shape[0], image_shape[1], image_shape[2])
            self.model = self.i3d()   
        elif model == 'r3d_18':
            print("Loading Resnet-3D 18")
            self.input_shape = (seq_length, image_shape[0], image_shape[1], image_shape[2])
            self.model = self.r3d_18()     
        elif model == 'r3d_34':
            print("Loading Resnet-3D 34")
            self.input_shape = (seq_length, image_shape[0], image_shape[1], image_shape[2])
            self.model = self.r3d_34()      
        elif model == 'r3d_50':
            print("Loading Resnet-3D 50")
            self.input_shape = (seq_length, image_shape[0], image_shape[1], image_shape[2])
            self.model = self.r3d_50() 
        elif model == 'r3d_101':
            print("Loading Resnet-3D 101")
            self.input_shape = (seq_length, image_shape[0], image_shape[1], image_shape[2])
            self.model = self.r3d_101()              
        elif model == 'r3d_152':
            print("Loading Resnet-3D 152")
            self.input_shape = (seq_length, image_shape[0], image_shape[1], image_shape[2])
            self.model = self.r3d_152()    
        elif model == 'r3d_152_BERT':
            print("Loading Resnet-3D 152 With BERT")
            self.input_shape = (seq_length, image_shape[0], image_shape[1], image_shape[2])
            self.model = self.r3d_152_BERT() 
        elif model == 'densenet_3d':
            print("Loading densenet_3d")
            self.input_shape = (seq_length, image_shape[0], image_shape[1], image_shape[2])
            self.model = self.densenet_3d()
        elif model == 'densenetresnet_3d':
            print("Loading denseResNet_3d")
            self.input_shape = (seq_length, image_shape[0], image_shape[1], image_shape[2])
            self.model = self.densenetresnet_3d()       
        elif model == 'ALEXNET':
            print("Loading ALEXNET")
            self.input_shape = (seq_length, image_shape[0], image_shape[1], image_shape[2])
            self.model = self.ALEXNET()              
        elif model in['3DR5I2','ours']:
            print("Loading Our model 3DR5I2")
            self.input_shape = (seq_length, image_shape[0], image_shape[1], image_shape[2])
            self.input_shape_features = (seq_length, features_length)
            self.model = self.ours()    
        elif model == 'googlenet':
            print("Loading GoogleNet aka InceptionV1")
            self.input_shape = (seq_length, image_shape[0], image_shape[1], image_shape[2])
            self.model = self.googlenet()                             
        else:
            print("Unknown network.")
            sys.exit()

        # We're using a smaller learning rate
        optimizer = Adam(lr=1e-7, decay=1e-8)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                           metrics=metrics)

        print(self.model.summary())

    def ours(self):
        model_1 = Resnet3DBuilder.build_resnet_50(self.input_shape, num_outputs=self.nb_classes,top=False)
        
        
        base_model = InceptionResNetV2(weights='imagenet', include_top=True)
        model_2 = Model(inputs=base_model.input,outputs=base_model.get_layer('avg_pool').output)
        cat=Concatenate(axis=-1)([model_1.output,model_2.output])

        '''
        x=GRU(120, return_sequences=True,input_shape=((2,1,2048))(cat)
        
        input_image = Input(shape=(self.input_shape))
        A1 = Capsule(8, 6, 1, True)(input_image)
        A2 = Capsule(14, 5, 1, True)(A1)
        A3 = Capsule(14, 5, 1, True)(A2)
        cat=Concatenate(axis=-1)([A2,A3])
        output = Lambda(lambda z: K.sqrt(K.sum(K.square(z), 2)))(cat)
        model = Model(inputs=input_image, outputs=output)        
        model.summary()
        
        '''
        return model


    def gru(self):
        model = tensorflow.keras.Sequential()
        model.add(GRU(120, return_sequences=True,
                       input_shape=self.input_shape))
        model.add(GRU(120, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(50, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))
        return model

    def lstm(self):
        """Build a simple LSTM network. We pass the extracted features from
        our CNN to this model predomenently."""
        # Model.
        model = Sequential()
        model.add(LSTM(2048, return_sequences=False,
                       input_shape=self.input_shape,
                       dropout=0.5))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))

        return model

    def lrcn(self):
        """Build a CNN into RNN.
        Starting version from:
            https://github.com/udacity/self-driving-car/blob/master/
                steering-models/community-models/chauffeur/models.py

        Heavily influenced by VGG-16:
            https://arxiv.org/abs/1409.1556

        Also known as an LRCN:
            https://arxiv.org/pdf/1411.4389.pdf
        """
        def add_default_block(model, kernel_filters, init, reg_lambda):

            # conv
            model.add(TimeDistributed(Conv2D(kernel_filters, (3, 3), padding='same',
                                             kernel_initializer=init, kernel_regularizer=regularizers.l2(l=reg_lambda))))
            model.add(TimeDistributed(BatchNormalization()))
            model.add(TimeDistributed(Activation('relu')))
            # conv
            model.add(TimeDistributed(Conv2D(kernel_filters, (3, 3), padding='same',
                                             kernel_initializer=init, kernel_regularizer=regularizers.l2(l=reg_lambda))))
            model.add(TimeDistributed(BatchNormalization()))
            model.add(TimeDistributed(Activation('relu')))
            # max pool
            model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

            return model

        initialiser = 'glorot_uniform'
        reg_lambda  = 0.001

        model = Sequential()

        # first (non-default) block
        model.add(TimeDistributed(Conv2D(32, (7, 7), strides=(2, 2), padding='same',
                                         kernel_initializer=initialiser, kernel_regularizer=regularizers.l2(0.01)),
                                  input_shape=self.input_shape))
        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(Activation('relu')))
        model.add(TimeDistributed(Conv2D(32, (3,3), kernel_initializer=initialiser, kernel_regularizer=regularizers.l2(0.01))))
        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(Activation('relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

        # 2nd-5th (default) blocks
        model = add_default_block(model, 64,  init=initialiser, reg_lambda=reg_lambda)
        model = add_default_block(model, 128, init=initialiser, reg_lambda=reg_lambda)
        model = add_default_block(model, 256, init=initialiser, reg_lambda=reg_lambda)
        model = add_default_block(model, 512, init=initialiser, reg_lambda=reg_lambda)

        # LSTM output head
        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(256, return_sequences=False, dropout=0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))

        return model

    def mlp(self):
        """Build a simple MLP. It uses extracted features as the input
        because of the otherwise too-high dimensionality."""
        # Model.
        model = Sequential()
        model.add(Flatten(input_shape=self.input_shape))
        model.add(Dense(512))
        model.add(Dropout(0.5))
        model.add(Dense(512))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))

        return model

    def conv_3d(self):
        """
        Build a 3D convolutional network, based loosely on C3D.
            https://arxiv.org/pdf/1412.0767.pdf
        """
        # Model.
        model = Sequential()
        model.add(Conv3D(
            32, (3,3,3), activation='relu', input_shape=self.input_shape
        ))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
        model.add(Conv3D(64, (3,3,3), activation='relu'))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
        model.add(Conv3D(128, (3,3,3), activation='relu'))
        model.add(Conv3D(128, (3,3,3), activation='relu'))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
        model.add(Conv3D(256, (2,2,2), activation='relu'))
        model.add(Conv3D(256, (2,2,2), activation='relu'))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))

        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Dropout(0.5))
        model.add(Dense(1024))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))

        return model
    def googlenet(self):
        input_layer = Input(shape = (224, 224, 3))
      
        # convolutional layer: filters = 64, kernel_size = (7,7), strides = 2
        X = Conv2D(filters = 64, kernel_size = (7,7), strides = 2, padding = 'valid', activation = 'relu')(input_layer)
      
        # max-pooling layer: pool_size = (3,3), strides = 2
        X = MaxPooling2D(pool_size = (3,3), strides = 2)(X)
      
        # convolutional layer: filters = 64, strides = 1
        X = Conv2D(filters = 64, kernel_size = (1,1), strides = 1, padding = 'same', activation = 'relu')(X)
      
        # convolutional layer: filters = 192, kernel_size = (3,3)
        X = Conv2D(filters = 192, kernel_size = (3,3), padding = 'same', activation = 'relu')(X)
      
        # max-pooling layer: pool_size = (3,3), strides = 2
        X = MaxPooling2D(pool_size= (3,3), strides = 2)(X)
      
        # 1st Inception block
        X = Inception_block(X, f1 = 64, f2_conv1 = 96, f2_conv3 = 128, f3_conv1 = 16, f3_conv5 = 32, f4 = 32)
      
        # 2nd Inception block
        X = Inception_block(X, f1 = 128, f2_conv1 = 128, f2_conv3 = 192, f3_conv1 = 32, f3_conv5 = 96, f4 = 64)
      
        # max-pooling layer: pool_size = (3,3), strides = 2
        X = MaxPooling2D(pool_size= (3,3), strides = 2)(X)
      
        # 3rd Inception block
        X = Inception_block(X, f1 = 192, f2_conv1 = 96, f2_conv3 = 208, f3_conv1 = 16, f3_conv5 = 48, f4 = 64)
      
        # Extra network 1:
        X1 = AveragePooling2D(pool_size = (5,5), strides = 3)(X)
        X1 = Conv2D(filters = 128, kernel_size = (1,1), padding = 'same', activation = 'relu')(X1)
        X1 = Flatten()(X1)
        #X1 = Dense(1024, activation = 'relu')(X1)
        #X1 = Dropout(0.7)(X1)
        #X1 = Dense(5, activation = 'softmax')(X1)
      
        
        # 4th Inception block
        X = Inception_block(X, f1 = 160, f2_conv1 = 112, f2_conv3 = 224, f3_conv1 = 24, f3_conv5 = 64, f4 = 64)
      
        # 5th Inception block
        X = Inception_block(X, f1 = 128, f2_conv1 = 128, f2_conv3 = 256, f3_conv1 = 24, f3_conv5 = 64, f4 = 64)
      
        # 6th Inception block
        X = Inception_block(X, f1 = 112, f2_conv1 = 144, f2_conv3 = 288, f3_conv1 = 32, f3_conv5 = 64, f4 = 64)
      
        # Extra network 2:
        X2 = AveragePooling2D(pool_size = (5,5), strides = 3)(X)
        X2 = Conv2D(filters = 128, kernel_size = (1,1), padding = 'same', activation = 'relu')(X2)
        X2 = Flatten()(X2)
        #X2 = Dense(1024, activation = 'relu')(X2)
        #X2 = Dropout(0.7)(X2)
        #X2 = Dense(1000, activation = 'softmax')(X2)
        
        
        # 7th Inception block
        X = Inception_block(X, f1 = 256, f2_conv1 = 160, f2_conv3 = 320, f3_conv1 = 32, 
                            f3_conv5 = 128, f4 = 128)
      
        # max-pooling layer: pool_size = (3,3), strides = 2
        X = MaxPooling2D(pool_size = (3,3), strides = 2)(X)
      
        # 8th Inception block
        X = Inception_block(X, f1 = 256, f2_conv1 = 160, f2_conv3 = 320, f3_conv1 = 32, f3_conv5 = 128, f4 = 128)
      
        # 9th Inception block
        X = Inception_block(X, f1 = 384, f2_conv1 = 192, f2_conv3 = 384, f3_conv1 = 48, f3_conv5 = 128, f4 = 128)
      
        # Global Average pooling layer 
        X = GlobalAveragePooling2D(name = 'GAPL')(X)
      
        # Dropoutlayer 
        #X = Dropout(0.4)(X)
      
        # output layer 
        #X = Dense(1000, activation = 'softmax')(X)
        
        # model
        model = Model(input_layer, [X, X1, X2], name = 'GoogLeNet')
       
        model.summary()
        return model    
    
        def Inception_block(input_layer, f1, f2_conv1, f2_conv3, f3_conv1, f3_conv5, f4): 
          # Input: 
          # - f1: number of filters of the 1x1 convolutional layer in the first path
          # - f2_conv1, f2_conv3 are number of filters corresponding to the 1x1 and 3x3 convolutional layers in the second path
          # - f3_conv1, f3_conv5 are the number of filters corresponding to the 1x1 and 5x5  convolutional layer in the third path
          # - f4: number of filters of the 1x1 convolutional layer in the fourth path
        
          # 1st path:
          path1 = Conv2D(filters=f1, kernel_size = (1,1), padding = 'same', activation = 'relu')(input_layer)
        
          # 2nd path
          path2 = Conv2D(filters = f2_conv1, kernel_size = (1,1), padding = 'same', activation = 'relu')(input_layer)
          path2 = Conv2D(filters = f2_conv3, kernel_size = (3,3), padding = 'same', activation = 'relu')(path2)
        
          # 3rd path
          path3 = Conv2D(filters = f3_conv1, kernel_size = (1,1), padding = 'same', activation = 'relu')(input_layer)
          path3 = Conv2D(filters = f3_conv5, kernel_size = (5,5), padding = 'same', activation = 'relu')(path3)
        
          # 4th path
          path4 = MaxPooling2D((3,3), strides= (1,1), padding = 'same')(input_layer)
          path4 = Conv2D(filters = f4, kernel_size = (1,1), padding = 'same', activation = 'relu')(path4)
        
          output_layer = concatenate([path1, path2, path3, path4], axis = -1)
        
          return output_layer
    def ALEXNET(self):
        AlexNet = Sequential()
        #1st Convolutional Layer
        AlexNet.add(Conv2D(filters=96, input_shape=(32,32,3), kernel_size=(11,11), strides=(4,4), padding='same'))
        AlexNet.add(BatchNormalization())
        AlexNet.add(Activation('relu'))
        AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

        #2nd Convolutional Layer
        AlexNet.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1,1), padding='same'))
        AlexNet.add(BatchNormalization())
        AlexNet.add(Activation('relu'))
        AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

        #3rd Convolutional Layer
        AlexNet.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
        AlexNet.add(BatchNormalization())
        AlexNet.add(Activation('relu'))

        #4th Convolutional Layer
        AlexNet.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
        AlexNet.add(BatchNormalization())
        AlexNet.add(Activation('relu'))

        #5th Convolutional Layer
        AlexNet.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
        AlexNet.add(BatchNormalization())
        AlexNet.add(Activation('relu'))
        AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

        #Passing it to a Fully Connected layer
        AlexNet.add(Flatten())
                    
        # We'll extract features at the final pool layer.
        model = AlexNet        
        model.summary()
        return model
        
    def c3d(self):
        """
        Build a 3D convolutional network, aka C3D.
            https://arxiv.org/pdf/1412.0767.pdf

        With thanks:
            https://gist.github.com/albertomontesg/d8b21a179c1e6cca0480ebdf292c34d2
        """
        model = Sequential()
        # 1st layer group
        model.add(Conv3D(64, (3, 3, 3), activation='relu',
                         name='conv1',
                         padding='same',
                         dilation_rate=(1, 1, 1),
                         input_shape=self.input_shape))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                               name='pool1'))
        # 2nd layer group
        model.add(Conv3D(128, (3, 3, 3), activation='relu',
                         name='conv2',
                         padding='same',
                         dilation_rate=(1, 1, 1)))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               name='pool2'))
        # 3rd layer group
        model.add(Conv3D(256, (3, 3, 3), activation='relu',
                         name='conv3a',
                         padding='same',
                         dilation_rate=(1, 1, 1)))
        model.add(Conv3D(256, (3, 3, 3), activation='relu',
                         name='conv3b',
                         dilation_rate=(1, 1, 1)))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               name='pool3'))
        # 4th layer group
        model.add(Conv3D(512, (3, 3, 3), activation='relu',
                         name='conv4a',
                         padding='same',
                         dilation_rate=(1, 1, 1)))
        model.add(Conv3D(512, (3, 3, 3), activation='relu',
                         name='conv4b',
                         padding='same',
                         dilation_rate=(1, 1, 1)))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               name='pool4'))

        # 5th layer group
        model.add(Conv3D(512, (3, 3, 3), activation='relu',
                         name='conv5a',
                         padding='same',
                         dilation_rate=(1, 1, 1)))
        model.add(Conv3D(512, (3, 3, 3), activation='relu',
                         name='conv5b',
                         padding='same',
                         dilation_rate=(1, 1, 1)))
        model.add(ZeroPadding3D(padding=(0, 1, 1)))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               name='pool5'))
        model.add(Flatten())

        # FC layers group
        model.add(Dense(4096, activation='relu', name='fc6'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu', name='fc7'))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))

        return model
    def capsnet(self):
        
        input_image = Input(shape=(self.input_shape))
        A1 = Capsule(8, 6, 1, True)(input_image)
        A2 = Capsule(14, 5, 1, True)(A1)
        A3 = Capsule(14, 5, 1, True)(A2)
        cat=Concatenate(axis=-1)([A2,A3])
        output = Lambda(lambda z: K.sqrt(K.sum(K.square(z), 2)))(cat)
        model = Model(inputs=input_image, outputs=output)
        
        
        return model
        
    def i3d(self):
        weights='rgb_kinetics_only'
        #weights='rgb_imagenet_and_kinetics'
        i3d = i3d_modified(weights = weights,seq_len=self.seq_length,shape_=self.input_shape[1],classes=self.nb_classes)
        model = i3d.i3d_flattened(num_classes = self.nb_classes)
        
        
        
        
        model.summary()
        return model
        
            
    def r3d_18(self):
        regularization_factor = 2.5e-2
        shape=(self.input_shape[0],self.input_shape[1],self.input_shape[2])
        model = Resnet3DBuilder.build_resnet_18(self.input_shape, num_outputs=self.nb_classes)
        model.summary()
        return model
    def r3d_34(self):
        regularization_factor = 2.5e-2
        shape=(self.input_shape[0],self.input_shape[1],self.input_shape[2])
        model = Resnet3DBuilder.build_resnet_34(self.input_shape, num_outputs=self.nb_classes)
        model.summary()
        return model  
    def r3d_50(self):
        regularization_factor = 2.5e-2
        shape=(self.input_shape[0],self.input_shape[1],self.input_shape[2])
        model = Resnet3DBuilder.build_resnet_50(self.input_shape, num_outputs=self.nb_classes)
        model.summary()
        return model  
    def r3d_101(self):
        regularization_factor = 2.5e-2
        shape=(self.input_shape[0],self.input_shape[1],self.input_shape[2])
        model = Resnet3DBuilder.build_resnet_101(self.input_shape, num_outputs=self.nb_classes)
        model.summary()
        return model       
    def r3d_152(self):
        regularization_factor = 2.5e-2
        shape=(self.input_shape[0],self.input_shape[1],self.input_shape[2])
        model = Resnet3DBuilder.build_resnet_152(self.input_shape, num_outputs=self.nb_classes)
        model.summary()
        return model
    
    def r3d_152_BERT(self):
        regularization_factor = 2.5e-2
        shape=(self.input_shape[0],self.input_shape[1],self.input_shape[2])
        model = Resnet3DBuilder_BERT.build_resnet_152(self.input_shape, num_outputs=self.nb_classes)
        model.summary()
        return model                      
 
    def densenet_3d(self):
        model = densenet_3d(nb_classes=self.nb_classes, input_shape=self.input_shape, dropout_rate=0.2)
        return model
    def densenetresnet_3d(self):
        model = dense_resnet_3d(nb_classes=self.nb_classes, input_shape=self.input_shape, dropout_rate=0.2)
        return model                            




