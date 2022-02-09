from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
import numpy as np


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
class Extractor():

  
    def __init__(self, weights=None):
        """Either load pretrained from imagenet, or load our saved
        weights from our own training."""

        self.weights = weights  # so we can check elsewhere which model

      
        if weights is None:
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
            model1 = Model(input_layer, [X, X1, X2], name = 'GoogLeNet')
            print("*******************",len([X, X1, X2]))
            model1.summary()
            self.model = model1

        else:
            # Load the model first.
            self.model = load_model(weights)
            
            # pop top 1 layers to get  maxpool layer
            self.model.layers.pop()

            
            self.model.outputs = [self.model.layers[-1].output]
            self.model.output_layers = [self.model.layers[-1]]
            self.model.layers[-1].outbound_nodes = []

    def extract(self, image_path):
        img = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Get the prediction.
        features = self.model.predict(x)

        if self.weights is None:
            # For imagenet/default network:
            features = features[0]
        else:
            # For loaded network:
            features = features[0]

        return features
