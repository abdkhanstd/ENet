from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import BatchNormalization,Input,Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
import numpy as np

class Extractor():
    def __init__(self, weights=None):
        """Either load pretrained from imagenet, or load our saved
        weights from our own training."""

        self.weights = weights  # so we can check elsewhere which model

        if weights is None:
            # Get model with pretrained weights.
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
            self.model = AlexNet
            AlexNet.summary()

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
