from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
import numpy as np

class Extractor():
    def __init__(self, weights=None):
        """Either load pretrained from imagenet, or load our saved
        weights from our own training."""

        self.weights = weights  # so we can check elsewhere which model

        if weights is None:
            # Get model with pretrained weights.
            base_model = VGG16(weights='imagenet', include_top=True)
            base_model.summary()
            
            # We'll extract features at the final pool layer.
            self.model = Model(
                inputs=base_model.input,
                outputs=base_model.get_layer('flatten').output
            )

        else:
            # Load the model first.
            self.model = load_model(weights)
            
            # pop top 3 layers to get  maxpool layer
            #self.model.layers.pop() 
            #self.model.layers.pop()
            self.model.layers.pop()
            self.model.layers.pop()
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
