from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing import image
import tensorflow
from tensorflow.keras.applications.mobilenet import preprocess_input

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model, load_model
import numpy as np


class Extractor():
    def __init__(self, weights=None):
        """Either load pretrained from imagenet, or load our saved
        weights from our own training."""

        self.weights = weights  # so we can check elsewhere which model

        if weights is None:
            base_model =tensorflow.keras.applications.mobilenet.MobileNet(input_shape=None, alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=True, weights='imagenet')
            
            # We'll extract features at the final pool layer.
            self.model = Model(
                inputs=base_model.input,
                outputs=base_model.layers[-6].output
            )

            self.model.summary()

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
