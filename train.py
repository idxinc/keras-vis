from __future__ import print_function

import numpy as np
from functions import get_data, make_model
import cv2
import matplotlib.cm as cm
from vis.visualization import visualize_cam, overlay
from vis.utils import utils
from keras import activations
import keras
from keras.applications.vgg16 import VGG16, preprocess_input
from matplotlib import pyplot as plt

x_train, y_train, x_test, y_test = get_data()

model = VGG16(weights=None, input_shape=(256, 256, 1), classes=2)
model.compile(loss=keras.losses.binary_crossentropy, optimizer="sgd")
epochs = 6

try:
    model.load_weights("model.h5")
except ValueError:
    model.fit(x_train, y_train,
              epochs=epochs,
              verbose=1)

    model.save_weights("model.h5")

# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score)
# print('Test accuracy:', score)

class_idx = 0
indices = np.where(y_test[:, class_idx] == 1.)[0]

# pick some random input from here.
idx = indices[0]

# Lets sanity check the picked image.
cv2.imwrite("test/raw.jpg", x_test[idx][0] * 255.)
# Utility to search for layer index by name.
# Alternatively we can specify this as -1 since it corresponds to the last layer.
layer_idx = utils.find_layer_idx(model, 'predictions')

# Swap softmax with linear
model.layers[layer_idx].activation = activations.linear
model = utils.apply_modifications(model)

penultimate_layer = utils.find_layer_idx(model, 'block5_conv3')

for modifier in [None, 'guided', 'relu']:
    plt.suptitle("vanilla" if modifier is None else modifier)
    for i, img in enumerate([x_test[indices[0]], x_test[indices[1]]]):
        grads = visualize_cam(model, layer_idx, filter_indices=0,
                              seed_input=img, penultimate_layer_idx=penultimate_layer,
                              backprop_modifier=modifier)
        # Lets overlay the heatmap onto original image.
        jet_heatmap = np.uint8(cm.jet(grads)[..., :3] * 255)
        cv2.imwrite("test/heatmap" + str(i) + ".jpg", jet_heatmap)
        cv2.imwrite("test/" + str(modifier) + "_" + str(i) + ".jpg", overlay(jet_heatmap, cv2.cvtColor(img[0] * 255., cv2.COLOR_GRAY2RGB)))



