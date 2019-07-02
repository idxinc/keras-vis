import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D


def get_data():

    train_file = "../../Pictures/train.npz"
    test_file = "../../Pictures/test.npz"

    train_db = np.load(train_file)
    test_db = np.load(test_file)

    X_train = train_db["features"]
    y_train = train_db["labels"]
    X_test = test_db["features"]
    y_test = test_db["labels"]

    y_train = ((y_train + 1) // 2)
    y_train = np.array([y_train, 1 - y_train]).transpose()

    y_test = ((y_test + 1) // 2)
    y_test = np.array([y_test, 1 - y_test]).transpose()

    # y_train = OneHotEncoder()
    X_train = np.reshape(X_train, (X_train.shape[0], 256, 256, 1))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, 256, 256, 1))

    print(f"Train size: {X_train.shape}")
    print(f"Test size: {X_test.shape}")

    return X_train, y_train, X_test, y_test


def make_model():
    # Instantiate an empty model
    model = Sequential()

    # 1st Convolutional Layer
    model.add(Conv2D(filters=96, input_shape=(256, 256, 1), kernel_size=(11, 11), strides=(4, 4)))
    model.add(Activation("relu"))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # 2nd Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(11, 11), strides=(1, 1)))
    model.add(Activation("relu"))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # 3rd Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1)))
    model.add(Activation("relu"))

    # 4th Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1)))
    model.add(Activation("relu"))

    # 5th Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1)))
    model.add(Activation("relu"))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Passing it to a Fully Connected layer
    model.add(Flatten())
    # 1st Fully Connected Layer
    model.add(Dense(4096))
    model.add(Activation("relu"))
    # Add Dropout to prevent overfitting
    model.add(Dropout(0.4))

    # 2nd Fully Connected Layer
    model.add(Dense(4096))
    model.add(Activation("relu"))
    # Add Dropout
    model.add(Dropout(0.4))

    # 3rd Fully Connected Layer
    model.add(Dense(1000))
    model.add(Activation("relu"))
    # Add Dropout
    model.add(Dropout(0.4))

    # Output Layer
    model.add(Dense(2))
    model.add(Activation("softmax", name="preds"))

    # Compile the model
    model.compile(loss=keras.losses.binary_crossentropy, optimizer="sgd")

    return model