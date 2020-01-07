import numpy
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras_preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
import cv2


def main():
    # TODO: Train
    data = load_data()
    # print(len(data))
    model = build_model(data)

    # TODO: Save Model
    # TODO: Load Model

    # TODO: Detect
    # TODO: Cut from music partiture
    # print(model.predict(x_test[:4]))


def load_data():
    image_data_generator = ImageDataGenerator()
    data = image_data_generator.flow_from_directory(
        directory="dataset",
        color_mode="grayscale",
        target_size=(256, 64)
    )

    # For showing data
    # x, y = train_data.next()
    # for i in range(0, 3):
    #     # Image file
    #     image = x[i]
    #
    #     # Label = directory label, sort by name
    #     label = y[i]
    #
    #     pyplot.imshow(numpy.squeeze(image))
    #     pyplot.show()

    return data


def build_model(data):
    model = Sequential()

    model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(256, 64, 1)))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(11, activation='softmax'))
    model.compile(optimizer="Adam", loss="binary_crossentropy", metrics=["acc"])

    model.fit_generator(data, validation_data=data, epochs=50)

    return model


if __name__ == "__main__":
    main()
