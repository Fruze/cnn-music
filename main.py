from keras.engine.saving import model_from_json
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras_preprocessing.image import ImageDataGenerator


class CNN:

    USE_EXISTING_MODEL = True

    def __init__(self):
        model = self.get_model(self.USE_EXISTING_MODEL)

        # TODO: Detect
        # TODO: Cut from music partiture
        # print(model.predict(x_test[:4]))

    def get_model(self, use_existing_model):
        print("Getting model..")
        if use_existing_model:
            return self.load_model()

        data = self.load_data()
        model = self.build_model(data)

        print("Getting model is Finished..")
        return model

    @staticmethod
    def load_model():
        print("Loading model..")
        json_file = open("model/model_structure.json", "r")
        loaded_model_json = json_file.read()
        json_file.close()

        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("model/model_weight.h5")
        print("Loading model is Finished..")

    @staticmethod
    def save_model(model):
        with open("model/model_structure.json", "w") as json_file:
            print("Saving / Replacing model..")
            model_json = model.to_json()
            json_file.write(model_json)

            model.save_weights("model/model_weight.h5")
            print("Saving / Replacing model is Finished..")

    @staticmethod
    def load_data():
        print("Loading data..")

        image_data_generator = ImageDataGenerator()
        train_data = image_data_generator.flow_from_directory(
            directory="data/train/",
            batch_size=1,
            class_mode="categorical",
            shuffle=True,
            seed=42,
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

        print("Loading data is Finished..")
        return train_data

    def build_model(self, data):
        print("Building model..")
        model = Sequential()

        model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(256, 64, 1)))
        model.add(Conv2D(32, kernel_size=3, activation='relu'))
        model.add(Flatten())
        model.add(Dense(11, activation='softmax'))
        model.compile(optimizer="Adam", loss="binary_crossentropy", metrics=["acc"])

        model.fit_generator(
            generator=data,
            validation_data=data,
            epochs=1
        )

        # TODO: Evaluate
        # model.evaluate_generator(
        #     generator=data,
        #     steps=train_step
        # )

        self.save_model(model)

        print("Building model is Finished..")
        return model


if __name__ == "__main__":
    CNN()
