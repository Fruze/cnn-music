from keras.engine.saving import model_from_json
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras_preprocessing.image import ImageDataGenerator, os


class CNN:

    USE_EXISTING_MODEL = False

    def __init__(self):
        model = self.get_model(self.USE_EXISTING_MODEL)
        # self.predict_data(model)
        # TODO: Cut from music sheet

    def get_model(self, use_existing_model):
        print("Getting model..")
        if use_existing_model:
            return self.load_model()

        train_data, validation_data = self.load_data()
        model = self.build_model(train_data, validation_data)

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

        return loaded_model

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

        validation_data = image_data_generator.flow_from_directory(
            directory="data/valid/",
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
        return train_data, validation_data

    def build_model(self, train_data, validation_data):
        print("Building model..")
        model = Sequential()

        model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(256, 64, 1)))
        model.add(Conv2D(32, kernel_size=3, activation='relu'))
        model.add(Flatten())
        model.add(Dense(13, activation='softmax'))
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        model.fit_generator(
            generator=train_data,
            validation_data=validation_data,
            epochs=50
        )

        # TODO: Evaluate
        # model.evaluate_generator(
        #     generator=data,
        #     steps=train_step
        # )

        self.save_model(model)

        print("Building model is Finished..")
        return model

    @staticmethod
    def predict_data(model):
        image_data_generator = ImageDataGenerator()
        test_data = image_data_generator.flow_from_directory(
            directory="data/test/",
            batch_size=1,
            class_mode=None,
            shuffle=False,
            color_mode="grayscale",
            target_size=(256, 64)
        )

        result = model.predict(test_data)
        print(result)

        classes = os.listdir("data/train")
        if ".DS_Store" in classes:
            classes.remove(".DS_Store")
            classes = sorted(classes)

        for val in result.argmax(axis=-1):
            print(classes[val])

        print([map(int, classes[x]) for x in result.argmax(axis=-1)])


if __name__ == "__main__":
    CNN()
