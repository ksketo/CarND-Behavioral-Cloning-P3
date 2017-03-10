import csv
import cv2
import numpy as np
import sklearn
import gc

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda, Cropping2D
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from tqdm import tqdm


class SDCModel(object):
    def __init__(self):
        self.data_path = 'data/'
        self.csv_path = 'data/driving_log.csv'
        self.batch_size = 32
        self.nb_classes = 1
        self.nb_epoch = 8

        self.ch, self.row, self.col = 3, 160, 320
        self.keep_prob = 0.2

        self.model = Sequential()

    def data_loading(self):
        """
        Load Udacity data and create train and validation set
        """
        lines = []
        with open(self.csv_path) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                lines.append(line)

        train_samples, validation_samples = train_test_split(lines[1:], test_size=0.2)
        return train_samples, validation_samples

    def image_flipping(self, image, steering=0):
        """
        Flip image to fix track left turn biasing
        """
        new_image, new_steering = cv2.flip(image, 1), -1.0 * steering
        return new_image, new_steering

    def image_brightness(self, image, steering=0):
        """
        Randomly change image brightness
        """
        new_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        random_brightness = 0.2 + np.random.uniform(0.2, 0.6)
        new_image[:, :, 2] = new_image[:, :, 2] * random_brightness
        new_image = cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)
        return new_image, steering

    def image_shifting(self, image, steering=0):
        """
        Randomly shift image on the X axis
        """
        max_shift = 55
        max_ang = 0.14  # ang_per_pixel = 0.0025

        rows, cols, _ = image.shape

        random_x = np.random.randint(-max_shift, max_shift + 1)
        new_steering = steering + (random_x / max_shift) * max_ang
        mat = np.float32([[1, 0, random_x], [0, 1, 0]])
        new_image = cv2.warpAffine(image, mat, (cols, rows))
        return new_image, new_steering

    def image_shadow(self, image, steering=0):
        """
        Generate shadow in random region of the image
        """
        top_x, bottom_x = np.random.randint(0, 160, 2)
        rows, cols, _ = image.shape
        new_image = image.copy()
        mask = image.copy()

        vertices = np.array([[(50, 65), (45, 0), (145, 0), (150, 65)]], dtype=np.int32)
        if np.random.randint(2):
            vertices = np.array([[top_x, 0], [cols, 0], [cols, rows], [bottom_x, rows]], dtype=np.int32)
        else:
            vertices = np.array([[top_x, 0], [0, 0], [0, rows], [bottom_x, rows]], dtype=np.int32)

        cv2.fillPoly(mask, [vertices], (0,) * image.shape[2])
        rand_a = np.random.uniform(0.5, 0.75)
        cv2.addWeighted(mask, rand_a, image, 1 - rand_a, 0., new_image)

        return new_image, steering

    def transform_image(self, image, angle):
        """
        Apply randomly image flipping, shifting, different brightness and shadow transformations
        """
        if np.random.randint(2):
            image, angle = self.image_flipping(image, angle)
        if np.random.randint(2):
            image, angle = self.image_shifting(image, angle)
        if np.random.randint(2):
            image, angle = self.image_brightness(image, angle)
        if np.random.randint(2):
            image, angle = self.image_shadow(image, angle)
        return (image, angle)

    def generator(self, samples, batch_size=32):
        """
        Generator for passing data to the model batch by batch
        """
        num_samples = len(samples)
        while 1:
            samples = shuffle(samples)
            for offset in range(0, num_samples, batch_size):
                batch_samples = samples[offset:offset+batch_size]

                images = []
                angles = []
                for line in batch_samples:
                    rint = np.random.randint(3)
                    image = cv2.imread(self.data_path + line[rint].strip())
                    angle = float(line[3]) + ([0.0, 0.22, -0.22][rint])
                    new_image, new_angle = self.transform_image(image, angle)
                    images.append(new_image)
                    angles.append(new_angle)

                X_train = np.array(images)
                y_train = np.array(angles)
                yield sklearn.utils.shuffle(X_train, y_train)

    def model_architecture(self, model):
        """
        Nvidia model architecture
        """
        model.add(Lambda(lambda x: x / 255.0 - 0.5,
        		input_shape=(self.row, self.col, self.ch)))
        model.add(Cropping2D(cropping=((60, 20), (0,0)), input_shape=(self.row, self.col, self.ch)))
        model.add(Convolution2D(24, 5, 5, subsample=(2, 2)))
        model.add(Activation('relu'))
        model.add(Dropout(self.keep_prob))

        model.add(Convolution2D(36, 5, 5, subsample=(2, 2)))
        model.add(Activation('relu'))
        model.add(Dropout(self.keep_prob))

        model.add(Convolution2D(48, 5, 5, subsample=(2, 2)))
        model.add(Activation('relu'))
        model.add(Dropout(self.keep_prob))

        model.add(Convolution2D(64, 3, 3, subsample=(1, 1)))
        model.add(Activation('relu'))
        model.add(Dropout(self.keep_prob))

        model.add(Convolution2D(64, 3, 3, subsample=(1, 1)))
        model.add(Activation('relu'))
        model.add(Dropout(self.keep_prob))

        model.add(Flatten())

        model.add(Dense(100))
        model.add(Activation('relu'))
        model.add(Dropout(self.keep_prob))

        model.add(Dense(50))
        model.add(Activation('relu'))
        model.add(Dropout(self.keep_prob))

        model.add(Dense(10))
        model.add(Activation('relu'))
        model.add(Dense(1))

        model.compile(loss='mse', optimizer='adam')

        return model

    def train_model(self):
        """
        Train model and save weights
        """
        # create train and validation set
        train_samples, validation_samples = self.data_loading()

        # compile and train the model using the generator function
        train_generator = self.generator(train_samples, batch_size=self.batch_size)
        validation_generator = self.generator(validation_samples, batch_size=self.batch_size)

        early_stop = EarlyStopping(monitor='val_loss', patience=2,
                                   verbose=0, mode='min')
        checkpoint = ModelCheckpoint('checkpoints/model-cp-{epoch:02d}.h5',
                                    monitor='val_loss',verbose=0, save_best_only=True,
                                    save_weights_only=False, mode='auto')

        self.model = self.model_architecture(self.model)
        self.model.fit_generator(
            train_generator,
            samples_per_epoch=len(train_samples),
            validation_data=validation_generator,
            nb_val_samples=len(validation_samples),
            nb_epoch=self.nb_epoch,
            callbacks=[early_stop, checkpoint]
        )
        self.model.save('model.h5')


if __name__ == '__main__':
    sdc_model = SDCModel()
    sdc_model.train_model()
    gc.collect()
    K.clear_session()
