import os

import keras
import numpy as np

# Classe ArrayDataGenerator :
# Permet de charger des tableaux enregistrés au format compressé .npz
# de la même manière que la classe ImageDataGenerator fournie par Keras.
class ArrayDataGenerator(keras.utils.Sequence):
    def __init__(self, data_folder, batch_size, input_shape, shuffle=True):
        self.batch_size = batch_size
        self.data_folder = data_folder
        self.shuffle = shuffle
        self.input_shape = input_shape

        self.id_list = []

        for directory in os.listdir(self.data_folder):
            dir_path = os.path.join(self.data_folder, directory)
            for img_name in os.listdir(dir_path):
                id_name = directory + '/' + img_name
                self.id_list.append(id_name)

        print('Found', len(self.id_list), 'arrays.')

        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.id_list))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.floor(len(self.id_list) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        batch = [self.id_list[k] for k in indexes]

        X = np.empty((self.batch_size, *self.input_shape))
        y = np.empty(self.batch_size, dtype=int)

        for i, filename in enumerate(batch):
            X[i,] = np.load(self.data_folder + filename)['arr_0']
            y[i] = int(filename.partition('/')[0])

        return X, keras.utils.to_categorical(y, num_classes=200)

