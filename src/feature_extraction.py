from keras import Sequential, callbacks, optimizers
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.optimizers import Adam

from class_utils import ArrayDataGenerator
from func_utils import train_model, plot_acc_loss

data_path = 'drive/My Drive/TER/'

train_gen = ArrayDataGenerator(data_path + 'feat_train_224/', input_shape = (7, 7, 2048), batch_size = 32)
val_gen = ArrayDataGenerator(data_path + 'feat_val_224/', input_shape = (7, 7, 2048), batch_size = 32)

model = Sequential()

model.add(GlobalAveragePooling2D(input_shape = (7, 7, 2048)))
model.add(Dense(200, activation = 'softmax'))

early_stop = callbacks.EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=10)
opt = Adam(lr = 1e-3)

history = train_model(model, train_gen, val_gen, 30, callbacks = [early_stop], opt = opt)

plot_acc_loss(history)

model.save(data_path + 'models/test_model.h5')