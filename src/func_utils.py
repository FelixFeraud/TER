import os

import keras
import numpy as np
from keras.layers import Input
from keras.models import Model
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import matplotlib.pyplot as plt


# Crée un répertoire si il n'existe pas déjà
def makedirs_if(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# Preprocess compact pour resnet
def img_preprocess(img_path, preprocessing_function, n=224):
    img = load_img(img_path, target_size=(n, n))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    preprocessed_img = preprocessing_function(img)
    return preprocessed_img


# Prédictions (top k, par défaut 5) sur les images de test pour l'extraction de caractéristiques (feature extraction)
# nécessite de préciser un modèle de base pour que le modèle final fasse ses prédictions sur les caractéristiques extraites.
# enregistre les résultats au format de la compétition dans un fichier csv au chemin 'csv_output'
def fe_top_k_predict(directory, base_model, final_model, preprocessing_function, csv_output, k=5):
    img_count = sum([len(files) for r, d, files in os.walk(directory)])
    print('Found', img_count, 'images')
    with open(csv_output, 'w', newline='') as csv_file:
        csv_file.write('Id,Category')
        cpt = 0
        for img_name in os.listdir(directory):

            img_path = os.path.join(directory, img_name)
            preprocessed_img = img_preprocess(img_path, preprocessing_function)

            features = base_model.predict(preprocessed_img)
            predictions = np.ravel(final_model.predict(features))
            top_values_index = sorted(range(len(predictions)), key=lambda i: predictions[i])[-k:]

            csv_file.write('\n' + str(img_name.partition('.')[0]) + ',')

            for index in top_values_index:
                csv_file.write(str(index) + ' ')

            cpt += 1
            if cpt % 100 == 0:
                print('Predicted', cpt, 'images')

# Prédictions (top k, par défaut 5) sur les images du répertoire 'directory'
def ft_top_k_predict(directory, model, preprocessing_function, csv_output, k=5):
    img_count = sum([len(files) for r, d, files in os.walk(directory)])
    print('Found', img_count, 'images')
    with open(csv_output, 'w', newline='') as csv_file:
        csv_file.write('Id,Category')
        cpt = 0
        for img_name in os.listdir(directory):

            img_path = os.path.join(directory, img_name)
            preprocessed_img = img_preprocess(img_path, preprocessing_function)

            predictions = np.ravel(model.predict(preprocessed_img))
            top_values_index = sorted(range(len(predictions)), key=lambda i: predictions[i])[-k:]

            csv_file.write('\n' + str(img_name.partition('.')[0]) + ',')

            for index in top_values_index:
                csv_file.write(str(index) + ' ')

            cpt += 1
            if cpt % 100 == 0:
                print('Predicted', cpt, 'images')

# Renvoie un générateur de données (batchs) pour l'ensemble d'entraînement, et un générateur pour l'ensemble de validation
# à partir d'un dossier 'directory' spécifié, qui doit contenir les dossier 'train' et 'val'.
def setup_data_generators(directory, image_size, gen_batch_size, model_preprocessing_function):
    train_data_gen = keras.preprocessing.image.ImageDataGenerator(preprocessing_function=model_preprocessing_function,
                                                                  rotation_range=20, width_shift_range=0.2,
                                                                  height_shift_range=0.2, shear_range=0.2)
    val_data_gen = keras.preprocessing.image.ImageDataGenerator(preprocessing_function=model_preprocessing_function)

    train_gen = train_data_gen.flow_from_directory(directory + 'train/', target_size=image_size,
                                                   batch_size=gen_batch_size, shuffle=True, class_mode='categorical')
    val_gen = val_data_gen.flow_from_directory(directory + 'val/', target_size=image_size, batch_size=gen_batch_size,
                                               class_mode='categorical')

    return train_gen, val_gen

# Prend en arguments, un modèle de base (ResNet50, VGG19 ...) et un constructeur de couches qui va rajouter
# des couches pour le transfer learning sur le modèle de base, et renvoyer le modèle combiné.
def build_tl_model(base_model, top_layers_builder, image_size, nb_classes):
    inputs = Input(shape=(*image_size, 3))
    outputs = top_layers_builder(base_model(inputs), nb_classes)
    return Model(inputs, outputs)

# Gèle les n premières couches, et dégèle les autres.
def freeze_n_layers(model, n):
    for layer in model.layers:
        layer.trainable = False
    for layer in model.layers[n:]:
        layer.trainable = True

# Compile et entraîne le modèle 'model' à partir des générateur 'train_gen' et 'val_gen'
def train_model(model, train_gen, val_gen, epochs, opt, callbacks=None):
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model.fit_generator(train_gen,
                               validation_data=val_gen,
                               steps_per_epoch=np.size(train_gen.classes) // train_gen.batch_size,
                               validation_steps=np.size(val_gen.classes) // val_gen.batch_size,
                               epochs=epochs,
                               verbose=1,
                               callbacks=callbacks)

# Prend en argument l'histoire d'entraînement renvoyée par train_model et
# trace la précision et la perte au fil des epochs (en train et en val)
def plot_acc_loss(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Précision en apprentissage et validation')
    plt.xlabel('Epoch')
    plt.legend(['Acc Train', 'Acc Val'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Perte en apprentissage et validation')
    plt.xlabel('Epoch')
    plt.legend(['Loss Train', 'Loss Val'], loc='upper left')
    plt.show()
