import os

from keras_applications import resnet
from keras_preprocessing.image import load_img, img_to_array, np

from func_utils import makedirs_if

data_path = 'drive/My Drive/TER/'
n = 224

model = resnet.ResNet50(weights='imagenet', include_top=False, input_shape=(n, n, 3))

categories = range(200)

features_train_path = os.path.join(data_path, 'feat_train_' + str(n))
features_val_path = os.path.join(data_path, 'feat_val_' + str(n))

makedirs_if(features_train_path)
makedirs_if(features_val_path)

for category in categories:
    makedirs_if(features_train_path + '/' + str(category))
    makedirs_if(features_val_path + '/' + str(category))

# Extract from train :

for category in categories:
    print("- Processing category ", str(category), "(train)")
    cat_path = os.path.join(data_path, 'train_split', str(category))
    for img_name in os.listdir(cat_path):
        img_path = os.path.join(cat_path, img_name)
        target_path = os.path.join(features_train_path, str(category), img_name)

        img = load_img(img_path, target_size=(n, n))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = resnet.preprocess_input(img)

        prediction = model.predict(img)

        np.savez_compressed(target_path, prediction)

# Extract from val :

for category in categories:
    print("- Processing category ", str(category), "(val)")
    cat_path = os.path.join(data_path, 'val_split', str(category))
    for img_name in os.listdir(cat_path):
        img_path = os.path.join(cat_path, img_name)
        target_path = os.path.join(features_val_path, str(category), img_name)

        img = load_img(img_path, target_size=(n, n))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = resnet.preprocess_input(img)

        prediction = model.predict(img)

        np.savez_compressed(target_path, prediction)
