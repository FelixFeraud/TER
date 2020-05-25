import json
import shutil

from func_utils import makedirs_if

data_path = 'drive/My Drive/TER/'

categories = range(0, 200)

makedirs_if(data_path + 'train/')
makedirs_if(data_path + 'val/')

for category in categories:
    makedirs_if(data_path + 'train/' + str(category))
    makedirs_if(data_path + 'val/' + str(category))

with open(data_path + 'annotation/anno_val.json', 'r') as f:
    val_data = json.load(f)

with open(data_path + 'annotation/anno_l_train.json', 'r') as f:
    train_data = json.load(f)

for img in val_data['images']:
    shutil.copy(data_path + img['file_name'], data_path + 'val/' + (img['file_name'])[15:])

print("\nValidation done.\n")

for img in train_data['images']:
    shutil.copy(data_path + img['file_name'], data_path + 'train/' + (img['file_name'])[15:])

print("\nTraining done.\n")
