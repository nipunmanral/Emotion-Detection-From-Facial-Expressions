import os
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image
import glob
import time
import h5py
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, SGD
from keras.utils import plot_model
from keras.regularizers import l2
from keras import regularizers
from keras.utils import Sequence
import collections
from sklearn.utils import class_weight as cw

# codePath = os.path.dirname(__file__) + '/train_image/'
# output_labels_path = os.path.dirname(__file__) + '/train.csv'
codePath = './train_image/'
output_labels_path = './train.csv'

def process_image(imagefile):
    im = Image.open(imagefile)
    im = im.convert(mode='L')
    im = im.resize((350,350))
    im = np.asarray(im, dtype=float).reshape(350, 350, 1)
    # Minmax normalization
    im_minmax = im/np.float(255)
    # GCN
    im_gcn = (im_minmax - np.mean(im_minmax))/np.std(im_minmax)
    return im_gcn

class BatchSampler(Sequence):
    def __init__(self, data_dir, label_file, batch_size):
        self.batch_size = batch_size
        self.image_files = glob.glob(data_dir + '*.jpg')
        self.labels_dict = {}
        self.label2id = {}
        self.class_weights = {}

        lines = open(label_file).read().splitlines()
        labels = []
        for line in lines:
            name, label = line.split(',')
            self.labels_dict[name] = label
            labels.append(label)

        count = 0
        for k in sorted(set(labels)):
            self.label2id[k] = count
            count += 1

        all_labels_as_integer = [self.label2id.get(n, n) for n in labels]
        self.class_weights = cw.compute_class_weight('balanced', np.unique(all_labels_as_integer), all_labels_as_integer)


    def __len__(self):
        return int(np.ceil(len(self.image_files)/float(self.batch_size)))


    def __getitem__(self, idx):
        batch_x = self.image_files[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = []
        for file in batch_x:
            file_name = file.split('/')[-1]
            label = self.labels_dict[file_name]
            batch_y.append(self.label2id[label])
        return np.array([process_image(x) for x in batch_x]), np.array(batch_y)


    def sample_count_of_all_labels(self):
        counter = collections.Counter(self.labels_dict.values())
        return counter.keys(), counter.values()


target_image_shape = (350, 350)

# def Preprocess_Image(codePath):
#
#     print("Pre processing images (Resizing and converting to black & white)")
#     start = time.time()
#     for path_image in glob.glob(codePath + '*.jpg'):
#         img = mpimg.imread(path_image)
#         # Convert from RGB image to black & white image
#         if img.ndim != 2:
#             image_pillow = Image.open(path_image)
#             image_pillow = image_pillow.convert(mode='L')
#             # plt.imshow(np.asarray(image_pillow), cmap='gray')
#             image_pillow.save(path_image, format='JPEG')
#         if img.shape != target_image_shape:
#             image_pillow = image_pillow.resize(target_image_shape)
#             image_pillow.save(path_image, format='JPEG')
#     end = time.time()
#     print("Pre processing of images completed in {} seconds".format(end-start))
#     print("-----------------------------------------------")


# Uncomment below function calls to pre-process and create data-sets
# Preprocess_Image(codePath)


# (3) Create a sequential model
model = Sequential()

# 1st Convolutional Layer
model.add(Conv2D(filters=96, input_shape=(350,350,1), kernel_size=(11,11), strides=(4,4), padding='valid'))
model.add(Activation('relu'))
# Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation before passing it to the next layer
model.add(BatchNormalization())

# 2nd Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation
model.add(BatchNormalization())

# 3rd Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Batch Normalisation
model.add(BatchNormalization())

# 4th Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation
model.add(BatchNormalization())

# 5th Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation
model.add(BatchNormalization())

# Passing it to a dense layer
model.add(Flatten())
# 1st Dense Layer
# model.add(Dense(4096, input_shape=(224*224*3,)))
model.add(Dense(1024))
model.add(Activation('relu'))
# Add Dropout to prevent overfitting
model.add(Dropout(0.4))
# Batch Normalisation
model.add(BatchNormalization())

# 2nd Dense Layer
model.add(Dense(4096))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.4))
# Batch Normalisation
model.add(BatchNormalization())

# 3rd Dense Layer
model.add(Dense(1000))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.4))
# Batch Normalisation
model.add(BatchNormalization())

# Output Layer
model.add(Dense(8))
model.add(Activation('softmax'))

print(model.summary())
# plot_model(model, to_file='emotion_model.png', show_shapes=True, show_layer_names=False)

# Train and test model
obj_batch_sampler = BatchSampler(codePath, output_labels_path, 32)
print("Number of training images = ", len(obj_batch_sampler.image_files))
print(obj_batch_sampler.label2id)
print("Number of batches = ", len(obj_batch_sampler))
label_type, label_count = obj_batch_sampler.sample_count_of_all_labels()
print(label_type)
print(label_count)
print(obj_batch_sampler.class_weights)

optimizer = Adam()
# optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# If targets are one-hot encoded, use categorical_crossentropy. If targets are integers,
# use sparse_categorical_crossentropy.
loss = 'sparse_categorical_crossentropy'
metrics= ['sparse_categorical_accuracy']
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
model.fit_generator(obj_batch_sampler, steps_per_epoch=len(obj_batch_sampler), epochs=10, verbose=1,
                    shuffle=True, class_weight=obj_batch_sampler.class_weights)

# score = model.evaluate(xte,yte, batch_size=batch_size)
# print('Test accuracy = {0}'.format(100*score[1]))
model.save('emotion_model.h5')
