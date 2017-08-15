# author: ZHANG wentong
# date: 2017.05.08
# email: wentong.zhang@groupe-esigelec.org
# code for building and training the model by existing photos

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D

import numpy as np
import os

from skimage import io
from sklearn.cross_validation import train_test_split


# at first we load the path of data base
DatasetPath = []
for i in os.listdir('./CNNdata'):
    DatasetPath.append(os.path.join('./CNNdata', i))

imageData = []
imageLabels = []


# then load all photos from the data base
# save the photos and labels
for i in DatasetPath:
    imgRead = io.imread(i,as_grey=True)
    imageData.append(imgRead)
    
    labelRead = int(os.path.split(i)[1].split("_")[0]) - 1
    imageLabels.append(labelRead)

# split randomly the photos into 2 parts, 
# 90% for training, 10% for testing
X_train, X_test, y_train, y_test = train_test_split(np.array(imageData),np.array(imageLabels), train_size=0.9, random_state = 4)

X_train = np.array(X_train)
X_test = np.array(X_test)

y_train = np.array(y_train) 
y_test = np.array(y_test)

# nb_classes is how many people for this model
nb_classes = 4
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# for tensorflow backend, it's (nb_of_photo, size, size, channel)
# for theanos backend, it's (channel, nb_of_photo, size, size)
# we are using tensorflow backend, so take first one (1500*0.1/0.9, 46, 46, 1)
X_train = X_train.reshape(X_train.shape[0], 46, 46, 1)
X_test = X_test.reshape(X_test.shape[0], 46, 46, 1)

# input_shape is for the first layer of model.
# 46, 46, 1 means size 46*46 pixels, 1 channel(because of read as gray,not RGB)
input_shape = (46, 46, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

# then we start the build of model
model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# then we compile this model
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

# and training
model.fit(X_train, Y_train, batch_size=32, epochs=20,
                 verbose=1, validation_data=(X_test, Y_test))

# when the training finishes, we need to save the trained model.
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# and use the 10% data as we have already splited to test the new model
scores = model.evaluate(X_test, Y_test, verbose=0)
print scores
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


