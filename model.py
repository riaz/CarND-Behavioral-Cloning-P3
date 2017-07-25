from utils import Utils
import tensorflow as tf
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.layers import Dense,Activation, MaxPooling2D, Flatten, Lambda
from keras.layers.convolutional import Convolution2D
import matplotlib.pyplot as plt

# Required by keras to use the tensorflow backend
tf.python.control_flow_ops = tf

learning_rate = 1e-4
nb_epochs = 10
nb_samples_per_epoch = 6000
nb_val_samples = 2037

api = Utils()

# Using the model based on the NVIDIA paper titled "End to End Learning for Self-Driving Cars"

model = Sequential()

# Normalizing
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(64, 64, 3)))

# Adding 5 convolution layers and max-pool layers

# 3 convolutional layers with 2x2 strides
model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

# 2 convolutional layers with 1x1 strides
model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Flatten())

# Five FC layers leading to an output control value, which is the inverse turning radius
model.add(Dense(1164))
model.add(Activation('relu'))

model.add(Dense(100))
model.add(Activation('relu'))

model.add(Dense(50))
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Activation('relu'))

model.add(Dense(1))

model.summary()

model.compile(optimizer=Adam(learning_rate), loss="mse")

train_data =  api.next_batch()
val_data = api.next_batch()



history_object = model.fit_generator(train_data,
                     samples_per_epoch=nb_samples_per_epoch,
                     nb_epoch=nb_epochs,
                     validation_data=val_data,
                     nb_val_samples=nb_val_samples,
                     verbose=1)

# finally save our model and weights
model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()


"""
docker run -it --rm -p 4567:4567 -v `pwd`:/src udacity/carnd-term1-starter-kit python drive.py model.h5
"""
