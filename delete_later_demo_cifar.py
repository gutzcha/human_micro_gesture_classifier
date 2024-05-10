# import the relevant modules
from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.utils import multi_gpu_model

# load the mnist data set
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# save the input shape for later
input_shape = x_train.shape


# visualize some pictures from the downloaded dataset for each class
import matplotlib.pyplot as plt
plt.switch_backend('agg')
plt.rcParams['figure.figsize'] = [8, 8]


# convert it to float and bring it into a range between 0. and 1.0
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# Create the model
model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3),
                 activation='relu',
                 padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Conv2D(64, kernel_size=(3, 3),
                 activation='relu',
                 padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3),
                 activation='relu',
                 padding='same'))
model.add(Conv2D(128, kernel_size=(3, 3),
                 activation='relu',
                 padding='same'))
model.add(Conv2D(128, kernel_size=(3, 3),
                 activation='relu',
                 padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, kernel_size=(3, 3),
                 activation='relu',
                 padding='same'))
model.add(Conv2D(256, kernel_size=(3, 3),
                 activation='relu',
                 padding='same'))
model.add(Conv2D(256, kernel_size=(3, 3),
                 activation='relu',
                 padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(2048, activation='relu'))
model.add(Dense(2048, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Enable multiple GPU usage
model = multi_gpu_model(model, gpus=2)

# Select optimizer
opt = keras.optimizers.Adam()

# Compile the model
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=opt,
              metrics=['accuracy'])

# Train model
history = model.fit(x_train, y_train,
                    batch_size=512,
                    epochs=15,
                    verbose=1,
                    validation_data=(x_test, y_test))

# Evaluate model loss and accuracy
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Plot loss curve
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('cifar_cnn_loss.png')

# Plot accuracy curve
plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('cifar_cnn_accuracy.png')
