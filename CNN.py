#data 
from keras.datasets import fashion_mnist
from keras.utils import np_utils


((trainX, trainY), (testX, testY)) = fashion_mnist.load_data()

print("Train set Images-> Main Shape :", trainX.shape)
print("Train set Images-> dimension :", trainX.ndim)
print("Train set Images-> data type :", trainX.dtype)

print("Train set labels-> Main Shape :", trainY.shape)
print("Train set labels-> dimension :", trainY.ndim)
print("Train set labels-> data type :", trainY.dtype)


#resim = trainX[1124]
#plt.imshow(resim, cmap='gray') #'binary' barakse color mape gray hastesh

X_train = trainX.reshape(60000,28,28,1)
X_test = testX.reshape(10000,28,28,1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255 
X_test /= 255

Y_train = np_utils.to_categorical(trainY)
Y_test = np_utils.to_categorical(testY)

#------------------------------------------------------------------------ 
from keras.models import Model
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Flatten, Input, Dense



myInput = Input(shape=(28,28,1))
conv1 = Conv2D(16,3, activation ='relu', padding='same')(myInput)
pool1 = MaxPooling2D(pool_size=2)(conv1)
conv2 = Conv2D(32,3, activation ='relu', padding='same')(pool1)
pool2 = MaxPooling2D(pool_size=2)(conv2)
flat = Flatten()(pool2)
out_layer= Dense(10,activation='softmax')(flat)


myModel = Model(myInput,out_layer)
myModel.summary()
#--------------------------------------------------------------------------
#callbacks and plot model

from keras.utils import plot_model
from keras import callbacks 
import keras



plot_model(myModel, to_file='rayan_cnn.png', show_shapes= True )

logger = callbacks.CSVLogger('trainingCnn.log')


xx = keras.callbacks.TensorBoard(log_dir="/home/rahem/SpyderProjects/CNN")


#=======================================================================
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy

myModel.compile(optimizer=Adam(), loss=categorical_crossentropy, metrics=['accuracy'])

networkHistory=myModel.fit(X_train, Y_train, batch_size=128, epochs=400,callbacks=[logger,xx])


T_LOSS, T_ACC = myModel.evaluate(X_test, Y_test)

test_labels_p = myModel.predict(X_test)
import numpy as np
test_labels_p = np.argmax(test_labels_p, axis=1)


