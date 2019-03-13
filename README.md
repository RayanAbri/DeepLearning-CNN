# Deep Learning - Convolutional Neural Network (CNN) Model

Convolutional Neural Networks are very similar to ordinary Neural Networks. they are made up of neurons that have learnable weights and biases. Each neuron receives some inputs, performs a dot product and optionally follows it with a non-linearity. 

Classification example of mnist Fashion dataset.


The Fashion dataset includes:

#60,000 training examples
#10,000 testing examples
#10 classes
#28×28 grayscale/single channel images
#The ten fashion class labels include:
#
#1     T-shirt/top  
#2     Trouser/pants
#3     Pullover shirt
#4     Dress
#5     Coat
#6     Sandal
#7     Shirt
#8     Sneaker
#9     Bag
#10    Ankle boot

from keras.datasets import fashion_mnist:
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

Returns:
2 tuples:
x_train, x_test: uint8 array of grayscale image data with shape (num_samples, 28, 28).
y_train, y_test: uint8 array of labels (integers in range 0-9) with shape (num_samples,).
