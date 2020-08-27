This my implementation of convolutional neural networks using only numpy.
Each layer/module has a forward operation and backward operation. The default
loss function is categorical cross entropy because I wanted to utilize this
neural network and train it on a classification task. Therefore, all gradients are computed
with respect to the categorical cross entropy loss function as of right now.

# How to use #

Define object layers as a list:

model = [convolution(num_filters=10,kernel_size=3,padding=1,stride=1,bias=True), relu(), max_pool(size=3), convolution(num_filters=10,kernel_size=3,padding=1,stride=1,bias=True),relu(), max_pool(size=3), fc(input_dim=90, output_dim=10,bias=True)]

convnet = ConvNet(model)

Forward propagation:
prediction = convnet.forward(data)

Backpropagation:
loss, correct, softmax = cross_entropy(prediction, onehot_labels)

gradient = convnet.backprop(learing_rate, softmax)

# Testing #
For testing my implementation of a convolutional neural network and its end-to-end training, I wrote unit tests and compared them to Pytorch outputs. By comparing my outputs and gradients to the Pytorch's outputs and gradients, I also gained a better a sense of how gradients flow in Pytorch and how to debug a Pytorch model.

You can run the tests via python test.py


# Smiles Detection

I wanted to display my results with a more user interactive method, so I decided to create and train the network on
detecting smiles.

Smiling dataset was parsed from Getty Images. Then images were cropped to only be faces
using the face_recognition library that was built using dlib's state of the art face recognition.

At the end of training, my model was getting 86% accuracy. You can try the smile detection model at ____
The code for the UI is [here.](https://github.com/ethsu1/smileAI)


![alt text](https://github.com/ethsu1/convnet_numpy/blob/master/results/train_loss.png)
![alt text](https://github.com/ethsu1/convnet_numpy/blob/master/results/train_accuracy.png)





