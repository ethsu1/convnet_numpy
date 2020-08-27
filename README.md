This my implementation of convolutional neural networks using only numpy.
Each layer/module has a forward operation and backward operation. The default
loss function is categorical cross entropy because I wanted to utilize this
neural network and train it on a classification task. Therefore, all gradients are computed
with respect to the categorical cross entropy loss function as of right now.

Smiling dataset was parsed from Getty Images. Then images were cropped to only be faces
using the face_recognition library that was built using dlib's state of the art face recognition

How to use:

Define object layers as a list:
model = [convolution(num_filters=10,kernel_size=3,padding=1,stride=1,bias=True), relu(), max_pool(size=3), convolution(num_filters=10,kernel_size=3,padding=1,stride=1,bias=True),relu(), max_pool(size=3), fc(input_dim=90, output_dim=10,bias=True)]

convnet = ConvNet(model)

Forward propagation:
prediction = convnet.forward(data)

Backpropagation:
loss, correct, softmax = cross_entropy(prediction, onehot_labels)

gradient = convnet.backprop(lr, softmax)
