This my implementation of convolutional neural networks using only numpy.
Each layer/module has a forward operation and backward operation. The default
loss function is categorical cross entropy because I wanted to utilize this
neural network and train it on a classification task. Therefore, all gradients are computed
with respect to the categorical cross entropy loss function as of right now.

[How to use:]

Define object layers as a list:

model = [convolution(num_filters=10,kernel_size=3,padding=1,stride=1,bias=True), relu(), max_pool(size=3), convolution(num_filters=10,kernel_size=3,padding=1,stride=1,bias=True),relu(), max_pool(size=3), fc(input_dim=90, output_dim=10,bias=True)]

convnet = ConvNet(model)

Forward propagation:
prediction = convnet.forward(data)

Backpropagation:
loss, correct, softmax = cross_entropy(prediction, onehot_labels)

gradient = convnet.backprop(lr, softmax)

I wrote the neural network using numpy and tested my implementations against Pytorch to ensure my layers were
doing the right operations. From this project, I gained a much deeper understanding of neural networks, specifically convolutional
networks.


I wanted to display my results with a more user interactive method, so I decided to create and train the network on
detecting smiles.

Smiling dataset was parsed from Getty Images. Then images were cropped to only be faces
using the face_recognition library that was built using dlib's state of the art face recognition.


For testing my implementation of a convolutional neural network and its end-to-end training, I wrote unit tests and compared them to Pytorch outputs. By comparing my outputs and gradients to the Pytorch's outputs and gradients, I also gained a better a sense of how gradients flow in Pytorch and how to debug a Pytorch model.

