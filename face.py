from convnet import *
from max_pool import *
from avg_pool import *
from layers import *
from fc import *
from loss_func import *
from relu import *
from flat import *
from matplotlib import pyplot as plt
from data_parse import *
import math
EPOCH = 20
BATCH_SIZE = 32
LR = 0.001
model = [convolution(num_filters=25, kernel_size=3,padding=1,stride=1),
relu(),
max_pool(size=3),
convolution(num_filters=50, kernel_size=3,padding=1,stride=1),
relu(),
max_pool(size=3),
flat(),
fc(input_dim=50*11*11, output_dim=500),
relu(),
fc(input_dim=500, output_dim=2)
]
data, labels, onehot_labels = load_data()
convnet = ConvNet(model)
convnet.forward(data[0:1])

epoch_list = []
loss_list = []
acc_list = []
batches = math.ceil(len(data)/BATCH_SIZE)
for epoch in range(EPOCH):
	running_loss = 0
	num_correct = 0
	permutation = np.random.permutation(len(labels))
	data = data[permutation]
	labels = labels[permutation]
	onehot_labels = onehot_labels[permutation]
	for j in range(batches):
		begin = j*BATCH_SIZE
		end = min(begin + BATCH_SIZE, data.shape[0])
		batch_data = data[begin:end]
		batch_labels = labels[begin:end]
		batch_labels_onehot = onehot_labels[begin:end]
		output = model.forward(batch_data)
		loss, correct, softmax = cross_entropy(output, batch_labels)
		c.backward(softmax, batch_labels_onehot, LR)


	loss_list.append(running_loss)
	acc_list.append(num_correct/len(labels))
	print("Accuracy: {}".format(num_correct/len(labels))) 
	print("Epoch: {} Loss: {}".format(epoch, running_loss))

fig1 = plt.figure()
plt.plot(loss_list, label="Train loss")
plt.show()
fig2 = plt.figure()
plt.plot(acc_list, label="Train Accuracy")
plt.show()



convnet.save_model('face.pkl')
convnet = ConvNet()
convnet.load_model('face.pkl')



