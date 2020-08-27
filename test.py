from convolution import *
import unittest
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.image import imread
import scipy
import torch
import torch.nn.functional as F
from max_pool import *
from avg_pool import *
from layers import *
from fc import *
from loss_func import *
from relu import *
from mnist import MNIST
from torchvision import datasets, transforms
from convnet import *
from cifar10_web import *
from tensorflow import keras
from flat import *

class testConvolution(unittest.TestCase):
	def test_convolution1(self):
		'''
		testing with sample operations from stanford's cs231n website
		'''
		red = [[0,1,1,2,1],[1,1,1,1,2],[1,2,0,0,0],[2,0,0,2,1],[1,0,1,2,1]]
		green = [[2,1,2,2,0],[0,1,0,2,0],[2,2,0,2,2],[0,1,1,1,2],[0,1,1,0,1]]
		blue = [[1,1,2,1,1],[0,2,1,2,0],[2,1,0,1,0],[1,1,2,1,0],[2,2,0,0,2]]
		img = np.asarray([[red,green,blue]], dtype=np.float64)
		conv = convolution(2,3,1,2)
		kernel1 = [[[0,-1,1],[1,-1,-1],[0,1,1]],[[1,-1,-1],[-1,-1,1],[0,-1,-1]],[[0,0,0],[0,-1,1],[1,-1,0]]]
		kernel2 = [[[1,0,0],[1,1,-1],[0,-1,0]],[[-1,1,1],[1,1,1],[1,-1,-1]],[[-1,1,0],[0,1,1],[-1,1,1]]]
		conv.filters = np.asarray([kernel1, kernel2], dtype=np.float64)
		conv.bias = np.asarray([[1],[0]], dtype=np.float64)
		output = conv.forward(img)
		expected_output = np.asarray([[[[0,-2,3],[-4,2,-3],[-2,-3,-3]],[[4,7,4],[6,9,-2],[8,3,8]]]],dtype=np.float64)
		np.testing.assert_almost_equal(output, expected_output)

	def test_convolution_identity(self):
		'''
		testing convolutional operation with an identity kernel and comparing against pytorch output
		'''
		data = imread('./testimages/image.png')
		data = data[:,:,:3]
		og = data
		data = np.asarray([data], dtype=np.float64)
		data = np.moveaxis(data, 3,1)
		conv = convolution(3,3,1,1)
		filter1 = [[[0,0,0],[0,1,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]], [[0,0,0],[0,0,0],[0,0,0]]]
		filter2 = [[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,1,0],[0,0,0]], [[0,0,0],[0,0,0],[0,0,0]]]
		filter3 = [[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]], [[0,0,0],[0,1,0],[0,0,0]]]
		conv.filters = np.asarray([filter1,filter2,filter3], dtype=np.float64)
		conv.bias = np.zeros(3)
		output = conv.forward(data)
		#move axis so that it becomes height x width x channel (needed to display image)
		output = np.moveaxis(output, 1,3)
		output = output[0]
		
		x = torch.from_numpy(data)
		expected_conv = torch.nn.Conv2d(in_channels=3,out_channels=3, kernel_size=3,stride=1,padding=1,bias=True)
		expected_conv.weight = torch.nn.Parameter(torch.from_numpy(conv.filters))
		expected_conv.bias = torch.nn.Parameter(torch.from_numpy(conv.bias))
		expected_output = expected_conv(x).detach().numpy()
		#move axis so that it becomes height x width x channel (needed to display image)
		expected_output = np.moveaxis(expected_output,1,3)
		expected_output = expected_output[0]
		
		'''
		fig = plt.figure()
		fig.suptitle("Identity")
		fig.add_subplot(1,3,1)
		plt.imshow(og, interpolation='nearest')
		fig.add_subplot(1,3,2)
		plt.imshow(output, interpolation='nearest')
		fig.add_subplot(1,3,3)
		plt.imshow(expected_output,interpolation='nearest')
		plt.show(block=True)'''
		np.testing.assert_allclose(output, expected_output)

	def test_convolution_sharpen(self):
		'''
		testing convolutional operation with an sharpen kernel and comparing against pytorch output
		'''
		data = imread('./testimages/image.png')
		data = data[:,:,:3]
		og = data
		data = np.asarray([data], dtype=np.float64)
		data = np.moveaxis(data,3, 1)
		conv = convolution(3,3,1,1)
		filter1 = [[[0,-1,0],[-1,5,-1],[0,-1,0]],[[0,0,0],[0,0,0],[0,0,0]], [[0,0,0],[0,0,0],[0,0,0]]]
		filter2 = [[[0,0,0],[0,0,0],[0,0,0]],[[0,-1,0],[-1,5,-1],[0,-1,0]], [[0,0,0],[0,0,0],[0,0,0]]]
		filter3 = [[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]], [[0,-1,0],[-1,5,-1],[0,-1,0]]]
		conv.filters = np.asarray([filter1,filter2,filter3], dtype=np.float64)
		#conv.filters = np.asarray([filter1])
		output = conv.forward(data)
		#move axis so that it becomes height x width x channel (needed to display image)
		output = np.moveaxis(output, 1, 3)
		output = output[0]
		
		x = torch.from_numpy(data)
		expected_conv = torch.nn.Conv2d(in_channels=3,out_channels=3, kernel_size=3,stride=1,padding=1,bias=True)
		expected_conv.weight = torch.nn.Parameter(torch.from_numpy(conv.filters))
		expected_conv.bias = torch.nn.Parameter(torch.from_numpy(conv.bias))
		expected_output = expected_conv(x).detach().numpy()

		#move axis so that it becomes height x width x channel (needed to display image)
		expected_output = np.moveaxis(expected_output,1,3)
		expected_output = expected_output[0]
		'''
		fig = plt.figure()
		fig.suptitle("Sharpen")
		fig.add_subplot(1,3,1)
		plt.imshow(og, interpolation='nearest')
		fig.add_subplot(1,3,2)
		plt.imshow(output, interpolation='nearest')
		fig.add_subplot(1,3,3)
		plt.imshow(expected_output,interpolation='nearest')
		#plt.show(block=True)'''
		np.testing.assert_allclose(output, expected_output)

	def test_convolutional_edge(self):
		'''
		testing convolutional operation with an edge kernel and comparing against pytorch output
		'''
		data = imread('./testimages/image.png')
		data = data[:,:,:3]
		og = data
		data = np.asarray([data], dtype=np.float64)
		data = np.moveaxis(data,3, 1)
		conv = convolution(3,3,1,1)
		filter1 = [[[1,0,-1],[0,0,0],[-1,0,1]],[[0,0,0],[0,0,0],[0,0,0]], [[0,0,0],[0,0,0],[0,0,0]]]
		filter2 = [[[0,0,0],[0,0,0],[0,0,0]],[[1,0,-1],[0,0,0],[1,0,-1]], [[0,0,0],[0,0,0],[0,0,0]]]
		filter3 = [[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]], [[1,0,-1],[0,0,0],[-1,0,1]]]
		conv.filters = np.asarray([filter1,filter2,filter3], dtype=np.float64)
		output = conv.forward(data)
		#move axis so that it becomes height x width x channel (needed to display image)
		output = np.moveaxis(output, 1,3)
		output = output[0]
		x = torch.from_numpy(data)
		expected_conv = torch.nn.Conv2d(in_channels=3,out_channels=3, kernel_size=3,stride=1,padding=1,bias=True)
		expected_conv.weight = torch.nn.Parameter(torch.from_numpy(conv.filters))
		expected_conv.bias = torch.nn.Parameter(torch.from_numpy(conv.bias))
		expected_output = expected_conv(x).detach().numpy()
		#move axis so that it becomes height x width x channel (needed to display image)
		expected_output = np.moveaxis(expected_output,1,3)
		expected_output = expected_output[0]
		'''
		fig = plt.figure()
		fig.suptitle("Edge")
		fig.add_subplot(1,3,1)
		plt.imshow(og, interpolation='nearest')
		fig.add_subplot(1,3,2)
		plt.imshow(output, interpolation='nearest')
		fig.add_subplot(1,3,3)
		plt.imshow(expected_output,interpolation='nearest')
		#plt.show(block=True)'''
		np.testing.assert_allclose(output, expected_output)


	def test_convolutional_blur(self):
		'''
		testing convolutional operation with a blur kernel and comparing against pytorch output
		'''
		data = imread('./testimages/image.png')
		data = data[:,:,:3]
		og = data
		data = np.asarray([data], dtype=np.float64)
		data = np.moveaxis(data,3, 1)
		conv = convolution(3,3,1,1)
		filter1 = [[[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]],[[0,0,0],[0,0,0],[0,0,0]], [[0,0,0],[0,0,0],[0,0,0]]]
		filter2 = [[[0,0,0],[0,0,0],[0,0,0]],[[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]], [[0,0,0],[0,0,0],[0,0,0]]]
		filter3 = [[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]], [[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]]]
		conv.filters = np.asarray([filter1,filter2,filter3], dtype=np.float64)
		output = conv.forward(data)
		#move axis so that it becomes height x width x channel (needed to display image)
		output = np.moveaxis(output, 1,3)
		output = output[0]
		x = torch.from_numpy(data)
		expected_conv = torch.nn.Conv2d(in_channels=3,out_channels=3, kernel_size=3,stride=1,padding=1,bias=True)
		expected_conv.weight = torch.nn.Parameter(torch.from_numpy(conv.filters))
		expected_conv.bias = torch.nn.Parameter(torch.from_numpy(conv.bias))
		expected_output = expected_conv(x).detach().numpy()
		#move axis so that it becomes height x width x channel (needed to display image)
		expected_output = np.moveaxis(expected_output,1,3)
		expected_output = expected_output[0]
		'''
		fig = plt.figure()
		fig.suptitle("Blur")
		fig.add_subplot(1,3,1)
		plt.imshow(og, interpolation='nearest')
		fig.add_subplot(1,3,2)
		plt.imshow(output, interpolation='nearest')
		fig.add_subplot(1,3,3)
		plt.imshow(expected_output,interpolation='nearest')
		#plt.show(block=True)'''
		np.testing.assert_allclose(output, expected_output)

	def test_convolution_no_padding(self):
		'''
		testing convolutional operation with a random kernel with no padding and comparing against pytorch output (output image
		will be of a smaller dimension than original image)
		'''
		data = imread('./testimages/image.png')
		data = data[:,:,:3]
		og = data
		data = np.asarray([data], dtype=np.float64)
		data = np.moveaxis(data,3, 1)
		conv = convolution(3,3,0,1)
		output = conv.forward(data)
		#move axis so that it becomes height x width x channel (needed to display image)
		output = np.moveaxis(output, 1,3)
		output = output[0]
		x = torch.from_numpy(data)
		expected_conv = torch.nn.Conv2d(in_channels=3,out_channels=3, kernel_size=3,stride=1,padding=0,bias=True)
		expected_conv.weight = torch.nn.Parameter(torch.from_numpy(conv.filters))
		expected_conv.bias = torch.nn.Parameter(torch.from_numpy(conv.bias))
		expected_output = expected_conv(x).detach().numpy()
		#move axis so that it becomes height x width x channel (needed to display image)
		expected_output = np.moveaxis(expected_output,1,3)
		expected_output = expected_output[0]
		'''
		fig = plt.figure()
		fig.suptitle("No Padding Random")
		fig.add_subplot(1,3,1)
		plt.imshow(og, interpolation='nearest')
		fig.add_subplot(1,3,2)
		plt.imshow(output, interpolation='nearest')
		fig.add_subplot(1,3,3)
		plt.imshow(expected_output,interpolation='nearest')
		#plt.show(block=True)'''
		np.testing.assert_allclose(output, expected_output)

	def test_convolution_extra_padding(self):
		'''
		testing convolutional operation with a random kernel with extra padding and comparing against pytorch output (output image
		will be of alarger dimension than input image)
		'''
		data = imread('./testimages/image.png')
		data = data[:,:,:3]
		og = data
		data = np.asarray([data], dtype=np.float64)
		data = np.moveaxis(data,3, 1)
		conv = convolution(3,3,3,2)
		output = conv.forward(data)
		#move axis so that it becomes height x width x channel (needed to display image)
		output = np.moveaxis(output, 1,3)
		output = output[0]
		x = torch.from_numpy(data)
		expected_conv = torch.nn.Conv2d(in_channels=3,out_channels=3, kernel_size=3,stride=2,padding=3,bias=True)
		expected_conv.weight = torch.nn.Parameter(torch.from_numpy(conv.filters))
		expected_conv.bias = torch.nn.Parameter(torch.from_numpy(conv.bias))
		expected_output = expected_conv(x).detach().numpy()
		#move axis so that it becomes height x width x channel (needed to display image)
		expected_output = np.moveaxis(expected_output,1,3)
		expected_output = expected_output[0]
		'''
		fig = plt.figure()
		fig.suptitle("Extra Padding Random")
		fig.add_subplot(1,3,1)
		plt.imshow(og, interpolation='nearest')
		fig.add_subplot(1,3,2)
		plt.imshow(output, interpolation='nearest')
		fig.add_subplot(1,3,3)
		plt.imshow(expected_output,interpolation='nearest')
		#plt.show(block=True)'''
		np.testing.assert_allclose(output, expected_output)

	def test_convolution_multiple_stride(self):
		'''
		testing convolutional operation with a random kernel with larger stride and comparing against pytorch output (output image
		will be of a smaller dimension than input image)
		'''
		data = imread('./testimages/image.png')
		data = data[:,:,:3]
		og = data
		data = np.asarray([data], dtype=np.float64)
		data = np.moveaxis(data,3, 1)
		conv = convolution(3,3,1,3)
		output = conv.forward(data)
		#move axis so that it becomes height x width x channel (needed to display image)
		output = np.moveaxis(output, 1, 3)
		output = output[0]
		x = torch.from_numpy(data)
		expected_conv = torch.nn.Conv2d(in_channels=3,out_channels=3, kernel_size=3,stride=3,padding=1,bias=True)
		expected_conv.weight = torch.nn.Parameter(torch.from_numpy(conv.filters))
		expected_conv.bias = torch.nn.Parameter(torch.from_numpy(conv.bias))
		expected_output = expected_conv(x).detach().numpy()
		#move axis so that it becomes height x width x channel (needed to display image)
		expected_output = np.moveaxis(expected_output,1,3)
		expected_output = expected_output[0]
		'''
		fig = plt.figure()
		fig.suptitle("Multiple Stride Random")
		fig.add_subplot(1,3,1)
		plt.imshow(og, interpolation='nearest')
		fig.add_subplot(1,3,2)
		plt.imshow(output, interpolation='nearest')
		fig.add_subplot(1,3,3)
		plt.imshow(expected_output,interpolation='nearest')
		#plt.show(block=True)'''
		np.testing.assert_allclose(output, expected_output)

	def test_convolution_big_kernel(self):
		'''
		testing convolutional operation with a random kernel of size 5x5 and comparing against pytorch output
		'''
		data = imread('./testimages/image.png')
		data = data[:,:,:3]
		og = data
		data = np.asarray([data], dtype=np.float64)
		data = np.moveaxis(data,3, 1)
		conv = convolution(3,5,1,1)
		output = conv.forward(data)
		output = np.moveaxis(output, 1, 3)
		output = output[0]
		x = torch.from_numpy(data)
		expected_conv = torch.nn.Conv2d(in_channels=3,out_channels=3, kernel_size=5,stride=1,padding=1,bias=True)
		expected_conv.weight = torch.nn.Parameter(torch.from_numpy(conv.filters))
		expected_conv.bias = torch.nn.Parameter(torch.from_numpy(conv.bias))
		expected_output = expected_conv(x).detach().numpy()
		expected_output = np.moveaxis(expected_output,1,3)
		expected_output = expected_output[0]
		'''
		fig = plt.figure()
		fig.suptitle("5x5 Kernel")
		fig.add_subplot(1,3,1)
		plt.imshow(og, interpolation='nearest')
		fig.add_subplot(1,3,2)
		plt.imshow(output, interpolation='nearest')
		fig.add_subplot(1,3,3)
		plt.imshow(expected_output,interpolation='nearest')
		#plt.show(block=True)'''
		np.testing.assert_allclose(output, expected_output)

	def test_max_pool(self):
		'''
		Testing max pool operation and comparing against pytorch output
		'''
		data = imread('./testimages/image.png')
		data = data[:,:,:3]
		og = data
		data = np.asarray([data], dtype=np.float64)
		data = np.moveaxis(data,3, 1)
		conv = convolution(3,3,1,3)
		filter1 = [[[0,-1,0],[-1,5,-1],[0,-1,0]],[[0,0,0],[0,0,0],[0,0,0]], [[0,0,0],[0,0,0],[0,0,0]]]
		filter2 = [[[0,0,0],[0,0,0],[0,0,0]],[[0,-1,0],[-1,5,-1],[0,-1,0]], [[0,0,0],[0,0,0],[0,0,0]]]
		filter3 = [[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]], [[0,-1,0],[-1,5,-1],[0,-1,0]]]
		conv.filters = np.asarray([filter1,filter2,filter3], dtype=np.float64)
		output = conv.forward(data)
		
		pool_layer = max_pool(size=3)
		output = pool_layer.forward(output)
		#move axis so that it becomes height x width x channel (needed to display image)
		output = np.moveaxis(output, 1, 3)
		output = output[0]
		x = torch.from_numpy(data)
		expected_conv = torch.nn.Conv2d(in_channels=3,out_channels=3, kernel_size=3,stride=3,padding=1,bias=True)
		expected_conv.weight = torch.nn.Parameter(torch.from_numpy(conv.filters))
		expected_conv.bias = torch.nn.Parameter(torch.from_numpy(conv.bias))
		expected_output = expected_conv(x)
		m = torch.nn.MaxPool2d(3)
		expected_output = m(expected_output)
		expected_output = expected_output.detach().numpy()
		#move axis so that it becomes height x width x channel (needed to display image)
		expected_output = np.moveaxis(expected_output,1,3)
		expected_output = expected_output[0]
		'''
		fig = plt.figure()
		fig.suptitle("Max Pool")
		fig.add_subplot(1,3,1)
		plt.imshow(og, interpolation='nearest')
		fig.add_subplot(1,3,2)
		plt.imshow(output, interpolation='nearest')
		fig.add_subplot(1,3,3)
		plt.imshow(expected_output,interpolation='nearest')
		#plt.show(block=True)'''
		np.testing.assert_allclose(output, expected_output)

	def test_max_pool_pad(self):
		'''
		Testing max pool operation with padding and comparing against pytorch output
		'''
		data = imread('./testimages/image.png')
		data = data[:,:,:3]
		og = data
		data = np.asarray([data], dtype=np.float64)
		data = np.moveaxis(data,3, 1)
		conv = convolution(3,3,1,3)
		filter1 = [[[0,-1,0],[-1,5,-1],[0,-1,0]],[[0,0,0],[0,0,0],[0,0,0]], [[0,0,0],[0,0,0],[0,0,0]]]
		filter2 = [[[0,0,0],[0,0,0],[0,0,0]],[[0,-1,0],[-1,5,-1],[0,-1,0]], [[0,0,0],[0,0,0],[0,0,0]]]
		filter3 = [[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]], [[0,-1,0],[-1,5,-1],[0,-1,0]]]
		conv.filters = np.asarray([filter1,filter2,filter3], dtype=np.float64)
		output = conv.forward(data)
		
		pool_layer = max_pool(size=5, padding=2)
		output = pool_layer.forward(output)

		#move axis so that it becomes height x width x channel (needed to display image)
		output = np.moveaxis(output, 1, 3)
		output = output[0]
		x = torch.from_numpy(data)
		expected_conv = torch.nn.Conv2d(in_channels=3,out_channels=3, kernel_size=3,stride=3,padding=1,bias=True)
		expected_conv.weight = torch.nn.Parameter(torch.from_numpy(conv.filters))
		expected_conv.bias = torch.nn.Parameter(torch.from_numpy(conv.bias))
		expected_output = expected_conv(x)
		m = torch.nn.MaxPool2d(5, padding=2)
		expected_output = m(expected_output)
		expected_output = expected_output.detach().numpy()
		#move axis so that it becomes height x width x channel (needed to display image)
		expected_output = np.moveaxis(expected_output,1,3)
		expected_output = expected_output[0]
		'''
		fig = plt.figure()
		fig.suptitle("Max Pool Pad")
		fig.add_subplot(1,3,1)
		plt.imshow(og, interpolation='nearest')
		fig.add_subplot(1,3,2)
		plt.imshow(output, interpolation='nearest')
		fig.add_subplot(1,3,3)
		plt.imshow(expected_output,interpolation='nearest')
		#plt.show(block=True)'''
		np.testing.assert_allclose(output, expected_output, rtol=0.1, atol=0.1)

	def test_avg_pool(self):
		'''
		Testing avg pool operation and comparing against pytorch output
		'''
		data = imread('./testimages/image.png')
		data = data[:,:,:3]
		og = data
		data = np.asarray([data], dtype=np.float64)
		data = np.moveaxis(data,3, 1)
		conv = convolution(3,3,1,1)
		filter1 = [[[0,0,0],[0,1,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]], [[0,0,0],[0,0,0],[0,0,0]]]
		filter2 = [[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,1,0],[0,0,0]], [[0,0,0],[0,0,0],[0,0,0]]]
		filter3 = [[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]], [[0,0,0],[0,1,0],[0,0,0]]]
		conv.filters = np.asarray([filter1,filter2,filter3], dtype=np.float64)
		output = conv.forward(data)
		pool_layer = avg_pool(size=3)
		output = pool_layer.forward(output)
		#move axis so that it becomes height x width x channel (needed to display image)
		output = np.moveaxis(output, 1, 3)
		output = output[0]
		x = torch.from_numpy(data)
		expected_conv = torch.nn.Conv2d(in_channels=3,out_channels=3, kernel_size=3,stride=1,padding=1,bias=True)
		expected_conv.weight = torch.nn.Parameter(torch.from_numpy(conv.filters))
		expected_conv.bias = torch.nn.Parameter(torch.from_numpy(conv.bias))
		expected_output = expected_conv(x)
		m = torch.nn.AvgPool2d(3)
		expected_output = m(expected_output)
		expected_output = expected_output.detach().numpy()
		#move axis so that it becomes height x width x channel (needed to display image)
		expected_output = np.moveaxis(expected_output,1,3)
		expected_output = expected_output[0]
		'''
		fig = plt.figure()
		fig.suptitle("Average Pool")
		fig.add_subplot(1,3,1)
		plt.imshow(og, interpolation='nearest')
		fig.add_subplot(1,3,2)
		plt.imshow(output, interpolation='nearest')
		fig.add_subplot(1,3,3)
		plt.imshow(expected_output,interpolation='nearest')
		#plt.show(block=True)'''
		np.testing.assert_allclose(output, expected_output)


	def test_pool_pad(self):
		'''
		Testing that an error is raised for padding size being greater than kernel size
		'''
		try:
			pool_layer = max_pool(size=5, padding=4)
		except RuntimeError:
			pass
		else:
			raise ValueError("Should have raised an error for padding size being greater than half of kernel size")
		data = imread('./testimages/image.png')
		data = data[:,:,:3]
		data = np.asarray([data], dtype=np.float64)
		data = np.moveaxis(data,3, 1)
		filter1 = [[[0,0,0],[0,1,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]], [[0,0,0],[0,0,0],[0,0,0]]]
		filter2 = [[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,1,0],[0,0,0]], [[0,0,0],[0,0,0],[0,0,0]]]
		filter3 = [[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]], [[0,0,0],[0,1,0],[0,0,0]]]
		filters = np.asarray([filter1,filter2,filter3], dtype=np.float64)
		x = torch.from_numpy(data)
		expected_conv = torch.nn.Conv2d(in_channels=3,out_channels=3, kernel_size=3,stride=1,padding=1,bias=True)
		expected_conv.weight = torch.nn.Parameter(torch.from_numpy(filters))
		expected_output = expected_conv(x)
		m = torch.nn.AvgPool2d(5, padding=4)
		try:
			m = m(expected_output)
		except RuntimeError:
			pass
		else:
			raise ValueError("Pytorch should have raised an error for padding size being greater than half of kernel size")

	def test_relu(self):
		'''
		Testing relu activation operation
		'''
		data = imread('./testimages/image.png')
		data = data[:,:,:3]
		og = data
		data = np.asarray([data], dtype=np.float64)
		data = np.moveaxis(data, 3,1)
		conv = convolution(3,3,1,1)
		filter1 = [[[-1,0,1],[-0.1,0,1],[0,0.5,-2]],[[0,0,0],[0,0,0],[0,0,0]], [[0,0,0],[0,0,0],[0,0,0]]]
		filter2 = [[[0,0,0],[0,0,0],[0,0,0]],[[-1,0,1],[-0.1,0,1],[0,0.5,-2]], [[0,0,0],[0,0,0],[0,0,0]]]
		filter3 = [[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]], [[-1,0,1],[-0.1,0,1],[0,0.5,-2]]]
		conv.filters = np.asarray([filter1,filter2,filter3], dtype=np.float64)
		output = conv.forward(data)
		act_layer = relu()
		output = act_layer.forward(output)
		#move axis so that it becomes height x width x channel (needed to display image)
		output = np.moveaxis(output, 1,3)
		output = output[0]

	
		x = torch.from_numpy(data)
		expected_conv = torch.nn.Conv2d(in_channels=3,out_channels=3, kernel_size=3,stride=1,padding=1,bias=True)
		expected_conv.weight = torch.nn.Parameter(torch.from_numpy(conv.filters))
		expected_conv.bias = torch.nn.Parameter(torch.from_numpy(conv.bias))
		expected_output = expected_conv(x)
		relu_pytorch = torch.nn.ReLU()
		expected_output = relu_pytorch(expected_output).detach().numpy()
		#move axis so that it becomes height x width x channel (needed to display image)
		expected_output = np.moveaxis(expected_output,1,3)
		expected_output = expected_output[0]
		'''fig = plt.figure()
		fig.suptitle("Relu")
		fig.add_subplot(1,3,1)
		plt.imshow(og, interpolation='nearest')
		fig.add_subplot(1,3,2)
		plt.imshow(output, interpolation='nearest')
		fig.add_subplot(1,3,3)
		plt.imshow(expected_output,interpolation='nearest')
		plt.show(block=True)'''
		np.testing.assert_allclose(output, expected_output)

	def test_fully_connnected(self):
		'''
		test the fully connected layers output
		'''
		data = imread('./testimages/image.png')
		data = data[:,:,:3]
		og = data
		data = np.asarray([data], dtype=np.float64)
		data = np.moveaxis(data, 3,1)
		conv = convolution(3,3,1,1)
		output = conv.forward(data)
		#output is (1x3x300x200)
		flatten = flat()
		output = flatten.forward(output)
		fully_connected = fc(input_dim=3*300*200, output_dim=2)
		output = fully_connected.forward(output)
		x = torch.from_numpy(data)
		expected_conv = torch.nn.Conv2d(in_channels=3,out_channels=3, kernel_size=3,stride=1,padding=1,bias=True)
		expected_conv.weight = torch.nn.Parameter(torch.from_numpy(conv.filters))
		expected_conv.bias = torch.nn.Parameter(torch.from_numpy(conv.bias))
		expected_output = expected_conv(x)
		expected_output = expected_output.view(-1, 3*300*200)
		linear = torch.nn.Linear(in_features=3*300*200,out_features=2,bias=True)
		linear.weight = torch.nn.Parameter(torch.from_numpy(fully_connected.weights.T))
		linear.bias = torch.nn.Parameter(torch.from_numpy(fully_connected.bias))
		expected_output = linear(expected_output).detach().numpy()
	
		np.testing.assert_allclose(fully_connected.weights, linear.weight.detach().numpy().T)
		np.testing.assert_allclose(fully_connected.bias, linear.bias.detach().numpy())
		'''fig = plt.figure()
		fig.suptitle("Fully Connected")
		fig.add_subplot(1,3,1)
		plt.imshow(og, interpolation='nearest')
		fig.add_subplot(1,3,2)
		plt.imshow(output, interpolation='nearest')
		fig.add_subplot(1,3,3)
		plt.imshow(expected_output,interpolation='nearest')
		plt.show(block=True)'''
		np.testing.assert_allclose(output, expected_output)

	def test_fully_connected_backprop(self):
		'''
		testing the backpropagation of fully connected layer with MNIST dataset
		'''

		from torchvision import datasets, transforms

		train_set = datasets.MNIST('./mnist', train=True, download=True)
		test_set = datasets.MNIST('./mnist', train=False, download=True)

		train_set_array = train_set.data.numpy()
		train_set_array = np.expand_dims(train_set_array, axis=1)
		#print(train_set_array.shape)
		train_set_labels = train_set.targets.numpy()
		train_set_one_hot = np.zeros((train_set_labels.size, train_set_labels.max()+1))
		train_set_one_hot[np.arange(train_set_labels.size),train_set_labels] = 1
		labels = train_set_labels[15:100]
		onehot_labels = train_set_one_hot[15:100]
		data = train_set_array[15:100]

		conv = convolution(5,3,1,1)
		pool_layer = max_pool(size=9)
		flatten = flat()
		fc1= fc(input_dim=45, output_dim=45)
		fc2 = fc(input_dim=45, output_dim=10)
		bias1 = np.copy(fc1.bias.T)
		weights1 = np.copy(fc1.weights.T)
		bias2 = np.copy(fc2.bias)
		weights2 = np.copy(fc2.weights.T)

		ogweights1 = np.copy(fc1.weights)
		ogbias1 = np.copy(fc1.bias)
		ogweights2 = np.copy(fc2.weights)
		ogbias2 = np.copy(fc2.bias)


		output = conv.forward(data)
		ogconvfilters = np.copy(conv.filters)
		ogconvbias = np.copy(conv.bias) 
		output = pool_layer.forward(output)
		output = flatten.forward(output)
		output = fc1.forward(output)
		output = fc2.forward(output)
		loss,_, logsoftmax = cross_entropy(output, labels)

		layer = layers()
		#batch size x num_classes
		#derivative of cross entropy loss wrt fully connected output
		gradient = layer.gradient_logsoftmax(logsoftmax, onehot_labels)
		lr = 10
		gradient, grad_weights2, grad_bias2 = fc2.backprop(lr,gradient)
		gradient, grad_weights1, grad_bias1 = fc1.backprop(lr, gradient)


		#pytorch
		inputs = torch.from_numpy(data).double()
		class Net(torch.nn.Module):
			def __init__(self):
				super(Net, self).__init__()
				self.conv1 = torch.nn.Conv2d(in_channels=1,out_channels=5, kernel_size=3,stride=1,padding=1,bias=True)
				self.conv1.weight = torch.nn.Parameter(torch.from_numpy(ogconvfilters))
				self.conv1.bias =  torch.nn.Parameter(torch.from_numpy(ogconvbias))
				self.maxpool = torch.nn.MaxPool2d(9)
				self.linear1 = torch.nn.Linear(in_features=45,out_features=45,bias=True)
				self.linear1.weight = torch.nn.Parameter(torch.from_numpy(weights1))
				self.linear1.bias = torch.nn.Parameter(torch.from_numpy(bias1))
				self.linear2 = torch.nn.Linear(in_features=45,out_features=10,bias=True)
				self.linear2.weight = torch.nn.Parameter(torch.from_numpy(weights2))
				self.linear2.bias = torch.nn.Parameter(torch.from_numpy(bias2))
				
			def forward(self,x):
				x = self.conv1(x)
				x = self.maxpool(x)
				x = x.view(-1,45)
				x = self.linear1(x)
				x = self.linear2(x)
				return x
		net = Net()
		optimizer = torch.optim.SGD(net.parameters(), lr=lr)
		labels = torch.from_numpy(labels)
		expected_output = net(inputs)
		#maake pytorch output the loss for each training example
		criterion = torch.nn.CrossEntropyLoss()
		expected_loss = criterion(expected_output, labels)
		np.testing.assert_allclose(loss, expected_loss.detach().numpy())
		optimizer.zero_grad()
		expected_loss.backward()
		optimizer.step()

		#ensure original weights and biases are axtually different for fully connected layers
		np.testing.assert_raises(AssertionError, np.testing.assert_allclose, fc2.weights, ogweights2)
		np.testing.assert_raises(AssertionError, np.testing.assert_allclose, fc2.bias, ogbias2)
		np.testing.assert_raises(AssertionError, np.testing.assert_allclose, fc1.weights, ogweights1)
		np.testing.assert_raises(AssertionError, np.testing.assert_allclose, fc1.bias, ogbias1)

		#ensure weights/bias/gradients align with pytorch implementation
		np.testing.assert_allclose(fc2.weights, net.linear2.weight.detach().numpy().T)
		np.testing.assert_allclose(fc2.bias, net.linear2.bias.detach().numpy())
		np.testing.assert_allclose(grad_weights2, net.linear2.weight.grad.detach().numpy().T)
		np.testing.assert_allclose(grad_bias2, net.linear2.bias.grad.detach().numpy())

		np.testing.assert_allclose(fc1.weights, net.linear1.weight.detach().numpy().T)
		np.testing.assert_allclose(fc1.bias, net.linear1.bias.detach().numpy())
		np.testing.assert_allclose(grad_weights1, net.linear1.weight.grad.detach().numpy().T)
		np.testing.assert_allclose(grad_bias1, net.linear1.bias.grad.detach().numpy())


	def test_conv_backprop_max(self):
		'''
		testing the backpropagation of convolutional layer with max pooling
		'''
		train_set = datasets.MNIST('./mnist', train=True, download=True)
		test_set = datasets.MNIST('./mnist', train=False, download=True)

		train_set_array = train_set.data.numpy()
		train_set_array = np.expand_dims(train_set_array, axis=1)

		train_set_labels = train_set.targets.numpy()
		train_set_one_hot = np.zeros((train_set_labels.size, train_set_labels.max()+1))
		train_set_one_hot[np.arange(train_set_labels.size),train_set_labels] = 1
		labels = train_set_labels[16:20]
		onehot_labels = train_set_one_hot[16:20]
		data = train_set_array[16:20]
		conv1 = convolution(10,3,1,1)
		conv2 = convolution(10,3,1,1)
		pool_layer = max_pool(size=3,padding=0)
		flatten = flat()
		fc1= fc(input_dim=10*9*9, output_dim=5*10*10)
		fc2 = fc(input_dim=5*10*10, output_dim=10)
		bias1 = np.copy(fc1.bias.T)
		weights1 = np.copy(fc1.weights.T)
		bias2 = np.copy(fc2.bias)
		weights2 = np.copy(fc2.weights.T)



		output = conv1.forward(data)
		output = conv2.forward(output)
		output = pool_layer.forward(output)
		output = flatten.forward(output)
		output = fc1.forward(output)
		output = fc2.forward(output)

		conv_filters1 = np.copy(conv1.filters)
		conv_bias1 = np.copy(conv1.bias)
		og_conv_filters1 = np.copy(conv1.filters)
		og_conv_bias1 = np.copy(conv1.bias)

		conv_filters2 = np.copy(conv2.filters)
		conv_bias2 = np.copy(conv2.bias)
		og_conv_filters2 = np.copy(conv2.filters)
		og_conv_bias2 = np.copy(conv2.bias)

		loss,_, logsoftmax = cross_entropy(output, labels)


		layer = layers()
		#batch size x num_classes
		#derivative of cross entropy loss wrt fully connected output
		gradient = layer.gradient_logsoftmax(logsoftmax, onehot_labels)
		lr = 10
		gradient, grad_weights2, grad_bias2 = fc2.backprop(lr,gradient)
		gradient, grad_weights1, grad_bias1 = fc1.backprop(lr, gradient)
		gradient = flatten.backprop(gradient)
		gradient = pool_layer.backprop(gradient)
		max_pool_grad = gradient
		gradient, grad_conv_filters2, grad_conv_bias2 = conv2.backprop(lr, gradient)
		gradient, grad_conv_filters1, grad_conv_bias1 = conv1.backprop(lr, gradient)


		#pytorch
		inputs = torch.from_numpy(data).double().detach().requires_grad_(True)
		#inputs = torch.tensor(inputs, requires_grad=True)
		class Net(torch.nn.Module):
			def __init__(self):
				super(Net, self).__init__()
				self.conv1 = torch.nn.Conv2d(in_channels=1,out_channels=10, kernel_size=3,stride=1,padding=1,bias=True)
				self.conv1.weight = torch.nn.Parameter(torch.from_numpy(conv_filters1))
				self.conv1.bias =  torch.nn.Parameter(torch.from_numpy(conv_bias1))
				self.conv2 = torch.nn.Conv2d(in_channels=10,out_channels=10, kernel_size=3,stride=1,padding=1,bias=True)
				self.conv2.weight = torch.nn.Parameter(torch.from_numpy(conv_filters2))
				self.conv2.bias =  torch.nn.Parameter(torch.from_numpy(conv_bias2))
				self.maxpool = torch.nn.MaxPool2d(3, padding=0)
				self.linear1 = torch.nn.Linear(in_features=10*9*9,out_features=5*10*10,bias=True)
				self.linear1.weight = torch.nn.Parameter(torch.from_numpy(weights1))
				self.linear1.bias = torch.nn.Parameter(torch.from_numpy(bias1))
				self.linear2 = torch.nn.Linear(in_features=5*10*10,out_features=10,bias=True)
				self.linear2.weight = torch.nn.Parameter(torch.from_numpy(weights2))
				self.linear2.bias = torch.nn.Parameter(torch.from_numpy(bias2))
			def forward(self,x):
				x = self.conv1(x)
				x = self.conv2(x)
				x.retain_grad()
				self.maxpool_grad = x
				x = self.maxpool(x)
				x = x.view(-1,10*9*9)
				x = self.linear1(x)
				x = self.linear2(x)
				return x
		net = Net()
		optimizer = torch.optim.SGD(net.parameters(), lr=lr)
		labels = torch.from_numpy(labels)
		expected_output = net(inputs)
		#maake pytorch output the loss for each training example
		criterion = torch.nn.CrossEntropyLoss()
		expected_loss = criterion(expected_output, labels)
		#np.testing.assert_allclose(loss, expected_loss.detach().numpy())
		optimizer.zero_grad()
		expected_loss.backward()
		optimizer.step()


		#ensure original weights and biases are actually different for conv layers
		try: 
			np.testing.assert_raises(AssertionError, np.testing.assert_allclose, net.conv1.weight.detach().numpy(), og_conv_filters1)
			np.testing.assert_raises(AssertionError, np.testing.assert_allclose, conv1.filters, og_conv_filters1)
			np.testing.assert_raises(AssertionError, np.testing.assert_allclose, net.conv1.bias.detach().numpy(), og_conv_bias1)
			np.testing.assert_raises(AssertionError, np.testing.assert_allclose, conv1.bias, og_conv_bias1)

			np.testing.assert_raises(AssertionError, np.testing.assert_allclose, net.conv2.weight.detach().numpy(), og_conv_filters2)
			np.testing.assert_raises(AssertionError, np.testing.assert_allclose, conv2.filters, og_conv_filters2)
			np.testing.assert_raises(AssertionError, np.testing.assert_allclose, net.conv2.bias.detach().numpy(), og_conv_bias2)
			np.testing.assert_raises(AssertionError, np.testing.assert_allclose, conv2.bias, og_conv_bias2)

		except AssertionError:
			print("gradient was zero so conv filters/biases were the same")
			pass

		np.testing.assert_allclose(max_pool_grad, net.maxpool_grad.grad.detach().numpy())
		np.testing.assert_allclose(conv1.filters, net.conv1.weight.detach().numpy())
		np.testing.assert_allclose(conv1.bias, net.conv1.bias.detach().numpy())
		np.testing.assert_allclose(conv2.filters, net.conv2.weight.detach().numpy())
		np.testing.assert_allclose(conv2.bias, net.conv2.bias.detach().numpy())

		np.testing.assert_allclose(grad_conv_filters1, net.conv1.weight.grad.detach().numpy())
		np.testing.assert_allclose(grad_conv_bias1, net.conv1.bias.grad.detach().numpy())
		np.testing.assert_allclose(grad_conv_filters2, net.conv2.weight.grad.detach().numpy())
		np.testing.assert_allclose(grad_conv_bias2, net.conv2.bias.grad.detach().numpy())

		np.testing.assert_allclose(gradient, inputs.grad)


	def test_conv_backprop_avg(self):
		'''
		testing the backpropagation of convolutional layer with avg pooling
		'''
		train_set = datasets.MNIST('./mnist', train=True, download=True)
		test_set = datasets.MNIST('./mnist', train=False, download=True)

		train_set_array = train_set.data.numpy()
		train_set_array = np.expand_dims(train_set_array, axis=1)

		train_set_labels = train_set.targets.numpy()
		train_set_one_hot = np.zeros((train_set_labels.size, train_set_labels.max()+1))
		train_set_one_hot[np.arange(train_set_labels.size),train_set_labels] = 1
		labels = train_set_labels[16:17]
		onehot_labels = train_set_one_hot[16:17]
		data = train_set_array[16:17]
		conv1 = convolution(5,3,1,1)
		conv2 = convolution(5,3,1,1)
		pool_layer = avg_pool(size=5, padding=2)
		flatten = flat()
		fc1= fc(input_dim=5*6*6, output_dim=5*10*10)
		fc2 = fc(input_dim=5*10*10, output_dim=10)
		bias1 = np.copy(fc1.bias.T)
		weights1 = np.copy(fc1.weights.T)
		bias2 = np.copy(fc2.bias)
		weights2 = np.copy(fc2.weights.T)

		output = conv1.forward(data)
		output = conv2.forward(output)
		output = pool_layer.forward(output)
		output = flatten.forward(output)
		output = fc1.forward(output)
		output = fc2.forward(output)

		conv_filters1 = np.copy(conv1.filters)
		conv_bias1 = np.copy(conv1.bias)
		og_conv_filters1 = np.copy(conv1.filters)
		og_conv_bias1 = np.copy(conv1.bias)

		conv_filters2 = np.copy(conv2.filters)
		conv_bias2 = np.copy(conv2.bias)
		og_conv_filters2 = np.copy(conv2.filters)
		og_conv_bias2 = np.copy(conv2.bias)

		loss,_, logsoftmax = cross_entropy(output, labels)


		layer = layers()
		#batch size x num_classes
		#derivative of cross entropy loss wrt fully connected output
		gradient = layer.gradient_logsoftmax(logsoftmax, onehot_labels)
		lr = 10
		gradient, grad_weights2, grad_bias2 = fc2.backprop(lr,gradient)
		gradient, grad_weights1, grad_bias1 = fc1.backprop(lr, gradient)
		gradient = flatten.backprop(gradient)
		gradient = pool_layer.backprop(gradient)
		avg_pool_grad = gradient
		gradient, grad_conv_filters2, grad_conv_bias2 = conv2.backprop(lr, gradient)
		gradient, grad_conv_filters1, grad_conv_bias1 = conv1.backprop(lr, gradient)


		#pytorch
		inputs = torch.from_numpy(data).double().detach().requires_grad_(True)
		#inputs = torch.tensor(inputs, requires_grad=True)
		class Net(torch.nn.Module):
			def __init__(self):
				super(Net, self).__init__()
				self.conv1 = torch.nn.Conv2d(in_channels=1,out_channels=5, kernel_size=3,stride=1,padding=1,bias=True)
				self.conv1.weight = torch.nn.Parameter(torch.from_numpy(conv_filters1))
				self.conv1.bias =  torch.nn.Parameter(torch.from_numpy(conv_bias1))
				self.conv2 = torch.nn.Conv2d(in_channels=5,out_channels=5, kernel_size=3,stride=1,padding=1,bias=True)
				self.conv2.weight = torch.nn.Parameter(torch.from_numpy(conv_filters2))
				self.conv2.bias =  torch.nn.Parameter(torch.from_numpy(conv_bias2))
				self.avgpool = torch.nn.AvgPool2d(5,padding=2)
				self.linear1 = torch.nn.Linear(in_features=5*6*6,out_features=5*10*10,bias=True)
				self.linear1.weight = torch.nn.Parameter(torch.from_numpy(weights1))
				self.linear1.bias = torch.nn.Parameter(torch.from_numpy(bias1))
				self.linear2 = torch.nn.Linear(in_features=5*10*10,out_features=10,bias=True)
				self.linear2.weight = torch.nn.Parameter(torch.from_numpy(weights2))
				self.linear2.bias = torch.nn.Parameter(torch.from_numpy(bias2))
			def forward(self,x):
				x = self.conv1(x)
				x = self.conv2(x)
				x.retain_grad()
				self.avgpool_grad = x
				x = self.avgpool(x)
				x = x.view(-1,5*6*6)
				x = self.linear1(x)
				x = self.linear2(x)
				return x
		net = Net()
		optimizer = torch.optim.SGD(net.parameters(), lr=lr)
		labels = torch.from_numpy(labels)
		expected_output = net(inputs)
		#maake pytorch output the loss for each training example
		criterion = torch.nn.CrossEntropyLoss()
		expected_loss = criterion(expected_output, labels)
		np.testing.assert_allclose(loss, expected_loss.detach().numpy())
		optimizer.zero_grad()
		expected_loss.backward()
		optimizer.step()

		np.testing.assert_allclose(avg_pool_grad, net.avgpool_grad.grad.detach().numpy())
		#ensure original weights and biases are actually different for conv layers
		try: 
			np.testing.assert_raises(AssertionError, np.testing.assert_allclose, net.conv1.weight.detach().numpy(), og_conv_filters1)
			np.testing.assert_raises(AssertionError, np.testing.assert_allclose, conv1.filters, og_conv_filters1)
			np.testing.assert_raises(AssertionError, np.testing.assert_allclose, net.conv1.bias.detach().numpy(), og_conv_bias1)
			np.testing.assert_raises(AssertionError, np.testing.assert_allclose, conv1.bias, og_conv_bias1)

			np.testing.assert_raises(AssertionError, np.testing.assert_allclose, net.conv2.weight.detach().numpy(), og_conv_filters2)
			np.testing.assert_raises(AssertionError, np.testing.assert_allclose, conv2.filters, og_conv_filters2)
			np.testing.assert_raises(AssertionError, np.testing.assert_allclose, net.conv2.bias.detach().numpy(), og_conv_bias2)
			np.testing.assert_raises(AssertionError, np.testing.assert_allclose, conv2.bias, og_conv_bias2)

		except AssertionError:
			print("gradient was zero so conv filters/biases were the same")
			pass


		np.testing.assert_allclose(conv1.filters, net.conv1.weight.detach().numpy())
		np.testing.assert_allclose(conv1.bias, net.conv1.bias.detach().numpy())
		np.testing.assert_allclose(conv2.filters, net.conv2.weight.detach().numpy())
		np.testing.assert_allclose(conv2.bias, net.conv2.bias.detach().numpy())

		np.testing.assert_allclose(grad_conv_filters1, net.conv1.weight.grad.detach().numpy())
		np.testing.assert_allclose(grad_conv_bias1, net.conv1.bias.grad.detach().numpy())
		np.testing.assert_allclose(grad_conv_filters2, net.conv2.weight.grad.detach().numpy())
		np.testing.assert_allclose(grad_conv_bias2, net.conv2.bias.grad.detach().numpy())

		np.testing.assert_allclose(gradient, inputs.grad)

	"""def test_relu_gradient(self):
		'''
		testing the gradient of relu activation
		'''

		data = np.random.normal(0,1,size=(10,))
		layer = relu()
		output = layer.forward(data)
		gradient_loss = layer.backprop(output)
		relu_torch = torch.nn.ReLU()
		inputs = torch.from_numpy(data).double().detach().requires_grad_(True)
		expected_output = relu_torch(inputs)
		expected_output.sum().backward()
		np.testing.assert_allclose(gradient_loss, inputs.grad.detach().numpy())"""
	
	def test_shallow_net_relu_backprop(self):
		'''
		testing the backprop of a shallow neural network with relu activation
		'''
		train_set = datasets.MNIST('./mnist', train=True, download=True)
		test_set = datasets.MNIST('./mnist', train=False, download=True)

		train_set_array = train_set.data.numpy()
		train_set_array = np.expand_dims(train_set_array, axis=1)

		train_set_labels = train_set.targets.numpy()
		train_set_one_hot = np.zeros((train_set_labels.size, train_set_labels.max()+1))
		train_set_one_hot[np.arange(train_set_labels.size),train_set_labels] = 1
		labels = train_set_labels[15:20]
		onehot_labels = train_set_one_hot[15:20]
		data = train_set_array[15:20]
		fc1= fc(input_dim=1*28*28, output_dim=100, bias=True)
		bias1 = np.copy(fc1.bias.T)
		weights1 = np.copy(fc1.weights.T)

		fc2= fc(input_dim=100, output_dim=10, bias=True)
		bias2 = np.copy(fc2.bias.T)
		weights2 = np.copy(fc2.weights.T)

		layer_relu = relu()

		flatten = flat()
		output = flatten.forward(data)
		output = fc1.forward(output)
		output = layer_relu.forward(output)
		output = fc2.forward(output)

		loss,_, logsoftmax = cross_entropy(output, labels)

		layer = layers()
		#batch size x num_classes
		#derivative of cross entropy loss wrt fully connected output
		gradient_loss = layer.gradient_logsoftmax(logsoftmax, onehot_labels)
		lr = 10

		gradient_loss_fc2, grad_weights2, grad_bias2 = fc2.backprop(lr,gradient_loss)
		gradient_loss_relu = layer_relu.backprop(gradient_loss_fc2)
		gradient_loss_fc1, grad_weights1, grad_bias1 = fc1.backprop(lr, gradient_loss_relu)
		#gradient_loss = flatten.backprop(gradient_loss_fc1)


		#pytorch
		inputs = torch.from_numpy(data).double().detach().requires_grad_(True)
		#inputs = torch.tensor(inputs, requires_grad=True)
		class Net(torch.nn.Module):
			def __init__(self):
				super(Net, self).__init__()
				self.linear1 = torch.nn.Linear(in_features=1*28*28,out_features=10*10,bias=True)
				self.linear1.weight = torch.nn.Parameter(torch.from_numpy(weights1))
				self.linear1.bias = torch.nn.Parameter(torch.from_numpy(bias1))
				self.linear2 = torch.nn.Linear(in_features=10*10,out_features=10,bias=True)
				self.linear2.weight = torch.nn.Parameter(torch.from_numpy(weights2))
				self.linear2.bias = torch.nn.Parameter(torch.from_numpy(bias2))
				self.relu = torch.nn.ReLU()
				self.logsoftmax = torch.nn.LogSoftmax()

			def forward(self,x):
				x = x.view(-1,1*28*28)
				linear_out = self.linear1(x)
				linear_out.retain_grad()
				self.after_relu_grad = linear_out
				relu_out = self.relu(linear_out)
				x = relu_out
				x = self.linear2(x)
				x.retain_grad()
				self.after_softmax = x
				x = self.logsoftmax(x)
				return x
		net = Net()
		optimizer = torch.optim.SGD(net.parameters(), lr=lr)
		labels = torch.from_numpy(labels)
		expected_output = net(inputs)
		#make pytorch output the loss for each training example
		criterion = torch.nn.NLLLoss()
		expected_loss = criterion(expected_output, labels)
		np.testing.assert_allclose(loss, expected_loss.detach().numpy())
		optimizer.zero_grad()
		expected_loss.backward()
		optimizer.step()

		#np.testing.assert_allclose(gradient_loss_fc, inputs.grad)
		np.testing.assert_allclose(gradient_loss, net.after_softmax.grad)
		#np.testing.assert_allclose(gradient_loss_relu, net.after_relu_grad.grad)
		#ensure original weights and biases are actually different for fully connected layers
		try: 
			np.testing.assert_raises(AssertionError, np.testing.assert_allclose, net.linear1.weight.detach().numpy(), weights1)
			np.testing.assert_raises(AssertionError, np.testing.assert_allclose, fc1.weights, weights1)
			np.testing.assert_raises(AssertionError, np.testing.assert_allclose, net.linear1.bias.detach().numpy(), bias1)
			np.testing.assert_raises(AssertionError, np.testing.assert_allclose, fc1.bias, bias1)

			np.testing.assert_raises(AssertionError, np.testing.assert_allclose, net.linear2.weight.detach().numpy(), weights2)
			np.testing.assert_raises(AssertionError, np.testing.assert_allclose, fc2.weights, weights2)
			np.testing.assert_raises(AssertionError, np.testing.assert_allclose, net.linear2.bias.detach().numpy(), bias2)
			np.testing.assert_raises(AssertionError, np.testing.assert_allclose, fc2.bias, bias2)


		except AssertionError:
			print("gradient was zero so weights/biases were the same")
			pass

		np.testing.assert_allclose(fc1.weights, net.linear1.weight.detach().numpy().T)
		np.testing.assert_allclose(fc1.bias, net.linear1.bias.detach().numpy())
		np.testing.assert_allclose(grad_weights1, net.linear1.weight.grad.detach().numpy().T)
		np.testing.assert_allclose(grad_bias1, net.linear1.bias.grad.detach().numpy())

		np.testing.assert_allclose(fc2.weights, net.linear2.weight.detach().numpy().T)
		np.testing.assert_allclose(fc2.bias, net.linear2.bias.detach().numpy())
		np.testing.assert_allclose(grad_weights2, net.linear2.weight.grad.detach().numpy().T)
		np.testing.assert_allclose(grad_bias2, net.linear2.bias.grad.detach().numpy())
		
		np.testing.assert_allclose(gradient_loss_fc1, inputs.grad)



	def test_conv_net_relu(self):
		'''
		testing the backpropagation of convolutional layer with relu activation
		'''
		train_set = datasets.MNIST('./mnist', train=True, download=True)
		test_set = datasets.MNIST('./mnist', train=False, download=True)

		train_set_array = train_set.data.numpy()
		train_set_array = np.expand_dims(train_set_array, axis=1)

		train_set_labels = train_set.targets.numpy()
		train_set_one_hot = np.zeros((train_set_labels.size, train_set_labels.max()+1))
		train_set_one_hot[np.arange(train_set_labels.size),train_set_labels] = 1
		labels = train_set_labels[16:17]
		onehot_labels = train_set_one_hot[16:17]
		data = train_set_array[16:17]
		conv1 = convolution(5,3,1,1)
		conv2 = convolution(5,3,1,1)
		pool_layer = max_pool(size=3)
		flatten = flat()
		fc1= fc(input_dim=5*9*9, output_dim=5*10*10)
		fc2 = fc(input_dim=5*10*10, output_dim=10)
		bias1 = np.copy(fc1.bias.T)
		weights1 = np.copy(fc1.weights.T)
		bias2 = np.copy(fc2.bias)
		weights2 = np.copy(fc2.weights.T)
		relu1 = relu()
		relu2 = relu()
		relu3 = relu()
		output = conv1.forward(data)
		output = relu1.forward(output)
		output = conv2.forward(output)
		output = relu2.forward(output)
		output = pool_layer.forward(output)
		output = flatten.forward(output)
		output = fc1.forward(output)
		output = relu3.forward(output)
		output = fc2.forward(output)

		conv_filters1 = np.copy(conv1.filters)
		conv_bias1 = np.copy(conv1.bias)
		og_conv_filters1 = np.copy(conv1.filters)
		og_conv_bias1 = np.copy(conv1.bias)

		conv_filters2 = np.copy(conv2.filters)
		conv_bias2 = np.copy(conv2.bias)
		og_conv_filters2 = np.copy(conv2.filters)
		og_conv_bias2 = np.copy(conv2.bias)

		loss,_, logsoftmax = cross_entropy(output, labels)


		layer = layers()
		#batch size x num_classes
		#derivative of cross entropy loss wrt fully connected output
		gradient = layer.gradient_logsoftmax(logsoftmax, onehot_labels)
		lr = 10
		gradient, grad_weights2, grad_bias2 = fc2.backprop(lr,gradient)
		gradient = relu3.backprop(gradient)
		gradient, grad_weights1, grad_bias1 = fc1.backprop(lr, gradient)
		gradient = flatten.backprop(gradient)
		gradient = pool_layer.backprop(gradient)
		gradient = relu2.backprop(gradient)
		gradient, grad_conv_filters2, grad_conv_bias2 = conv2.backprop(lr, gradient)
		gradient = relu1.backprop(gradient)
		gradient, grad_conv_filters1, grad_conv_bias1 = conv1.backprop(lr, gradient)


		#pytorch
		inputs = torch.from_numpy(data).double().detach().requires_grad_(True)
		#inputs = torch.tensor(inputs, requires_grad=True)
		class Net(torch.nn.Module):
			def __init__(self):
				super(Net, self).__init__()
				self.conv1 = torch.nn.Conv2d(in_channels=1,out_channels=5, kernel_size=3,stride=1,padding=1,bias=True)
				self.conv1.weight = torch.nn.Parameter(torch.from_numpy(conv_filters1))
				self.conv1.bias =  torch.nn.Parameter(torch.from_numpy(conv_bias1))
				self.conv2 = torch.nn.Conv2d(in_channels=5,out_channels=5, kernel_size=3,stride=1,padding=1,bias=True)
				self.conv2.weight = torch.nn.Parameter(torch.from_numpy(conv_filters2))
				self.conv2.bias =  torch.nn.Parameter(torch.from_numpy(conv_bias2))
				self.maxpool = torch.nn.MaxPool2d(3)
				self.linear1 = torch.nn.Linear(in_features=5*9*9,out_features=5*10*10,bias=True)
				self.linear1.weight = torch.nn.Parameter(torch.from_numpy(weights1))
				self.linear1.bias = torch.nn.Parameter(torch.from_numpy(bias1))
				self.linear2 = torch.nn.Linear(in_features=5*10*10,out_features=10,bias=True)
				self.linear2.weight = torch.nn.Parameter(torch.from_numpy(weights2))
				self.linear2.bias = torch.nn.Parameter(torch.from_numpy(bias2))
				self.relu1 = torch.nn.ReLU()
				self.relu2 = torch.nn.ReLU()
				self.relu3 = torch.nn.ReLU()
			def forward(self,x):
				x = self.conv1(x)
				x = self.relu1(x)
				x = self.conv2(x)
				x = self.relu2(x)
				x = self.maxpool(x)
				x = x.view(-1,5*9*9)
				x = self.linear1(x)
				x = self.relu3(x)
				x = self.linear2(x)
				return x
		net = Net()
		optimizer = torch.optim.SGD(net.parameters(), lr=lr)
		labels = torch.from_numpy(labels)
		expected_output = net(inputs)
		#make pytorch output the loss for each training example
		criterion = torch.nn.CrossEntropyLoss()
		expected_loss = criterion(expected_output, labels)
		np.testing.assert_allclose(loss, expected_loss.item())
		optimizer.zero_grad()
		expected_loss.backward()
		optimizer.step()

		#ensure original weights and biases are actually different for conv layers
		try: 
			np.testing.assert_raises(AssertionError, np.testing.assert_allclose, net.conv1.weight.detach().numpy(), og_conv_filters1)
			np.testing.assert_raises(AssertionError, np.testing.assert_allclose, conv1.filters, og_conv_filters1)
			np.testing.assert_raises(AssertionError, np.testing.assert_allclose, net.conv1.bias.detach().numpy(), og_conv_bias1)
			np.testing.assert_raises(AssertionError, np.testing.assert_allclose, conv1.bias, og_conv_bias1)

			np.testing.assert_raises(AssertionError, np.testing.assert_allclose, net.conv2.weight.detach().numpy(), og_conv_filters2)
			np.testing.assert_raises(AssertionError, np.testing.assert_allclose, conv2.filters, og_conv_filters2)
			np.testing.assert_raises(AssertionError, np.testing.assert_allclose, net.conv2.bias.detach().numpy(), og_conv_bias2)
			np.testing.assert_raises(AssertionError, np.testing.assert_allclose, conv2.bias, og_conv_bias2)

		except AssertionError:
			print("gradient was zero so conv filters/biases were the same")
			pass


		np.testing.assert_allclose(conv1.filters, net.conv1.weight.detach().numpy())
		np.testing.assert_allclose(conv1.bias, net.conv1.bias.detach().numpy())
		np.testing.assert_allclose(conv2.filters, net.conv2.weight.detach().numpy())
		np.testing.assert_allclose(conv2.bias, net.conv2.bias.detach().numpy())

		np.testing.assert_allclose(grad_conv_filters1, net.conv1.weight.grad.detach().numpy())
		np.testing.assert_allclose(grad_conv_bias1, net.conv1.bias.grad.detach().numpy())
		np.testing.assert_allclose(grad_conv_filters2, net.conv2.weight.grad.detach().numpy())
		np.testing.assert_allclose(grad_conv_bias2, net.conv2.bias.grad.detach().numpy())

		np.testing.assert_allclose(gradient, inputs.grad)
	
	def test_conv_training(self):
		'''
		testing the training of a convolutional neural network through a few epochs
		'''
		train_set = datasets.MNIST('./mnist', train=True, download=True)
		test_set = datasets.MNIST('./mnist', train=False, download=True)

		train_set_array = train_set.data.numpy()
		train_set_array = np.expand_dims(train_set_array, axis=1)

		train_set_labels = train_set.targets.numpy()
		train_set_one_hot = np.zeros((train_set_labels.size, train_set_labels.max()+1))
		train_set_one_hot[np.arange(train_set_labels.size),train_set_labels] = 1
		labels = train_set_labels[0:5]
		onehot_labels = train_set_one_hot[0:5]
		data = train_set_array[0:5]
		conv1 = convolution(5,3,1,1)
		conv2 = convolution(5,3,1,1)
		pool_layer = max_pool(size=3)
		flatten = flat()
		fc1= fc(input_dim=5*9*9, output_dim=5*10*10)
		fc2 = fc(input_dim=5*10*10, output_dim=10)
		bias1 = np.copy(fc1.bias.T)
		weights1 = np.copy(fc1.weights.T)
		bias2 = np.copy(fc2.bias)
		weights2 = np.copy(fc2.weights.T)
		relu1 = relu()
		relu2 = relu()
		relu3 = relu()
		temp = conv1.forward(data)
		conv2.forward(temp)
		conv_filters1 = np.copy(conv1.filters)
		conv_bias1 = np.copy(conv1.bias)
		og_conv_filters1 = np.copy(conv1.filters)
		og_conv_bias1 = np.copy(conv1.bias)

		conv_filters2 = np.copy(conv2.filters)
		conv_bias2 = np.copy(conv2.bias)
		og_conv_filters2 = np.copy(conv2.filters)
		og_conv_bias2 = np.copy(conv2.bias)

		#pytorch
		inputs = torch.from_numpy(data).double().detach().requires_grad_(True)
		class Net(torch.nn.Module):
			def __init__(self):
				super(Net, self).__init__()
				self.conv1 = torch.nn.Conv2d(in_channels=1,out_channels=5, kernel_size=3,stride=1,padding=1,bias=True)
				self.conv1.weight = torch.nn.Parameter(torch.from_numpy(conv_filters1))
				self.conv1.bias =  torch.nn.Parameter(torch.from_numpy(conv_bias1))
				self.conv2 = torch.nn.Conv2d(in_channels=5,out_channels=5, kernel_size=3,stride=1,padding=1,bias=True)
				self.conv2.weight = torch.nn.Parameter(torch.from_numpy(conv_filters2))
				self.conv2.bias =  torch.nn.Parameter(torch.from_numpy(conv_bias2))
				self.maxpool = torch.nn.MaxPool2d(3)
				self.linear1 = torch.nn.Linear(in_features=5*9*9,out_features=5*10*10,bias=True)
				self.linear1.weight = torch.nn.Parameter(torch.from_numpy(weights1))
				self.linear1.bias = torch.nn.Parameter(torch.from_numpy(bias1))
				self.linear2 = torch.nn.Linear(in_features=5*10*10,out_features=10,bias=True)
				self.linear2.weight = torch.nn.Parameter(torch.from_numpy(weights2))
				self.linear2.bias = torch.nn.Parameter(torch.from_numpy(bias2))
				self.relu1 = torch.nn.ReLU()
				self.relu2 = torch.nn.ReLU()
				self.relu3 = torch.nn.ReLU()
			def forward(self,x):
				x = self.conv1(x)
				x = self.relu1(x)
				x = self.conv2(x)
				x = self.relu2(x)
				x = self.maxpool(x)
				x = x.view(-1,5*9*9)
				x = self.linear1(x)
				x = self.relu3(x)
				x = self.linear2(x)
				return x
		lr = 0.5
		running_loss = 0
		final_grad = None
		for i in range(1):
			output = conv1.forward(data)
			output = relu1.forward(output)
			output = conv2.forward(output)
			output = relu2.forward(output)
			output = pool_layer.forward(output)
			output = flatten.forward(output)
			output = fc1.forward(output)
			output = relu3.forward(output)
			output = fc2.forward(output)


			loss,_, logsoftmax = cross_entropy(output, labels)
			running_loss += loss

			layer = layers()
			#batch size x num_classes
			#derivative of cross entropy loss wrt fully connected output
			gradient = layer.gradient_logsoftmax(logsoftmax, onehot_labels)
			gradient, grad_weights2, grad_bias2 = fc2.backprop(lr,gradient)
			gradient = relu3.backprop(gradient)
			gradient, grad_weights1, grad_bias1 = fc1.backprop(lr, gradient)
			gradient = flatten.backprop(gradient)
			gradient = pool_layer.backprop(gradient)
			gradient = relu2.backprop(gradient)
			gradient, grad_conv_filters2, grad_conv_bias2 = conv2.backprop(lr, gradient)
			gradient = relu1.backprop(gradient)
			gradient, grad_conv_filters1, grad_conv_bias1 = conv1.backprop(lr, gradient)
			if(final_grad is None):
				final_grad = gradient
			else:
				final_grad += gradient


		net = Net()
		optimizer = torch.optim.SGD(net.parameters(), lr=lr)
		labels = torch.from_numpy(labels)
		expected_running_loss = 0
		for i in range(1):
			expected_output = net(inputs)
			#maake pytorch output the loss for each training example
			criterion = torch.nn.CrossEntropyLoss()
			expected_loss = criterion(expected_output, labels)
			expected_running_loss += expected_loss.item()
			optimizer.zero_grad()
			expected_loss.backward()
			optimizer.step()

		#np.testing.assert_allclose(running_loss, expected_running_loss)

		#ensure original weights and biases are actually different for conv layers
		try: 
			np.testing.assert_raises(AssertionError, np.testing.assert_allclose, net.conv1.weight.detach().numpy(), og_conv_filters1)
			np.testing.assert_raises(AssertionError, np.testing.assert_allclose, conv1.filters, og_conv_filters1)
			np.testing.assert_raises(AssertionError, np.testing.assert_allclose, net.conv1.bias.detach().numpy(), og_conv_bias1)
			np.testing.assert_raises(AssertionError, np.testing.assert_allclose, conv1.bias, og_conv_bias1)

			np.testing.assert_raises(AssertionError, np.testing.assert_allclose, net.conv2.weight.detach().numpy(), og_conv_filters2)
			np.testing.assert_raises(AssertionError, np.testing.assert_allclose, conv2.filters, og_conv_filters2)
			np.testing.assert_raises(AssertionError, np.testing.assert_allclose, net.conv2.bias.detach().numpy(), og_conv_bias2)
			np.testing.assert_raises(AssertionError, np.testing.assert_allclose, conv2.bias, og_conv_bias2)

		except AssertionError:
			print("gradient was zero so conv filters/biases were the same")
			pass


		np.testing.assert_allclose(conv1.filters, net.conv1.weight.detach().numpy())
		np.testing.assert_allclose(conv1.bias, net.conv1.bias.detach().numpy())
		np.testing.assert_allclose(conv2.filters, net.conv2.weight.detach().numpy())
		np.testing.assert_allclose(conv2.bias, net.conv2.bias.detach().numpy())

		np.testing.assert_allclose(grad_conv_filters1, net.conv1.weight.grad.detach().numpy())
		np.testing.assert_allclose(grad_conv_bias1, net.conv1.bias.grad.detach().numpy())
		np.testing.assert_allclose(grad_conv_filters2, net.conv2.weight.grad.detach().numpy())
		np.testing.assert_allclose(grad_conv_bias2, net.conv2.bias.grad.detach().numpy())

		#input gradient is accumulated instead of reset to (because it is not part of the model parameters)
		np.testing.assert_allclose(final_grad, inputs.grad)

	def test_convnet_training(self):
		'''
		testing the training of a convolutional neural network with convnet class through a few epochs and minibatch SGD
		'''
		model = [convolution(num_filters=10,kernel_size=3,padding=1,stride=1,bias=True), relu(), max_pool(size=3), convolution(num_filters=10,kernel_size=3,padding=1,stride=1,bias=True),relu(), max_pool(size=3), flat(), fc(input_dim=90, output_dim=10,bias=True)]
		c = ConvNet(model)
		train_set = datasets.MNIST('./mnist', train=True, download=True)
		test_set = datasets.MNIST('./mnist', train=False, download=True)

		train_set_array = train_set.data.numpy()
		train_set_array = np.expand_dims(train_set_array, axis=1)

		train_set_labels = train_set.targets.numpy()
		train_set_one_hot = np.zeros((train_set_labels.size, train_set_labels.max()+1))
		train_set_one_hot[np.arange(train_set_labels.size),train_set_labels] = 1
		labels = train_set_labels
		onehot_labels = train_set_one_hot
		data = train_set_array

		temp_data = data[0:128]
		fc_weights = np.copy(c.model[7].weights.T)
		fc_bias = np.copy(c.model[7].bias.T)
		c.forward(temp_data)

		conv_filters1 = np.copy(c.model[0].filters)
		conv_bias1 = np.copy(c.model[0].bias)
		conv_filters2 = np.copy(c.model[3].filters)
		conv_bias2 = np.copy(c.model[3].bias)

		class Net(torch.nn.Module):
			def __init__(self):
				super(Net, self).__init__()
				self.conv1 = torch.nn.Conv2d(in_channels=1,out_channels=10, kernel_size=3,stride=1,padding=1,bias=True)
				self.conv1.weight = torch.nn.Parameter(torch.from_numpy(conv_filters1))
				self.conv1.bias =  torch.nn.Parameter(torch.from_numpy(conv_bias1))
				self.maxpool1 = torch.nn.MaxPool2d(3)
				self.conv2 = torch.nn.Conv2d(in_channels=10,out_channels=10, kernel_size=3,stride=1,padding=1,bias=True)
				self.conv2.weight = torch.nn.Parameter(torch.from_numpy(conv_filters2))
				self.conv2.bias =  torch.nn.Parameter(torch.from_numpy(conv_bias2))
				self.maxpool2 = torch.nn.MaxPool2d(3)
				self.linear1 = torch.nn.Linear(in_features=90,out_features=10,bias=True)
				self.linear1.weight = torch.nn.Parameter(torch.from_numpy(fc_weights))
				self.linear1.bias = torch.nn.Parameter(torch.from_numpy(fc_bias))
				self.relu1 = torch.nn.ReLU()
				self.relu2 = torch.nn.ReLU()
			def forward(self,x):
				x = self.conv1(x)
				x = self.relu1(x)
				x = self.maxpool1(x)
				x = self.conv2(x)
				x = self.relu2(x)
				x = self.maxpool2(x)
				x = x.view(-1,90)
				x = self.linear1(x)
				return x

		batches = math.ceil(1000/128)
		batch_size = 128

		lr = 0.01
		net = Net()
		optimizer = torch.optim.SGD(net.parameters(), lr=lr)
		for epoch in range(5):
			running_loss = 0
			num_correct = 0
			expected_running_loss = 0
			permutation = np.random.permutation(len(labels))
			data = data[permutation]
			labels = labels[permutation]
			onehot_labels = onehot_labels[permutation]
			num_expected_correct = 0
			for j in range(batches):
				begin = j*128
				end = min(begin + batch_size, data.shape[0])
				batch_data = data[begin:end]
				batch_label = labels[begin:end]
				batch_label_onehot = onehot_labels[begin:end]

				output = c.forward(batch_data)
				loss, correct, softmax = cross_entropy(output, batch_label)
				running_loss += loss
				num_correct += correct
				gradient = c.backward(softmax, batch_label_onehot, lr)

				batch_inputs_pytorch = torch.from_numpy(batch_data).double().detach().requires_grad_(True)
				batch_label_pytorch =  torch.from_numpy(batch_label)
				expected_output = net(batch_inputs_pytorch)
				_, predicted = torch.max(expected_output.data, 1)
				expected_correct = (predicted == batch_label_pytorch).sum().item()
				num_expected_correct += expected_correct
				criterion = torch.nn.CrossEntropyLoss()
				expected_loss = criterion(expected_output, batch_label_pytorch)
				expected_running_loss += expected_loss.item()
				optimizer.zero_grad()
				expected_loss.backward()
				optimizer.step()

				np.testing.assert_allclose(c.model[0].filters, net.conv1.weight.detach().numpy(), rtol=0.1, atol=0.1)
				np.testing.assert_allclose(c.model[0].bias, net.conv1.bias.detach().numpy(), rtol=0.1, atol=0.1)
				np.testing.assert_allclose(c.model[3].filters, net.conv2.weight.detach().numpy(), rtol=0.1, atol=0.1)
				np.testing.assert_allclose(c.model[3].bias, net.conv2.bias.detach().numpy(), rtol=0.1, atol=0.1)
				np.testing.assert_allclose(output, expected_output.detach().numpy(), rtol=0.1, atol=0.1)

				#input gradient is accumulated instead of reset to (because it is not part of the model parameters)
				np.testing.assert_allclose(gradient, batch_inputs_pytorch.grad, rtol=0.1, atol=0.1)
				"""print("Done with batch {}".format(j))
				print("Correct: {}".format(correct))
				print("Expected correct: {}".format(expected_correct))
				print("batch loss: {}".format(loss))
				print("expected batch loss: {}".format(expected_loss.sum().detach()))


			print("Epoch: {} Number of total correct: {}".format(epoch, num_correct)) 
			print("Accuracy: {}".format(num_correct/len(train_set_labels))) 
			print("Expected Accuracy: {}".format(num_expected_correct/len(train_set_labels)))
			print("Epoch: {} Loss: {}".format(epoch, running_loss))
			print("Epoch: {} Expected Loss: {}".format(epoch, expected_running_loss))"""

	def test_convnet_training_color(self):
		'''
		testing the training of a convolutional neural network with convnet class through an epoch and with color images
		'''
		model = [convolution(num_filters=10,kernel_size=3,padding=1,stride=1,bias=True), relu(), max_pool(size=3), convolution(num_filters=10,kernel_size=3,padding=1,stride=1,bias=True),relu(), max_pool(size=3), flat(), fc(input_dim=90, output_dim=10,bias=True)]
		c = ConvNet(model)
		train_images, train_labels, test_images, test_labels = cifar10(path=None)
		labels = np.argmax(train_labels, axis=1)
		onehot_labels = train_labels
		data = train_images.reshape((50000, 32,32,3))
		data = np.moveaxis(data, 3,1)
	
		temp_data = data[0:128]
		fc_weights = np.copy(c.model[7].weights.T)
		fc_bias = np.copy(c.model[7].bias.T)
		c.forward(temp_data)

		conv_filters1 = np.copy(c.model[0].filters)
		conv_bias1 = np.copy(c.model[0].bias)
		conv_filters2 = np.copy(c.model[3].filters)
		conv_bias2 = np.copy(c.model[3].bias)

		class Net(torch.nn.Module):
			def __init__(self):
				super(Net, self).__init__()
				self.conv1 = torch.nn.Conv2d(in_channels=3,out_channels=10, kernel_size=3,stride=1,padding=1,bias=True)
				self.conv1.weight = torch.nn.Parameter(torch.from_numpy(conv_filters1))
				self.conv1.bias =  torch.nn.Parameter(torch.from_numpy(conv_bias1))
				self.maxpool1 = torch.nn.MaxPool2d(3)
				self.conv2 = torch.nn.Conv2d(in_channels=10,out_channels=10, kernel_size=3,stride=1,padding=1,bias=True)
				self.conv2.weight = torch.nn.Parameter(torch.from_numpy(conv_filters2))
				self.conv2.bias =  torch.nn.Parameter(torch.from_numpy(conv_bias2))
				self.maxpool2 = torch.nn.MaxPool2d(3)
				self.linear1 = torch.nn.Linear(in_features=90,out_features=10,bias=True)
				self.linear1.weight = torch.nn.Parameter(torch.from_numpy(fc_weights))
				self.linear1.bias = torch.nn.Parameter(torch.from_numpy(fc_bias))
				self.relu1 = torch.nn.ReLU()
				self.relu2 = torch.nn.ReLU()
			def forward(self,x):
				x = self.conv1(x)
				x = self.relu1(x)
				x = self.maxpool1(x)
				x = self.conv2(x)
				x = self.relu2(x)
				x = self.maxpool2(x)
				x = x.view(-1,90)
				x = self.linear1(x)
				return x

		batches = math.ceil(500/128)
		batch_size = 128

		lr = 0.01
		net = Net()
		optimizer = torch.optim.SGD(net.parameters(), lr=lr)
		for epoch in range(2):
			running_loss = 0
			num_correct = 0
			expected_running_loss = 0
			permutation = np.random.permutation(len(labels))
			data = data[permutation]
			labels = labels[permutation]
			onehot_labels = onehot_labels[permutation]
			num_expected_correct = 0
			for j in range(batches):
				begin = j*128
				end = min(begin + batch_size, data.shape[0])
				batch_data = data[begin:end]
				batch_label = labels[begin:end]
				batch_label_onehot = onehot_labels[begin:end]

				output = c.forward(batch_data)
				loss, correct, softmax = cross_entropy(output, batch_label)
				running_loss += loss
				num_correct += correct
				gradient = c.backward(softmax, batch_label_onehot, lr)

				batch_inputs_pytorch = torch.from_numpy(batch_data).double().detach().requires_grad_(True)
				batch_label_pytorch =  torch.from_numpy(batch_label)
				expected_output = net(batch_inputs_pytorch)
				_, predicted = torch.max(expected_output.data, 1)
				expected_correct = (predicted == batch_label_pytorch).sum().item()
				num_expected_correct += expected_correct
				criterion = torch.nn.CrossEntropyLoss()
				expected_loss = criterion(expected_output, batch_label_pytorch)
				expected_running_loss += expected_loss.item()
				optimizer.zero_grad()
				expected_loss.backward()
				optimizer.step()

				np.testing.assert_allclose(c.model[0].filters, net.conv1.weight.detach().numpy(), rtol=0.1, atol=0.1)
				np.testing.assert_allclose(c.model[0].bias, net.conv1.bias.detach().numpy(), rtol=0.1, atol=0.1)
				np.testing.assert_allclose(c.model[3].filters, net.conv2.weight.detach().numpy(), rtol=0.1, atol=0.1)
				np.testing.assert_allclose(c.model[3].bias, net.conv2.bias.detach().numpy(), rtol=0.1, atol=0.1)
				np.testing.assert_allclose(output, expected_output.detach().numpy(), rtol=0.1, atol=0.1)

				#input gradient is accumulated instead of reset to (because it is not part of the model parameters)
				np.testing.assert_allclose(gradient, batch_inputs_pytorch.grad, rtol=0.1, atol=0.1)
				"""print("Done with batch {}".format(j))
				print("Correct: {}".format(correct))
				print("Expected correct: {}".format(expected_correct))
				print("batch loss: {}".format(loss))
				print("expected batch loss: {}".format(expected_loss.sum().detach()))


			print("Epoch: {} Number of total correct: {}".format(epoch, num_correct)) 
			print("Accuracy: {}".format(num_correct/1000)) 
			print("Expected Accuracy: {}".format(num_expected_correct/1000))
			print("Epoch: {} Loss: {}".format(epoch, running_loss))
			print("Epoch: {} Expected Loss: {}".format(epoch, expected_running_loss))"""


if __name__ == '__main__':
	unittest.main()