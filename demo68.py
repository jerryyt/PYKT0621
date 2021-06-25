import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print("train shape={}".format(train_images.shape))
print("test shape={}".format(test_images.shape))
image1 = train_images[0]
print("train label shape={}".format(train_labels.shape))
print("test label shape={}".format(test_labels.shape))


def plotImage(index):
    plt.title("The image #%d marked as %d" % (index, train_labels[index]))
    plt.imshow(train_images[index], cmap="binary")
    plt.show()


def plotTestImage(index):
    plt.title("The test image #%d marked as %d" % (index, test_labels[index]))
    plt.imshow(test_images[index], cmap='binary')
    plt.show()


plotTestImage(2000)