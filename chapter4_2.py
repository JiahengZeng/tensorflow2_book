import tensorflow as tf
import numpy as np

# X = tf.constant([[2013, 2014, 2015, 2016, 2017], [2013, 2014, 2015, 2016, 2017]])
# Y = tf.constant([[12000, 14000, 15000, 16500, 17500], [2013, 2014, 2015, 2016, 2017]])  # 第0维必须相同

# dataset = tf.data.Dataset.from_tensor_slices((X, Y))

# for x, y in dataset:
#     print(x.numpy(), y.numpy())

import matplotlib.pyplot as plt

(train_data, train_label), (_, _) = tf.keras.datasets.mnist.load_data()
train_data = np.expand_dims(train_data.astype(np.float32) / 255.0, axis=-1)
mnist_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_label))
for image, label in mnist_dataset:
    plt.title(label.numpy())
    plt.imshow(image.numpy()[:, :, 0])
    plt.show()