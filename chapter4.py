import tensorflow as tf
import tensorflow_datasets as tfds

num_epoch = 5
batch_size = 19
learning_rate = 0.001
dataset = tfds.load("tf_flowers", split=tfds.Split.TRAIN, as_supervised=True)
dataset = dataset.map(lambda img, label:(tf.image.resize(img, (224, 224))/255.0, label)).shuffle(1024).batch(batch_size)
a = 1