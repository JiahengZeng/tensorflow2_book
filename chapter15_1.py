import tensorflow as tf
import numpy as np
import time

class Dense(tf.keras.layers.Layer):
    def __init__(self, num_units, **kwargs):
        super().__init__(**kwargs)
        self.num_units = num_units

    def build(self, input_shape):
        # 在首次调用call之前自动调用build
        # 自动提取input的size, 所以只需要管这一层多少神经元
        self.weight = self.add_variable(name='weight', shape=[input_shape[-1], self.num_units])
        self.bais = self.add_variable(name='bais', shape=[self.num_units])
    
    def call(self, inputs):
        y_pred = tf.matmul(inputs, self.weight) + self.bais  # bias忽略batch_size 自动补全了
        return y_pred
    
class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = Dense(num_units=10, name='dense1')
        self.dense2 = Dense(num_units=10, name='dense2')
    
    @tf.function
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense2(x)
        return x

if __name__ == "__main__":
    model = Model()
    # print(model(np.random.rand(10, 32)))
    graph = model.call.get_concrete_function(np.random.rand(10, 32))
    print(graph.variables)