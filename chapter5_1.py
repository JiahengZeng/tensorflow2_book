import tensorflow as tf
from chapter3_1 import MNISTLoader

# Sequntial 方法构建模型
num_epochs = 1
batch_size = 50
learning_rate = 1e-3

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation=tf.nn.relu),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Softmax()
])
data_loader = MNISTLoader()

def train():
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=[tf.keras.metrics.sparse_categorical_accuracy]
    )
    model.fit(data_loader.train_data, data_loader.train_label, epochs=num_epochs, batch_size=batch_size)
    tf.saved_model.save(model, "saved/1")

def test():
    # batch_size = 5
    model = tf.saved_model.load("saved/2")
    sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    num_batches = int(data_loader.num_test_data // batch_size)
    for batch_index in range(num_batches):
        start_index, end_index = batch_index * batch_size, (batch_index + 1) *batch_size
        y_pred = model(data_loader.test_data[start_index: end_index])
        sparse_categorical_accuracy.update_state(y_true=data_loader.test_label[start_index: end_index], y_pred=y_pred)
    print("test accuracy: {:.4f}".format(sparse_categorical_accuracy.result()))    

def train_with_super_model():

    class MLP(tf.keras.models.Model):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.flatten = tf.keras.layers.Flatten()
            self.dense1 = tf.keras.layers.Dense(100, activation=tf.nn.relu)
            self.dense2 = tf.keras.layers.Dense(10)
        
        @tf.function
        def call(self, x):
            x = self.flatten(x)
            x = self.dense1(x)
            x = tf.nn.softmax(self.dense2(x))
            return x
        
    model = MLP()
    num_batches = int(data_loader.num_train_data // batch_size)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    for batch_index in range(num_batches):
        start_index, end_index = batch_index * batch_size, (batch_index + 1) *batch_size
        train_data, train_label = data_loader.train_data[start_index: end_index], data_loader.train_label[start_index: end_index]
        with tf.GradientTape() as tape:
            y_pred = model(train_data)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=train_label, y_pred=y_pred)
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
    tf.saved_model.save(model, "saved/2")

if __name__ == "__main__":
    # train()
    # train_with_super_model()
    test()

