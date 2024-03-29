import tensorflow as tf
import numpy as np
import random
from chapter3_1 import MLP
from chapter3_1 import MNISTLoader
import argparse


parser = argparse.ArgumentParser(description="Process some integers.")  # argument ->命令行   parser->解释器
parser.add_argument('--mode', default='train', help='train or test')
parser.add_argument('--num_epochs', default=1)
parser.add_argument('--batch_size', default=50)
parser.add_argument('--learning_rate', default=1e-3)
args = parser.parse_args()
data_loader = MNISTLoader()

def train():
    model = MLP()
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    num_batches = int(data_loader.num_train_data // args.batch_size * args.num_epochs) # 确定需要每个batch_size放多少数据
    checkpoint = tf.train.Checkpoint(myAwesomeModel=model)
    summary_writer = tf.summary.create_file_writer('./tensorboard')
    tf.summary.trace_on(profiler=True)

    for batch_index in range(1, num_batches+1):
        X, y = data_loader.get_batch(args.batch_size)
        with tf.GradientTape() as tape:
            y_pred = model(X)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
            loss = tf.reduce_mean(loss)
            with summary_writer.as_default():
                tf.summary.scalar("loss", loss, step=batch_index)
            # print('batch %d, loss %f' % (batch_index, loss.numpy()))
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
        if batch_index % 100 == 0:
            path = checkpoint.save('./save/model.ckpt')
            print('Model saved to %s' % path)
    with summary_writer.as_default():
        tf.summary.trace_export(name='model_trace', step=0, profiler_outdir='./tensorboard')

def test():
    model_to_restored = MLP()
    checkpoint = tf.train.Checkpoint(myAwesomeModel=model_to_restored)
    checkpoint.restore(tf.train.latest_checkpoint('./save/model.ckpt'))
    y_pred = np.argmax(model_to_restored.predict(data_loader.test_data), axis=-1)
    print("test accuracy: %f" % (sum(y_pred == data_loader.test_label) / data_loader.num_test_data))

if __name__ == "__main__":
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test()
