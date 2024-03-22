import gym
import tensorflow as tf
import numpy as np
import random
from collections import deque



class QNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(units=24, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=24, activation=tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(units=2)
    
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        output = self.dense3(x)
        return output 
    
    def predict(self, inputs):
        q_value = self(inputs)
        return tf.argmax(q_value, axis=-1)

if __name__ == "__main__":
    num_episodes = 500
    num_exploration_episodes = 100
    max_len_episode = 1000
    batch_size = 32
    learning_rate = 1e-3
    gamma = 1.
    initial_epsilon = 1.
    final_epsilon = 0.01

    env = gym.make('CartPole-v1', render_mode = "human")
    model = QNetwork()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    replay_buffer = deque(maxlen=10000)
    epsilon = initial_epsilon
    for episode_id in range(num_episodes):
        state, _ = env.reset()
        epsilon = max(initial_epsilon * (num_exploration_episodes - episode_id) / num_exploration_episodes, final_epsilon)  # 1->0.01
        for t in range(max_len_episode):
            env.render()
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = model.predict(np.expand_dims(state, axis=0)).numpy()
                action = action[0]
            next_state, reward, done, _, info = env.step(action)
            reward = -10. if done else reward
            replay_buffer.append((state, action, reward, next_state, 1 if done else 0))
            state = next_state

            if done:
                print("episode %d, epsilon %fm score %d" % (episode_id, epsilon, t))
                break
            if len(replay_buffer) >= batch_size:
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*random.sample(replay_buffer, batch_size))
                batch_state, batch_reward, batch_next_state, batch_done = [np.array(a, dtype=np.float32) for a in [batch_state, batch_reward, batch_next_state, batch_done]]
                batch_action = np.array(batch_action, dtype=np.int32)
                q_value = model(batch_next_state)
                y = batch_reward + (gamma*tf.reduce_max(q_value, axis=1)) * (1-batch_done)
                with tf.GradientTape() as tape:
                    loss = tf.keras.losses.mean_squared_error(y_true=y,
                                                              y_pred=tf.reduce_sum(model(batch_state) * tf.one_hot(batch_action, depth=2), axis=1))
                grads = tape.gradient(loss, model.variables)
                optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
                    