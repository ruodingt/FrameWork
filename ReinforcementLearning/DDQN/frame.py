import tensorflow as tf
import time
import gym


"""
DDQN key equation:

q_true(t) = R(t+1) + gamma * Q(S(t+1), act=argmax(S(t+1),a; theta1), theta0) 


"""


EXPERIENCE_REPLAY = 10000


TARGET_FREEZE_ITERATION = 10


def dqn_graph(state_space_dim, action_space_dim):
    graph = tf.Graph()

    with graph.as_default():

        with tf.name_scope('init'):
            init_op_global = tf.global_variables_initializer()

        with tf.name_scope('feed'):

            x = tf.placeholder(tf.float32, shape=(None, state_space_dim))

            y = tf.placeholder(tf.float32, shape=(None,))

        with tf.name_scope('Qnet'):

            a_onehot = tf.placeholder(tf.int32, shape=(None, action_space_dim))

            h1 = tf.layers.dense(inputs=x, units=action_space_dim*2, activation=tf.nn.relu)

            o = tf.layers.dense(inputs=h1, units=action_space_dim, activation=None)

            act = tf.argmax(o)

        with tf.name_scope('loss'):

            loss = tf.losses.mean_squared_error(tf.boolean_mask(o, a_onehot), y)

    return init_op_global, x, y, o, loss, act



def dueling_dqn_graph(state_space_dim, action_space_dim):
    graph = tf.Graph()

    with graph.as_default():

        with tf.name_scope('init'):
            init_op_global = tf.global_variables_initializer()

        with tf.name_scope('feed'):

            x = tf.placeholder(tf.float32, shape=(None, state_space_dim))

            value = tf.placeholder(tf.float32, shape=(None,))

            advantage = tf.placeholder(tf.float32, shape=(None,))

        with tf.name_scope('Qnet'):

            a_onehot = tf.placeholder(tf.int32, shape=(None, action_space_dim))

            h1 = tf.layers.dense(inputs=x, units=action_space_dim*2, activation=tf.nn.relu)

            o_value = tf.layers.dense(inputs=h1, units=action_space_dim, activation=None)

            o_advantage = tf.layers.dense(inputs=h1, units=action_space_dim, activation=None)

            act = tf.argmax(o_advantage)

        with tf.name_scope('loss'):

            loss = tf.losses.mean_squared_error(tf.boolean_mask(o_advantage, a_onehot), advantage)

    return init_op_global, x, advantage, o_value, o_advantage, loss, act




def train_dueling_dqn():

    env = gym.make('CartPole-v0')
    env.reset()
    for _ in range(1000):
        env.render()
        time.sleep(0.01)
        env.step(env.action_space.sample())