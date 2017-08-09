""" Finetune the Demon iterative network """

import sys

import tensorflow as tf
from depthmotionnet.train import finetune

weights_dir = os.path.join(examples_dir, '..', 'weights')
sys.path.insert(0, os.path.join(examples_dir, '..', 'python'))

with tf.Session() as session:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # TODO: Load data from somewhere
    image1 = None
    image2 = None
    depth = None
    normals = None
    flow = None
    rotation = None
    translation = None

    # Hyperparams
    max_iterations = 100000
    learning_rate = 0.2E-4

    finetune(session, image1, image2, depth, normals, flow, rotation, translation,
             learning_rate, max_iterations, weights_dir, log_dir)
