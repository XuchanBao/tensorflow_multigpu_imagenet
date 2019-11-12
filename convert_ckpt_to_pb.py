import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util

from architectures.model import model
from architectures.common import SAVE_VARIABLES
import utils
from images.caffe_classes import class_names


save_model_filename = 'epoch-1_1'
checkpoint_path = 'experiments/googlenet/checkpoints/{}'.format(save_model_filename)
save_pb_dir = 'experiments/googlenet/pb'
os.makedirs(save_pb_dir, exist_ok=True)

# -------------- Prepare images --------------------------

# mean of imagenet dataset in BGR
imagenet_mean = np.array([104., 117., 124.], dtype=np.float32)

image_dir = 'images'

image_files = []
for f in os.listdir(image_dir):
    if f.lower().endswith('.jpeg'):
        image_files.append(os.path.join(image_dir, f))

images = []
for f in image_files:
    img = cv2.imread(f)
    img = cv2.resize(img.astype(np.float32), (224, 224))
    img -= imagenet_mean

    images.append(img)

images = np.array(images)
print("images.shape =", images.shape)

# ---------- Initialize Googlenet -------------------

images_ph = tf.placeholder(tf.float32, shape=(None,) + (224, 224, 3), name='input')
labels_ph = tf.placeholder(tf.int32, shape=(None), name='label')

# is_training always False
is_training_ph = tf.constant(False, dtype=tf.bool, name='is_training')

# epoch number
epoch_number = tf.get_variable('epoch_number', [], dtype=tf.int32, initializer=tf.constant_initializer(0),
                               trainable=False, collections=[tf.GraphKeys.GLOBAL_VARIABLES, SAVE_VARIABLES])
global_step = tf.get_variable('global_step', [], dtype=tf.int32, initializer=tf.constant_initializer(0),
                              trainable=False, collections=[tf.GraphKeys.GLOBAL_VARIABLES, SAVE_VARIABLES])

# Weight Decay policy
wd = utils.get_policy('piecewise_linear', '30, 0.0005, 0.0')

# Learning rate decay policy (if needed)
lr = utils.get_policy('piecewise_linear', '19, 30, 44, 53, 0.01, 0.005, 0.001, 0.0005, 0.0001')

# Create an optimizer that performs gradient descent.
optimizer = utils.get_optimizer('Adam', lr)

dnn_model = model(images_ph, labels_ph, utils.loss, optimizer, wd,
                  architecture='googlenet', num_classes=1000,
                  is_training=is_training_ph, transfer_mode=[0],
                  num_gpus=0, max_to_keep=12)

# convert the model in inference mode
with tf.device('/cpu:0'):
    logits = dnn_model.inference()

# Build an initialization operation to run below.
init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

saver = tf.train.Saver(tf.get_collection(SAVE_VARIABLES))

with tf.Session() as sess:
    sess.run(init)

    saver.restore(sess, checkpoint_path)

    result_logits = sess.run(logits, feed_dict={images_ph: images})

    class_inds = np.argmax(result_logits, axis=-1)

    for class_i in class_inds:
        print(class_names[class_i])

    # dump .pb file
    print("graph def size:", sess.graph_def.ByteSize())
    with gfile.GFile("{}/{}.pb".format(save_pb_dir, save_model_filename), 'wb') as f:
        f.write(sess.graph_def.SerializeToString())
    # convert graph to constants
    output_graph_def = graph_util.convert_variables_to_constants(
      sess,
      sess.graph_def,
      ['output/xw_plus_b'])
    with gfile.GFile("{}/{}_frozen.pb".format(save_pb_dir, save_model_filename), 'wb') as f:
        f.write(output_graph_def.SerializeToString())
