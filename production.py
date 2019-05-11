import numpy as np
import tensorflow as tf
import rawpy
import glob
import imageio
import sys
from utilities import pack_raw, calculate_black_level
from cnn import network

# Initialize Tensorflow 
tf.reset_default_graph()
sess = tf.Session()
input_image = tf.placeholder(tf.float32, [None, None, None, 4])
output_image = network(input_image)
sess.run(tf.global_variables_initializer())

# CLI input
filename = sys.argv[1]
savename = sys.argv[2]

# Calculate Black Level
black_level = calculate_black_level(filename)

# Read in image
raw = rawpy.imread(filename)
resized = np.expand_dims(pack_raw(raw, black_level), axis=0) * 300
input_full = np.minimum(resized, 1.0)

# Restore trained model
saver = tf.train.Saver()
saver.restore(sess, "models/my-test-model8.ckpt")

# Run forward pass to get output
output = sess.run([output_image], feed_dict={ input_image: input_full})
output = np.minimum(np.maximum(output, 0), 1)
output = output[0,0,:,:,:]

# Render out image
render = output*255
img = render.astype(np.uint8)
imageio.imwrite('images/'+ savename + '.png', img)
