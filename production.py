import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as sl
import rawpy
import glob
import os
import gc
import imageio
import sys

tf.reset_default_graph()

sess = tf.Session()
input_image = tf.placeholder(tf.float32, [None, None, None, 4])

# Custom Layer
def upsample_and_concat(c1, c2, output_channels, in_channels):
    pool_size = 2
    dcf = tf.Variable(tf.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02))
    dc = tf.nn.conv2d_transpose(c1, dcf, tf.shape(c2), strides=[1, pool_size, pool_size, 1])

    output = tf.concat([dc, c2], 3)
    output.set_shape([None, None, None, output_channels * 2])

    return output

# Custom Activation
def leaky_relu(x):
    return tf.maximum(x * 0.2, x) 

# Network
c1 = sl.conv2d(input_image, 32,[3,3], activation_fn=leaky_relu)
c1 = sl.conv2d(c1, 32,[3,3], activation_fn=leaky_relu)
p1 = sl.max_pool2d(c1, [2,2], padding='SAME')
# Unit 2
c2 = sl.conv2d(p1, 64,[3,3], activation_fn=leaky_relu)
c2 = sl.conv2d(c2, 64,[3,3], activation_fn=leaky_relu)
p2 = sl.max_pool2d(c2, [2,2], padding='SAME')
# Unit 3
c3 = sl.conv2d(p2, 128,[3,3], activation_fn=leaky_relu)
c3 = sl.conv2d(c3, 128,[3,3], activation_fn=leaky_relu)
p3 = sl.max_pool2d(c3, [2,2], padding='SAME')
# Unit 4
c4 = sl.conv2d(p3, 256,[3,3], activation_fn=leaky_relu)
c4 = sl.conv2d(c4, 256,[3,3], activation_fn=leaky_relu)
p4 = sl.max_pool2d(c4, [2,2], padding='SAME')
# Unit 5
c5 = sl.conv2d(p4, 512,[3,3], activation_fn=leaky_relu)
c5 = sl.conv2d(c5, 512,[3,3], activation_fn=leaky_relu)
# Unit 6
uc6 = upsample_and_concat(c5,c4,256,512)
c6 = sl.conv2d(uc6, 256, [3,3], activation_fn=leaky_relu)
c6 = sl.conv2d(c6, 256, [3,3], activation_fn=leaky_relu)
# Unit 7
uc7 = upsample_and_concat(c6,c3,128,256)
c7 = sl.conv2d(uc7, 128, [3,3], activation_fn=leaky_relu)
c7 = sl.conv2d(c7, 128, [3,3], activation_fn=leaky_relu)
# Unit 8
uc8 = upsample_and_concat(c7,c2,64,128)
c8 = sl.conv2d(uc8, 64, [3,3], activation_fn=leaky_relu)
c8 = sl.conv2d(c8, 64, [3,3], activation_fn=leaky_relu)
# Unit 9
uc9 = upsample_and_concat(c8,c1,32,64)
c9 = sl.conv2d(uc9, 32, [3,3], activation_fn=leaky_relu)
c9 = sl.conv2d(c9, 32, [3,3], activation_fn=leaky_relu)
# Final Unit
c10 = sl.conv2d(c9, 12, [1,1], activation_fn=None)
output_image = tf.depth_to_space(c10,2)


sess.run(tf.global_variables_initializer())

# Resizer
def pack_raw(raw, black_level):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - black_level, 0) / (16383 - black_level)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out

# CLI
patch_size = 1400
filename = sys.argv[1]
savename = sys.argv[2]
black_level = int(sys.argv[3])
raw = rawpy.imread(filename)
resized = np.expand_dims(pack_raw(raw, black_level), axis=0) * 300
H = resized.shape[1]
W = resized.shape[2]
xx = np.random.randint(0, W - patch_size)
yy = np.random.randint(0, H - patch_size)
input_crop = resized[:, yy:yy + patch_size, xx:xx + patch_size, :]
input_full = np.minimum(input_crop, 1.0)

# Restore trained model
saver = tf.train.Saver()
saver.restore(sess, "models/my-test-model6.ckpt")

# Run forward pass to get output
output = sess.run([output_image], feed_dict={ input_image: input_full})
output = np.minimum(np.maximum(output, 0), 1)
output = output[0,0,:,:,:]
render = output*255
img = render.astype(np.uint8)
imageio.imwrite('images/'+ savename + '.png', img)
