import numpy as np
import exifread

# Convert Bayer to 4 channel RGBa image
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


# Works only on dng files taken using lightroom app
def dng_blacklevel(file):
    f = open(file, 'rb')
    tags = exifread.process_file(f)
    return tags['Image Tag 0xC61A'].values[0].num
