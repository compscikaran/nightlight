import numpy as np
import rawpy
import colour
import cv2
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear, demosaicing_CFA_Bayer_DDFAPD, demosaicing_CFA_Bayer_Malvar2004, demosaicing_CFA_Bayer_Menon2007

OETF = colour.OETFS['sRGB']

def bayer_to_rgba(filename):
    im = rawpy.imread(filename)
    image = im.raw_image_visible.astype(np.float32)
    x = demosaicing_CFA_Bayer_bilinear(image)
    r_channel, g_channel, b_channel = cv2.split(x)
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype)
    img_RGBA = cv2.merge((r_channel, g_channel, b_channel, alpha_channel))
    y = [img_RGBA/np.max(img_RGBA)]
    return np.array(y)
    

