import numpy as np
import rawpy
import colour
import cv2
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear, demosaicing_CFA_Bayer_DDFAPD, demosaicing_CFA_Bayer_Malvar2004, demosaicing_CFA_Bayer_Menon2007

OETF = colour.OETFS['sRGB']

def bayer_to_rgba(filename):
    im = rawpy.imread(filename)
    image = im.raw_image_visible.astype(np.float32)
    print('bayer',image)
    rgb = demosaicing_CFA_Bayer_Malvar2004(image)
    print('rgb', rgb)
    r, g, b = cv2.split(rgb)
    alpha = np.ones(r.shape, dtype=r.dtype)
    rgba = cv2.merge((r, g, b, alpha))
    rgba = rgba/np.max(rgba)
    # rgba = rgba * 255
    y = [rgba]
    return np.array(y)    

