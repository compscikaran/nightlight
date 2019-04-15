import numpy as np
import rawpy
import colour
import cv2
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear, demosaicing_CFA_Bayer_DDFAPD, demosaicing_CFA_Bayer_Malvar2004, demosaicing_CFA_Bayer_Menon2007

OETF = colour.OETFS['sRGB']

def bayer_to_rgba(filename):
    im = rawpy.imread(filename)
    rgb = im.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
    rgb = rgb / np.max(rgb)
    r, g, b = cv2.split(rgb)
    alpha = np.ones(r.shape, dtype=r.dtype)
    rgba = cv2.merge((r, g, b, alpha))
    rgba = rgba/np.max(rgba)
    # rgba = rgba * 255
    y = [rgba]
    return np.array(y)    

