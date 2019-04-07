import rawpy
import sys
import imageio
import numpy as np
image = sys.argv[1]
filename = sys.argv[2]
im = rawpy.imread(image)
rgb = im.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
imageio.imwrite('files/' + filename + '.png', rgb)