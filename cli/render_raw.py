import rawpy
import sys
import imageio
import numpy as np
from core.utilities import render_raw

image = sys.argv[1]
savename = sys.argv[2]
render_raw(image, savename)