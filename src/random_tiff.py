from tifffile import imsave
import numpy as np

# generate random 2d array
byte_array = np.random.randint(0, 255, size=(46340, 46340), dtype='uint8')
imsave("test.tiff", byte_array)
