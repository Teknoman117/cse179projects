#!/usr/bin/python

import numpy
import matplotlib.pyplot
import pickle
import sys

# Visualize the results from a run
def visualizeW1(opt_W1, vis_patch_side, hid_patch_side):
    figure, axes = matplotlib.pyplot.subplots(nrows = hid_patch_side,
                                              ncols = hid_patch_side)
    index = 0
    for axis in axes.flat:
        image = axis.imshow(opt_W1[index, :].reshape(vis_patch_side, vis_patch_side),
                            cmap = matplotlib.pyplot.cm.gray, interpolation = 'nearest')
        axis.set_frame_on(False)
        axis.set_axis_off()
        index += 1

    matplotlib.pyplot.show()

# Main fucktion
if __name__ == "__main__":
    with open(sys.argv[1], 'rb') as jar:
        opt_W1, vis_patch_size, hid_patch_size = pickle.load(jar)
        visualizeW1(opt_W1, vis_patch_size, hid_patch_size)
