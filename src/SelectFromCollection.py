"""Small lasso selector used when drawing an irregular domain interactively."""

import numpy as np
from matplotlib import path
from matplotlib.widgets import LassoSelector


class SelectFromCollection:
    """Select scatter points inside a freehand polygon with the mouse."""

    def __init__(self, ax, collection):
        self.collection = collection
        self.xys = collection.get_offsets()
        self.ind = np.array([], dtype=int)
        self.selector = LassoSelector(ax, onselect=self.onselect)

    def onselect(self, vertices):
        self.ind = np.flatnonzero(path.Path(vertices).contains_points(self.xys))
        self.collection.set_facecolors("tab:blue")
        colors = self.collection.get_facecolors()
        if len(colors) == 1:
            colors = np.tile(colors, (len(self.xys), 1))
        colors[self.ind] = (1, 0.2, 0.2, 1)
        self.collection.set_facecolors(colors)
        self.collection.figure.canvas.draw_idle()

    def disconnect(self):
        self.selector.disconnect_events()

