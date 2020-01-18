import pyqtgraph as pg

__all__ = ["Viewer"]


class Viewer(object):
    def __init__(self, title="Preview"):
        # create window
        window = pg.GraphicsLayoutWidget()
        window.setWindowTitle(title)
        self._window = window

        # create viewbox
        vb = window.addViewBox()
        vb.setAspectLocked()
        vb.invertY()
        vb.enableAutoRange()

    ##

    @property
    def viewbox(self):
        return self._viewbox

    @property
    def window(self):
        return self._window

    ##

    def set_visible(self, flag):
        self.window.setVisible(flag)
