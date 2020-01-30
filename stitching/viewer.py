import logging

import pyqtgraph as pg

__all__ = ["Viewer"]

logger = logging.getLogger(__name__)


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
        self._viewbox = vb

    ##

    @property
    def viewbox(self):
        return self._viewbox

    @property
    def window(self):
        return self._window

    ##

    def add_image(self, image):
        handle = pg.ImageItem(image)
        handle.setOpts(axisOrder="row-major")

        self.viewbox.addItem(handle)

        return handle

    def hide(self):
        self.window.hide()

    def show(self):
        self.window.show()

    def update(self):
        pg.QtGui.QApplication.processEvents()
