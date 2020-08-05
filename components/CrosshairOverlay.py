#TODO fix imports
#TODO fix parameters
import pyqtgraph as pg

class CrosshairOverlay(pg.CrosshairROI):
    def __init__(self, pos=None, index=None, size=None, **kargs):
        self._shape = None
        pg.ROI.__init__(self, pos, size, **kargs)
        self.sigRegionChanged.connect(self.invalidate)
        self.aspectLocked = True
        self.index = index

    def get_index(self):
        return self.index
