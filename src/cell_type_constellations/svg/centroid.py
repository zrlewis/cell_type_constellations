class Centroid(object):

    def __init__(
            self,
            x,
            y,
            n_cells,
            color):

        self._x = x
        self._y = y
        self._color = color
        self._n_cells = n_cells

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def color(self):
        return self._color

    @property
    def n_cells(self):
        return self._n_cells

    @property
    def x_values(self):
        return [self._x]

    @property
    def y_values(self):
        return [self._y]
