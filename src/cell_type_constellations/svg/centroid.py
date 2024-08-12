class Centroid(object):

    def __init__(
            self,
            x,
            y,
            n_cells,
            color,
            label,
            name):

        self._x = x
        self._y = y
        self._color = color
        self._n_cells = n_cells
        self._name = name
        self._label = label

        self._pixel_coords = None

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

    @property
    def name(self):
        return self._name

    @property
    def label(self):
        return self._label

    def set_pixel_coords(self, x, y, radius):
        self._pixel_coords = {
            'x': x,
            'y': y,
            'radius': radius
        }
