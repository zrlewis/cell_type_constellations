class Connection(object):

    def __init__(
            self,
            src_centroid,
            dst_centroid,
            src_neighbors,
            dst_neighbors):

        self.src = src_centroid
        self.dst = dst_centroid
        self.src_neighbors = src_neighbors
        self.dst_neighbors = dst_neighbors

    @property
    def x_values(self):
        return []

    @property
    def y_values(self):
        return []
