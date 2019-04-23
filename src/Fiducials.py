import math
import numpy as np

class Point(object):
    def __init__(self,x,y):
        self.x = x
        self.y = y

    def __iter__(self):
        yield self.x
        yield self.y

    def __repr__(self):
        return '(%f,%f)' % (self.x, self.y)

    def __add__(self, pt):
        if isinstance(pt, Point):
            self.x += pt.x
            self.y += pt.y
        elif isinstance(pt, tuple):
            self.x += pt[0]
            self.y += pt[1]
        else:
            raise ValueError('cannot add pt')
        return self


class Fiducial_corners(object):
    def __init__(self, shape, initial_transform=None, unit='radians'):
        h,w = shape
        self.center = Point(w//2, h//2)
        self.corners = []
        self.corners.append(Point(0,0)) # tl
        self.corners.append(Point(w,0)) # tr
        self.corners.append(Point(w,h)) # br
        self.corners.append(Point(0,h)) # bl

        self.reassign_pts()
        if initial_transform:
            self.transform(*initial_transform, unit=unit, zero_out=True)

    def __repr__(self):
        return str(self.corners)

    def min_x(self):
        return min(c.x for c in self.corners)

    def max_x(self):
        return max(c.x for c in self.corners)

    def min_y(self):
        return min(c.y for c in self.corners)

    def max_y(self):
        return max(c.y for c in self.corners)

    def reassign_pts(self):
        '''
        reassign shortcuts to corners
        '''
        self.tl = self.corners[0]
        self.tr = self.corners[1]
        self.br = self.corners[2]
        self.bl = self.corners[3]

    def transform(self, _x, _y, theta, unit='radians', temp_center=None, zero_out=False):
        if unit[0] == 'd':
            theta = theta * math.pi/180
        if temp_center:
            cx,cy = temp_center
        else:
            cx,cy = self.center
        new_corners = []
        # rotate
        if theta != 0:
            for pt in self.corners:
                nx, ny = pt
                cos_t = math.cos(theta)
                sin_t = math.sin(theta)
                # temporary points
                tx = nx - cx
                ty = ny - cy
                # rotate
                nx = tx * cos_t - ty * sin_t
                ny = ty * cos_t + tx * sin_t
                # restore old center
                nx += cx
                ny += cy
                new_corners.append(Point(nx,ny))
            self.corners = new_corners
        # translate
        new_corners = []
        if zero_out:
            _x -= self.min_x()
            _y -= self.min_y()
        for pt in self.corners:
            nx = pt.x + _x
            ny = pt.y + _y
            new_corners.append(Point(nx,ny))
        self.center.x += _x
        self.center.y += _y
        self.corners = new_corners
        self.reassign_pts()
        return self

    def transform_affine(self, M, temp_center=None):
        '''
        Transform the corners by affine matrix M
        (M already accounts for center offsets unlike the other transform func)
        '''
        if M.shape[0] < 3:
            M = np.vstack([M, [0,0,1]])
        if temp_center:
            cx,cy = temp_center
        else:
            cx,cy = self.center
        new_corners = []
        for pt in self.corners:
            x,y = pt
            # transform
            v = np.array([x,y,1])
            res = np.matmul(M, v)
            nx, ny = res[:2]
            new_corners.append(Point(nx,ny))
        self.corners = new_corners
        self.reassign_pts()
        return self


def group_transform(group, _x, _y, theta, unit='radians', temp_center=None):
    for f in group:
        f.transform(_x, _y, theta, unit=unit, temp_center=temp_center)


def group_transform_affine(group, M, temp_center=None):
    for f in group:
        f.transform_affine(M, temp_center=temp_center)

