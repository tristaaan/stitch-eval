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


class Fiducial_corners(object):
    def __init__(self, shape, initial_transform=None):
        h,w = shape
        self.center = Point(w//2, h//2)
        self.corners = []
        self.corners.append(Point(0,0)) # tl
        self.corners.append(Point(w,0)) # tr
        self.corners.append(Point(w,h)) # br
        self.corners.append(Point(0,h)) # bl

        self.reassign_pts()
        if initial_transform:
            self.transform(*initial_transform)

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

    def transform(self, _x, _y, theta, unit='radians', temp_center=None):
        if unit[0] == 'd':
            theta = theta * math.pi/180
        if temp_center:
            cx,cy = temp_center
        else:
            cx,cy = self.center
        new_corners = []
        # rotate
        if theta != 0:
            min_x = np.inf
            min_y = np.inf
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
                min_x = min(nx, min_x)
                min_y = min(ny, min_y)
            # the points should not be less than 0
            if not temp_center:
                _x += -1 * min_x
                _y += -1 * min_y
            self.corners = new_corners
        # translate
        new_corners = []
        for pt in self.corners:
            nx = pt.x + _x
            ny = pt.y + _y
            new_corners.append(Point(nx,ny))
        self.center.x += _x
        self.center.y += _y
        self.corners = new_corners
        self.reassign_pts()
        return self

def group_transform(group, _x, _y, theta, temp_center=None, unit='radians'):
    min_x = np.inf
    min_y = np.inf
    for f in group:
        f.transform(0, 0, theta, unit=unit, temp_center=temp_center)
        for c in f.corners:
            mx,my = c
            min_x = min(mx, min_x)
            min_y = min(my, min_y)
    min_x *= -1
    min_y *= -1
    for f in group:
        f.transform(_x+min_x, _y+min_y, 0)
