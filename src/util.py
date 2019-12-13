import numpy as np


def pad(im,x,y):
    '''
    pad an image for pasting
    '''
    x = int(x)
    y = int(y)
    if y < 0:
        vert = (0, abs(y))
    else:
        vert = (y, 0)
    if x < 0:
        horz = (0, abs(x))
    else:
        horz = (x, 0)
    return zero_pad(im, vert, horz)


def zero_pad(im, tb, rl):
    '''
    wrapper for zero padding
    '''
    return np.pad(im, (tb, rl), 'constant', constant_values=(0,0))


def paste(canvas, paint):
    '''
    Paste an image on to another using masks.
    '''
    ch, cw = canvas.shape
    ph, pw = paint.shape
    nw, nh = (0,0)
    # if the canvas is smaller than the paint size, pad the image.
    if cw < pw:
        nw = pw - cw
    if ch < ph:
        nh = ph - ch
    output = np.copy(canvas)
    output = np.pad(output, ((0, nh), (0, nw)), 'constant', constant_values=(0,0))

    # mask magic! This is very fast considering the size of what's being merged
    o_mask = np.ma.equal(output, 0)
    output[o_mask] = paint[o_mask] # for places where output is 0: paint.
    return output


def eq_paste(canvas, paint):
    '''
    Similar to paste, but resizes paint if necessary
    '''
    ch, cw = canvas.shape
    ph, pw = paint.shape
    nw, nh = (0,0)
    # if the canvas is smaller than the paint size, pad the output image.
    if cw < pw:
        nw = pw - cw
    if ch < ph:
        nh = ph - ch
    output = np.copy(canvas)
    output = np.pad(output, ((0, nh), (0, nw)), 'constant', constant_values=(0,0))
    ch, cw = output.shape

    # if the paint is not the right size adjust it
    nw, nh = (0,0)
    if pw < cw:
        nw = cw - pw
    if ph < ch:
        nh = ch - ph
    paint = np.pad(paint, ((0,nh), (0, nw)), 'constant', constant_values=(0,0))

    # mask magic! This is very fast considering the size of what's being merged
    o_mask = np.ma.equal(output, 0)
    output[o_mask] = paint[o_mask] # for places where output is 0: paint.
    return output


def merge(ref, mov, x, y, zero=0):
    rx, ry = (-x,-y)
    mx, my = (x,y)

    ref = pad(ref, rx, ry)
    mov = pad(mov, mx, my)
    merged = eq_paste(ref, mov)
    return crop_zeros(merged, zero=zero)


def tuple_sub(t1, t2):
    '''
    Value-wise subtraction of two tuples.
    '''
    return (t1[0] - t2[0], t1[1] - t2[1])


def crop_zeros(im, zero=0, crop_vals=False):
    '''
    Crop zeros around an image
    '''
    # get the coordinates of every point > zero
    true_points = np.argwhere(im > zero)
    tl = true_points.min(axis=0) # top left corner
    br = true_points.max(axis=0) # bottom right corner
    out = im[tl[0]:br[0]+1,
             tl[1]:br[1]+1]
    if crop_vals: # top, bottom, left, right
        return out, (tl[0], br[0], tl[1], br[1])
    return out


def clamp_uint16(val):
    return min(val, (2**16)-1)


def uint16_to_uint8(im):
    im = (im / (2**16-1) * 255).astype('uint8')
    return im


def equalize_image(image, target_size):
    '''
    crop or pad an image to a target size
    '''
    sh, sw = image.shape
    th, tw = target_size

    nh, nw = (0,0)
    if sw < tw:
        nw = tw - sw
    elif sw > tw:
        image = image[:,:tw]

    if sh < th:
        nh = th - sh
    elif sh > th:
        image = image[:th,:]

    if nw + nh > 0:
        return zero_pad(image, (0, nh), (0, nw))
    return image


def bounds_equalize(ref, target, padded_vals=False):
    '''
    Given two images, pad them to make them the same size
    padding keeps image in the center.
    '''
    r_pad = [0,0]
    t_pad = [0,0]
    yd,xd = tuple_sub(target.shape, ref.shape)
    if yd < 0:
        target = zero_pad(target, (-yd//2,-yd//2), (0,0))
        t_pad[1] += abs(yd)
    else:
        ref = zero_pad(ref, (yd//2,yd//2), (0,0))
        r_pad[1] += yd

    if xd < 0:
        target = zero_pad(target, (0,0), (-xd//2,-xd//2))
        t_pad[0] += abs(xd)
    else:
        ref = zero_pad(ref, (0,0), (xd//2,xd//2))
        r_pad[0] += xd

    # there can be some off-by-one errors from division, make sure they're the same size
    if target.shape != ref.shape:
        py = target.shape[0] - ref.shape[0] if target.shape[0] > ref.shape[0] else 0
        px = target.shape[1] - ref.shape[1] if target.shape[1] > ref.shape[1] else 0
        ref = zero_pad(ref, (0,py), (0,px))
        r_pad[0] += px
        r_pad[1] += py

        py = ref.shape[0] - target.shape[0] if ref.shape[0] > target.shape[0] else 0
        px = ref.shape[1] - target.shape[1] if ref.shape[1] > target.shape[1] else 0
        target = zero_pad(target, (0,py), (0,px))
        t_pad[0] += px
        t_pad[1] += py

    if padded_vals:
        return ref, target, (tuple(r_pad), tuple(t_pad))
    return ref, target


def ul_equalize(ref, mov):
    '''
    Make two images the same size by padding them on the bottom and right
    '''
    yd,xd = tuple_sub(mov.shape, ref.shape)
    if yd < 0:
        mov = pad(mov, 0, yd)
    elif yd > 0:
        ref = pad(ref, 0, -yd)
    if xd < 0:
        mov = pad(mov, xd, 0)
    elif xd > 0:
        ref = pad(ref, -xd, 0)
    return ref, mov


def square_off(im):
    '''
    bottom/right-pad an image on its shorter side so that it is square
    '''
    which_side = np.argmin(im.shape) # adjust this side
    new_len = max(im.shape)          # find max side
    diff = new_len - im.shape[which_side] # difference between max and min
    yx = [0,0]                       # zero padding
    yx[which_side] = -diff           # add the diff to the minimal side
    return pad(im, *yx[::-1])        # pad
