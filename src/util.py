import numpy as np

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
    # if the canvas is smaller than the paint size, pad the image.
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


def tuple_sub(t1, t2):
    '''
    Value-wise subtraction of two tuples.
    '''
    return (t1[0] - t2[0], t1[1] - t2[1])


def crop_zeros(im, zero=0):
    '''
    Crop zeros around an image
    '''
    r,c = im.shape
    top = 0
    left = 0
    bottom = r-1
    right = c-1
    while sum(im[top,:]) == zero:
        top += 1

    while sum(im[bottom,:]) == zero:
        bottom -=1

    while sum(im[:,left]) == zero:
        left +=1

    while sum(im[:,right]) == zero:
        right -=1

    return im[top:bottom+1,left:right+1]


def clamp_uint16(val):
  return min(val, (2**16)-1)


def bounds_equalize(target, ref):
    '''
    Given two images, target larger than ref, 0-pad ref so that
    ref becomes the same size as target
    '''
    yd,xd = tuple_sub(target.shape, ref.shape)
    if yd < 0:
        ref = ref[-yd//2:yd//2,:]
    else:
        ref = np.pad(ref, ((yd//2,yd//2), (0,0)), 'constant', constant_values=(0,0))

    if xd < 0:
        ref = ref[:,-xd//2:xd//2]
    else:
        ref = np.pad(ref, ((0,0), (xd//2,xd//2)), 'constant', constant_values=(0,0))

    # there can be some off-by-one errors from division,
    # make sure they're the same size
    if sum(tuple_sub(target.shape, ref.shape)) != 0:
        py = target.shape[0] - ref.shape[0] if target.shape[0] > ref.shape[0] else 0
        px = target.shape[1] - ref.shape[1] if target.shape[1] > ref.shape[1] else 0
        ref = np.pad(ref, ((0,py),(0,px)), 'constant', constant_values=(0,0))

    return ref
