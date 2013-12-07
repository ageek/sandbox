from numpy import shape, ones, sqrt, vstack, hstack, reshape

def pad(A, width=1, v=0):
    '''Adds a padding of size width of value v around A'''
    # if A is a 1D array, convert to shape (1, d)
    if len(shape(A))==1: A.shape = (1, len(A))
    m,n = A.shape
    pA = ones((m + 2*width, n + 2*width)) * v
    pA[width:-width, width:-width] = A
    return pA

def create_image_matrix(imgs, width=None, pad_width=1):
    num_imgs = len(imgs)
    if width is None:
        width = int(sqrt(len(imgs)))
        while num_imgs % width != 0:
            width += 1
    height = num_imgs / width
    padded_imgs = [pad(img, pad_width) for img in imgs]
    h_imgs = []
    for i in xrange(height):
        h_imgs.append(hstack(padded_imgs[width*i:width*(i+1)]))
    return vstack(h_imgs)

def plot_data(data, height=28, width=28):
    pts = [reshape(pt,[height,width]) for pt in data]
    return create_image_matrix(pts)
