import theano.tensor as T


def conv2d_same(image, filters, filter_size, **kwargs):
    # Borrowed from Lasagne
    # https://github.com/Lasagne/Lasagne/blob/master/lasagne/layers/conv.py#L465-L474
    conved = T.nnet.conv2d(image, filters,
                           border_mode='full', **kwargs)
    crop_x = filter_size[0] // 2
    crop_y = filter_size[1] // 2
    return conved[:, :, crop_x:-crop_x or None,
                  crop_y:-crop_y or None]