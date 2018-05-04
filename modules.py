import tensorflow as tf

class LateralBlock(object):

    def __init__(self, filters, shortcut_conv = False, name = 'lateral'):
        self.filters = filters
        self.shortcut_conv = shortcut_conv
        self.name = name
            
    def __call__(self, x):
        
        with tf.variable_scope(self.name) as vs:
            
            fx = tf.keras.layers.PReLU()(x)
            fx = tf.layers.Conv2D(self.filters, kernel_size = (3, 3),
                                 padding = 'same')(fx)
            fx = tf.keras.layers.PReLU()(fx)
            fx = tf.layers.Conv2D(self.filters, kernel_size = (3, 3),
                                 padding = 'same')(fx)
            
            if self.shortcut_conv: # should be called first or final lateral block
                x = tf.layers.Conv2D(self.filters, kernel_size = (3, 3),
                                     padding = 'same')(x)
            return fx + x

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]
                
class DownSamplingBlock(object):

    def __init__(self, filters, name = 'down'):
        self.filters = filters
        self.name = name

    def __call__(self, x):

        with tf.variable_scope(self.name) as vs:

            fx = tf.keras.layers.PReLU()(x)
            fx = tf.layers.Conv2D(self.filters, kernel_size = (3, 3),
                                  strides = (2, 2), padding = 'same')(fx)
            fx = tf.keras.layers.PReLU()(fx)
            fx = tf.layers.Conv2D(self.filters, kernel_size = (3, 3),
                                  padding = 'same')(fx)
            return fx

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

class UpSamplingBlock(object):

    def __init__(self, filters, name = 'up'):
        self.filters = filters
        self.name = name

    def __call__(self, x):

        with tf.variable_scope(self.name) as vs:

            # x_shape = tf.shape(x)
            # fx = tf.image.resize_bilinear(x, size = (x_shape[1]*2, x_shape[2]*2))
            # fx = tf.image.resize_bilinear(x, size = (x_shape[1]*2, x_shape[2]*2))
            fx = tf.keras.layers.UpSampling2D(size = (2, 2))(x)
            fx = tf.keras.layers.PReLU()(fx)
            fx = tf.layers.Conv2D(self.filters, kernel_size = (3, 3),
                                  padding = 'same')(fx)
            fx = tf.keras.layers.PReLU()(fx)
            fx = tf.layers.Conv2D(self.filters, kernel_size = (3, 3),
                                  padding = 'same')(fx)
            return fx

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]
