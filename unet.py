
import tensorflow as tf

def _conv_relu(filters, kernel_size = (3, 3), strides = (1, 1)):
    def f(inputs):
        x = tf.layers.Conv2D(filters, kernel_size, strides, 'same')(inputs)
        x = tf.nn.relu(x)
        return x
    return f

def _conv_bn_relu(filters, kernel_size = (3, 3), strides = (1, 1)):
    def f(inputs):
        x = tf.layers.Conv2D(filters, kernel_size, strides, 'same')(inputs)
        x = tf.layers.BatchNormalization()(x)
        x = tf.nn.relu(x)
        return x
    return f

def down_block(block_fn, filters):
    def f(inputs):
        x = block_fn(filters)(inputs)
        x = block_fn(filters)(x)
        down = tf.layers.MaxPooling2D((3, 3), (2, 2), 'same')(x)
        return x, down # x:same size of inputs, down: downscaled
    return f

def up_block(block_fn, filters):
    def f(inputs, down):
        inputs_ = tf.layers.Conv2DTranspose(filters, (3, 3), (2, 2), 'same')(down)
        x = tf.concat([inputs, inputs_], axis = 3)
        x = block_fn(filters)(x)
        x = block_fn(filters)(x)
        return x # same size of 'inputs'
    return f

class U_Net(object):
    def __init__(self,
                 output_ch, # outputs channel, size is same as inputs
                 batch_norm = True,
                 name = 'unet'):
        self.output_ch = output_ch
        self.name = name

        if batch_norm:
            self.block_fn = _conv_bn_relu
        else:
            self.block_fn = _conv_relu
            
    def __call__(self, images):
        with tf.variable_scope(self.name) as vs:
            
            x1, down1 = down_block(self.block_fn, 64)(images)
            x2, down2 = down_block(self.block_fn, 128)(down1)
            x3, down3 = down_block(self.block_fn, 256)(down2)

            down3 = self.block_fn(512)(down3)

            up3 = up_block(self.block_fn, 256)(x3, down3)
            up2 = up_block(self.block_fn, 128)(x2, up3)
            up1 = up_block(self.block_fn, 64)(x1, up2)

            outputs = tf.layers.Conv2D(self.output_ch, (1, 1), (1, 1), 'same')(up1)
            return outputs

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]
