
import tensorflow as tf
import tensorflow.contrib.layers as tcl

def _conv_relu(filters, kernel_size = (3, 3), stride = (1, 1)):
    def f(inputs):
        x = tcl.conv2d(inputs,
                       num_outputs = filters,
                       kernel_size = kernel_size,
                       stride = stride,
                       padding = 'SAME')
        x = tf.nn.relu(x)
        return x
    return f

def _conv_bn_relu(filters, kernel_size = (3, 3), stride = (1, 1)):
    def f(inputs):
        x = tcl.conv2d(inputs,
                       num_outputs = filters,
                       kernel_size = kernel_size,
                       stride = stride,
                       padding = 'SAME')
        x = tcl.batch_norm(x)
        x = tf.nn.relu(x)
        return x
    return f

def down_block(block_fn, filters):
    def f(inputs):
        x = block_fn(filters)(inputs)
        x = block_fn(filters)(x)
        down = tcl.max_pool2d(x,
                              kernel_size = (3, 3),
                              stride = (2, 2),
                              padding = 'SAME')
        return x, down # x:same size of inputs, down: downscaled
    return f

def up_block(block_fn, filters):
    def f(inputs, down):
        inputs_ = tcl.conv2d_transpose(down, # double size of 'down'
                                       num_outputs = inputs.shape.as_list()[3],
                                       kernel_size = (3, 3),
                                       stride = (2, 2),
                                       padding = 'SAME')
        x = tf.concat([inputs, inputs_], axis = 3)
        x = block_fn(filters)(x)
        x = block_fn(filters)(x)
        return x # same size of 'inputs'
    return f

class U_Net(object):
    def __init__(self,
                 output_ch, # outputs channel, size is same as inputs
                 block_fn = 'origin',
                 name = 'unet'):
        self.output_ch = output_ch
        self.name = name

        assert block_fn in ['batch_norm', 'origin'], 'choose \'batch_norm\' or \'origin\''
        if block_fn == 'batch_norm':
            self.block_fn = _conv_bn_relu
        elif block_fn == 'origin':
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
                
            outputs = tcl.conv2d(up1,
                                 num_outputs = self.output_ch,
                                 kernel_size = (1, 1),
                                 stride = (1, 1),
                                 padding = 'SAME')

            return outputs

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]
