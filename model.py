import tensorflow as tf

from modules import LateralBlock, DownSamplingBlock, UpSamplingBlock

class GridNet(object):

    def __init__(self, filters_out = 3, filters_level = [32, 64, 96],
                 name = 'gridnet'):
        self.n_row = 3
        self.n_col = 6
        self.f_out = filters_out
        self.f_level = filters_level
        self.name = name

    def __call__(self, x):
        with tf.variable_scope(self.name) as vs:
            
            x_0 = LateralBlock(filters = self.f_level[0], shortcut_conv = True,
                               name = 'lateral_in')(x)
            x_1 = DownSamplingBlock(filters = self.f_level[1], name = 'down_00')(x_0)
            x_2 = DownSamplingBlock(filters = self.f_level[2], name = 'down_10')(x_1)

            for i in range(1, self.n_col):
            
                if i < self.n_col/2:
                    x_0 = LateralBlock(self.f_level[0], name = f'lateral_0{i-1}')(x_0)
                    x_1 = DownSamplingBlock(self.f_level[1], f'down_0{i}')(x_0)\
                          + LateralBlock(self.f_level[1], name = f'lateral_1{i-1}')(x_1)
                    x_2 = DownSamplingBlock(self.f_level[2], f'down_1{i}')(x_1)\
                          + LateralBlock(self.f_level[2], name = f'lateral_2{i-1}')(x_2)
                else:
                    x_2 = LateralBlock(self.f_level[2], name = f'lateral_2{i-1}')(x_2)
                    x_1 = UpSamplingBlock(self.f_level[1], f'up_1{i}')(x_2)\
                          + LateralBlock(self.f_level[1], name = f'lateral_1{i-1}')(x_1)
                    x_0 = UpSamplingBlock(self.f_level[0], f'up_0{i}')(x_1)\
                          + LateralBlock(self.f_level[0], name = f'lateral_0{i-1}')(x_0)

            return LateralBlock(self.f_out, True,
                                'lateral_out')(x_0)

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]
                
