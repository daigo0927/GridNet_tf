import sys, os
import torch
import argparse
import numpy as np

import tensorflow as tf

from torch.utils import data

from unet import U_Net
from loss import sparse_softmax_cross_entropy2d
from utils import show_progress, vis_semseg

from datahandler.utils import get_dataset

import pdb

class Trainer(object):
    def __init__(self, args):
        self.args = args
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config = config)
        self._build_dataloader()
        self._build_graph()

    def _build_dataloader(self):
        # dataset config
        dataset = get_dataset(self.args.dataset)
        data_args = {'dataset_dir':self.args.dataset_dir,
                     'cropper':self.args.crop_type, 'crop_shape':self.args.crop_shape,
                     'resize_shape':self.args.resize_shape, 'resize_scale':self.args.resize_scale}
        self.tset = dataset(train_or_val = 'train', **data_args)
        vset = dataset(train_or_val = 'val', **data_args)
        
        self.n_classes = self.tset.n_classes
        self.num_batches = int(len(self.tset)/self.args.batch_size)
        load_args = {'batch_size':self.args.batch_size, 'num_workers':self.args.num_workers,
                     'pin_memory':True, 'drop_last':True}
        self.tloader = data.DataLoader(self.tset, shuffle = True, **load_args)
        self.vloader = data.DataLoader(vset, shuffle = False, **load_args)

    def _build_graph(self):
        self.images = tf.placeholder(tf.float32, shape = [None]+self.args.image_size+[3],
                                    name = 'images')
        self.labels = tf.placeholder(tf.int32,   shape = [None]+self.args.image_size,
                                    name = 'labels')
        self.model = U_Net(output_ch = self.n_classes, batch_norm = self.args.batch_norm, name = 'unet')
        self.logits = self.model(self.images)

        self.loss, self.accuracy, self.preds \
          = sparse_softmax_cross_entropy2d(self.labels, self.logits, name = 'loss')

        self.optimizer = tf.train.AdamOptimizer()\
                                 .minimize(self.loss, var_list = self.model.vars)
        self.saver = tf.train.Saver()

        if self.args.resume is not None:
            print(f'Loading learned model from checkpoint {self.args.resume}')
            self.saver.restore(self.sess, self.args.resume)
        else:
            self.sess.run(tf.global_variables_initializer())

    def train(self):
        for e in range(args.n_epoch):
            for i, (images, labels) in enumerate(self.tloader):
                images = images.numpy()/255.
                labels = labels.numpy()

                _, loss, acc = self.sess.run([self.optimizer, self.loss, self.accuracy],
                                             feed_dict = {self.images: images,
                                                          self.labels: labels})
                if i%10 == 0:
                    show_progress(e+1, i+1, self.num_batches, loss, acc)

            loss_vals, acc_vals = [], []
            for images_val, labels_val in self.vloader:
                images_val = images_val.numpy()/255.
                labels_val = labels_val.numpy() # sparse (not onehot) labels
                loss_val, acc_val, preds_val = self.sess.run([self.loss, self.accuracy, self.preds],
                                                             feed_dict = {self.images : images_val,
                                                                          self.labels : labels_val})
                loss_vals.append(loss_val)
                acc_vals.append(acc_val)

            print(f'\r{e+1} epoch validation loss: {np.mean(loss_vals)}, acc: {np.mean(acc_vals)}')

            if self.args.visualize:
                if not os.path.exists('./figure_unet'):
                    os.mkdir('./figure_unet')
                image_v = images_val[0]
                pred_v = self.tset.decode_segmap(preds_val[0])
                label_v = self.tset.decode_segmap(labels_val[0])
                vis_semseg(image_v, pred_v, label_v,
                           filename = f'./figure_unet/seg_{str(e+1).zfill(3)}.pdf')
            
            if not os.path.exists('./model_unet'):
                os.mkdir('./model_unet')
            self.saver.save(self.sess, f'./model_unet/model_{e}.ckpt')
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--dataset', type = str, default='CityScapes',
                        help = 'Dataset to use [CityScapes, SYNTHIA, PlayingforData etc]')
    parser.add_argument('--dataset_dir', required = True, type = str,
                        help = 'Directory containing target dataset')
    parser.add_argument('--n_epoch', type = int, default = 50,
                        help = '# of epochs [50]')
    parser.add_argument('--batch_size', type = int, default = 4,
                        help = 'Batch size [4]')
    parser.add_argument('--num_workers', type = int, default = 4,
                        help = '# of workers for data loading [4]')

    parser.add_argument('--crop_type', type = str, default = 'random',
                        help = 'Crop type for raw image data [random]')
    parser.add_argument('--crop_shape', nargs = 2, type = int, default = [512, 1024],
                        help = 'Crop shape for raw image data [512, 1024]')
    parser.add_argument('--resize_shape', nargs = 2, type = int, default = [256, 512],
                        help = 'Resize shape for raw image data [256, 512]')
    parser.add_argument('--resize_scale', nargs = 2, type = int, default = None,
                        help = 'Resize scale for raw image data [None]')
    parser.add_argument('--image_size', nargs = 2, type = int, default = [256, 512],
                        help = 'Image size to be processed [256, 512]')

    parser.add_argument('-bn', '--batch_norm', dest = 'batch_norm', action = 'store_true',
                        help = 'Enable batch normalization, [enabled] as default.')
    parser.add_argument('--no-batch_norm', dest = 'batch_norm', action = 'store_false',
                        help = 'Disable batch normalization, [enabled] as default.')
    parser.set_defaults(batch_norm = True)

    parser.add_argument('-v', '--visualize', dest = 'visualize', action = 'store_true',
                        help = 'Enable to visualize estimated segmentation, [enabled] as default.')
    parser.add_argument('--no-visualize', dest = 'visualize', action = 'store_false',
                        help = 'Disable to visualize estimated segmentation, [enabled] as default.')
    parser.set_defaults(visualize = True)
    parser.add_argument('--resume', type = str, default = None,
                        help = 'Path to previous saved model to restart from [None]')

    args = parser.parse_args()
    for key, item in vars(args).items():
        print(f'{key} : {item}')

    os.environ['CUDA_VISIBLE_DEVICES'] = input('Input utilize gpu id (-1:cpu) : ')
        
    trainer = Trainer(args)
    trainer.train()
    

























                    
