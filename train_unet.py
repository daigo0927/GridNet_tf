import sys, os
import torch
import argparse
import numpy as np

import tensorflow as tf

from torch.utils import data

from unet import U_Net
from loss import sparse_softmax_cross_entropy2d
from utils import show_progress, vis_semseg

from ptsemseg.loader import get_loader
from ptsemseg.augmentations import *

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
        daug = Compose([RandomRotate(10),
                        RandomHorizontallyFlip()])
        dset = get_loader(self.args.dataset)
        dpath = self.args.dataset_dir
        tset = dset(dpath, is_transform = True,
                    img_size = (self.args.img_rows, self.args.img_cols),
                    augmentations = daug, img_norm = self.args.img_norm)
        vset = dset(dpath, is_transform = True,
                    split = 'validation',
                    img_size = (self.args.img_rows, self.args.img_cols),
                    img_norm = self.args.img_norm)
        
        self.n_classes = tset.n_classes
        self.num_batches = int(len(tset.files['training'])/self.args.batch_size)
        self.tloader = data.DataLoader(tset, batch_size = self.args.batch_size,
                                       num_workers = 8, shuffle = True)
        self.vloader = data.DataLoader(vset, batch_size = self.args.batch_size,
                                       num_workers = 8)

    def _build_graph(self):
        self.images = tf.placeholder(tf.float32,
                                     shape = (None, self.args.img_rows, self.args.img_cols, 3))
        self.labels = tf.placeholder(tf.int32,
                                     shape = (None, self.args.img_rows, self.args.img_cols))
        self.model = U_Net(output_ch = self.n_classes, block_fn = 'batch_norm', name = 'unet')
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
                images = np.transpose(images.numpy(), axes = (0, 2, 3, 1)) # transpose to channel last
                labels = labels.numpy()

                _, loss, acc = self.sess.run([self.optimizer, self.loss, self.accuracy],
                                             feed_dict = {self.images: images,
                                                          self.labels: labels})
                if i%10 == 0:
                    show_progress(e+1, i+1, self.num_batches, loss, acc)

            loss_vals, acc_vals = [], []
            for images_val, labels_val in self.vloader:
                images_val = np.transpose(images_val.numpy(), axes = (0, 2, 3, 1))
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
                vis_semseg(images_val[0], preds_val[0], labels_val[0],
                           filename = f'./figure_unet/seg_{str(e+1).zfill(3)}.pdf')
            
            if not os.path.exists('./model_unet'):
                os.mkdir('./model_unet')
            self.saver.save(self.sess, f'./model_unet/model_{e}.ckpt')
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--dataset', nargs='?', type=str, default='mit_sceneparsing_benchmark',
                        help = 'Dataset to use [\'pascal, camvid, ade20k etc\']')
    parser.add_argument('--dataset_dir', required = True, type = str,
                        help = 'Directory containing target dataset')
    parser.add_argument('--img_rows', nargs='?', type=int, default=256,
                        help = 'Height of the input')
    parser.add_argument('--img_cols', nargs='?', type=int, default=256,
                        help = 'Width of input')

    parser.add_argument('--img_norm', dest = 'img_norm', action = 'store_true',
                        help = 'Enable input images scales normalization [0, 1] | True by default')
    parser.add_argument('--no-img_norm', dest = 'img_norm', action = 'store_false',
                        help = 'Disable input images scales normalization [0, 1] | True by Default')
    parser.set_defaults(img_norm = True)
    
    parser.add_argument('--n_epoch', nargs = '?', type = int, default = 50,
                        help = '# of epochs')
    parser.add_argument('--batch_size', nargs = '?', type = int, default = 1,
                        help = 'Batch size')

    parser.add_argument('-v', '--visualize', action = 'store_true',
                        help = 'Stored option for visualize estimated segmentation')
    parser.add_argument('--resume', nargs = '?', type = str, default = None,
                        help = 'Path to previous saved model to restart from')

    args = parser.parse_args()
    for key, item in vars(args).items():
        print(f'{key} : {item}')

    os.environ['CUDA_VISIBLE_DEVICES'] = input('Input utilize gpu id (-1:cpu) : ')
        
    trainer = Trainer(args)
    trainer.train()
    

























                    
