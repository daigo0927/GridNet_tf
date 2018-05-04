import sys, os
import torch
import argparse
import numpy as np

import tensorflow as tf

from torch.utils import data

from model import GridNet
from loss import sparse_softmax_cross_entropy2d
from utils import show_progress

from ptsemseg.loader import get_loader
from ptsemseg.augmentations import *

import pdb

class Trainer(object):

    def __init__(self, args):
        self.args = args
        self.sess = tf.Session()
        self._build_dataloader()
        self._build_graph()

    def _build_dataloader(self):
        # dataset config
        data_aug = Compose([RandomRotate(10),
                            RandomHorizontallyFlip()])
        data_loader = get_loader(args.dataset)
        data_path = args.dataset_dir
        t_loader = data_loader(data_path, is_transform = True,
                                    img_size = (args.img_rows, args.img_cols),
                                    augmentations = data_aug, img_norm = args.img_norm)
        v_loader = data_loader(data_path, is_transform = True,
                                    split = 'validation',
                                    img_size = (args.img_rows, args.img_cols),
                                    img_norm = args.img_norm)
        self.n_classes = t_loader.n_classes
        self.num_batches = int(len(t_loader.files['training'])/args.batch_size)
        self.trainloader = data.DataLoader(t_loader, batch_size = args.batch_size, shuffle = True)
        self.valloader = data.DataLoader(v_loader, batch_size = args.batch_size)

    def _build_graph(self):
        
        self.images = tf.placeholder(tf.float32,
                                     shape = (None, self.args.img_rows, self.args.img_cols, 3))
        self.labels = tf.placeholder(tf.int32,
                                     shape = (None, self.args.img_rows, self.args.img_cols))
        self.model = GridNet(filters_out = self.n_classes, filters_level = (32, 64, 96))
        self.logits = self.model(self.images)

        self.loss, self.accuracy = sparse_softmax_cross_entropy2d(self.labels, self.logits, name = 'loss')

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
            for i, (images, labels) in enumerate(self.trainloader):
                images = np.transpose(images.numpy(), axes = (0, 2, 3, 1)) # transpose to channel last
                labels = labels.numpy()

                _, loss, acc = self.sess.run([self.optimizer, self.loss, self.accuracy],
                                             feed_dict = {self.images : images,
                                                          self.labels : labels})
                if i%10 == 0:
                    show_progress(e+1, i+1, self.num_batches, loss, acc)

            loss_vals, acc_vals = [], []
            for images_val, labels_val in self.valloader:
                images_val = np.transpose(images_val.numpy(), axes = (0, 2, 3, 1))
                labels_val = labels_val.numpy() # sparse (not onehot) labels
                loss_val, acc_val = self.sess.run([self.loss, self.accuracy],
                                                  feed_dict = {self.images : images_val,
                                                               self.labels : labels_val})
                loss_vals.append(loss_val)
                acc_vals.append(acc_val)
            print(f'{e} epoch validation loss: {np.mean(loss_vals)}, acc: {np.mean(acc_vals)}')

            if not os.path.exists('./model_tf'):
                os.mkdir('./model_tf')
            self.saver.save(self.sess, f'./model_tf/model_{e}.ckpt')
                

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
    
    parser.add_argument('--n_epoch', nargs = '?', type = int, default = 100,
                        help = '# of epochs')
    parser.add_argument('--batch_size', nargs = '?', type = int, default = 8,
                        help = 'Batch size')

    parser.add_argument('--resume', nargs = '?', type = str, default = None,
                        help = 'Path to previous saved model to restart from')

    args = parser.parse_args()
    for key, item in vars(args).items():
        print(f'{key} : {item}')

    os.environ['CUDA_VISIBLE_DEVICES'] = input('Input utilize gpu id (-1:cpu) : ')
        
    trainer = Trainer(args)
    trainer.train()
    

























                    
