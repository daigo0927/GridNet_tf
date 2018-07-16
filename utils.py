import sys
import matplotlib
matplotlib.use('Agg')
from pylab import box
import matplotlib.pyplot as plt

def show_progress(epoch, batch, batch_total, loss, acc):
    sys.stdout.write(f'\r{epoch} epoch: [{batch}/{batch_total}, loss: {loss}, acc: {acc}]')
    sys.stdout.flush()

def vis_semseg(image, label2d_pred, label2d, filename = None):
    fig = plt.figure(figsize = (12, 40))

    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.tick_params(labelbottom = False, bottom = False)
    plt.tick_params(labelleft = False, left = False)
    plt.xticks([])
    box(False)

    plt.subplot(1, 3, 2)
    plt.imshow(label2d_pred)
    plt.tick_params(labelbottom = False, bottom = False)
    plt.tick_params(labelleft = False, left = False)
    plt.xticks([])
    box(False)

    plt.subplot(1, 3, 3)
    plt.imshow(label2d)
    plt.tick_params(labelbottom = False, bottom = False)
    plt.tick_params(labelleft = False, left = False)
    plt.xticks([])
    box(False)

    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, bbox_inches = 'tight', pad_inches = 0.1)
    plt.close()
