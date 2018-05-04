import tensorflow as tf
import pdb

def sparse_softmax_cross_entropy2d(labels2d, logits2d, name = 'loss'):
    
    # shape_labels = tf.shape(labels2d) # (#batches, h, w)
    # shape_logits = tf.shape(logits2d) # (#batches, h, w, #classes)
    # for i, s_labels in enumerate(shape_labels):
    #     assert s_labels == shape_logits[i], 'labels and logits must have same shapes'

    # n_classes = shape_logits[-1] # channel last
    n_classes = tf.shape(logits2d)[-1]
    
    with tf.variable_scope(name) as vs:
        
        labels = tf.reshape(labels2d, [-1]) # shape(#batchesxhxw)
        labels_onehot = tf.one_hot(labels, depth = n_classes,
                                   dtype = tf.float32) # shape(#batchesxhxw, #classes)

        logits = tf.reshape(logits2d, [-1, n_classes]) # shape(#batchesxhxw, #classes)

        loss = tf.losses.softmax_cross_entropy(labels_onehot, logits)
        
        preds = tf.nn.softmax(logits, axis = 1)
        accuracy = tf.reduce_mean(tf.reduce_sum(labels_onehot*preds,
                                                axis = 1))
        return loss, accuracy
        
        
