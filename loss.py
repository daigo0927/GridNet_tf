import tensorflow as tf

def sparse_softmax_cross_entropy2d(labels2d, logits2d, name = 'loss'):
    with tf.name_scope(name) as ns:
        n_classes = tf.shape(logits2d)[-1]

        labels = tf.reshape(labels2d, [-1]) # shape(#batchesxhxw)
        labels_onehot = tf.one_hot(labels, depth = n_classes, axis = -1,
                                   dtype = tf.float32) # shape(#batchesxhxw, #classes)

        logits = tf.reshape(logits2d, [-1, n_classes]) # shape(#batchesxhxw, #classes)

        loss = tf.losses.softmax_cross_entropy(labels_onehot, logits)
        
        preds = tf.nn.softmax(logits, axis = -1)
        accuracy = tf.reduce_mean(tf.reduce_sum(labels_onehot*preds,
                                                axis = -1))
        labels2d_pred = tf.argmax(logits2d, axis = -1)
        
        return loss, accuracy, labels2d_pred
        
        
