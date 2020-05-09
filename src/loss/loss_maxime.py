from tensorflow.python.keras.losses import Loss
from tensorflow.python.keras.losses import BinaryCrossentropy
from tensorflow.python.keras.losses import LossFunctionWrapper
from tensorflow.keras.losses import binary_crossentropy
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.losses import loss_reduction
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.ops.losses import util as tf_losses_utils


class MyCrossentropy(Loss):
    def call(self, y_true, y_pred):
        nbr_classes = len(y_pred)
        size = 1920
        discrete_space = np.array([i / nbr_classes * size for i in range(nbr_classes)])
        pos = np.argmax(discrete_space < y_true[1])
        y_true = np.zeros(nbr_classes)
        y_true[pos] = 1
        return binary_crossentropy(y_true, y_pred)


class MyCrossentropy2:
    def __init__(self,
                 from_logits=False,
                 label_smoothing=0,
                 reduction=losses_utils.ReductionV2.AUTO,
                 name='my_binary_crossentropy'
                 ):
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.name = name

    def __call__(self,
                 y_true,
                 y_pred,
                 sample_weight=None
                 ):
        nbr_classes = len(y_pred)
        size = 1920
        discrete_space = np.array([i / nbr_classes * size for i in range(nbr_classes)])
        pos = np.argmax(discrete_space < y_true[1])
        y_true = np.zeros(nbr_classes)
        y_true[pos] = 1
        return binary_crossentropy(y_true, y_pred, sample_weight)


class MyCrossentropy3(LossFunctionWrapper):
    def __init__(self,
                 from_logits=False,
                 label_smoothing=0,
                 reduction=losses_utils.ReductionV2.AUTO,
                 name='my2_binary_crossentropy'):
        super(MyCrossentropy3, self).__init__(
            binary_crossentropy,
            name=name,
            reduction=reduction,
            from_logits=from_logits,
            label_smoothing=label_smoothing)
        self.from_logits = from_logits

    def __call__(self,
                 y_true,
                 y_pred,
                 sample_weight=None
                 ):
        nbr_classes = len(y_pred)
        size = 1920
        discrete_space = np.array([i / nbr_classes * size for i in range(nbr_classes)])
        pos = np.argmax(discrete_space < y_true[1])
        y_true = np.zeros(nbr_classes)
        y_true[pos] = 1
        return binary_crossentropy(y_true, y_pred)
