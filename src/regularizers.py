import numpy as np
import keras.backend as K
from keras.regularizers import Regularizer

class SparseActivityRegularizer(Regularizer):

	def __init__(self, alpha=0.5, **kwargs):
		super(SparseActivityRegularizer, self).__init__(**kwargs)
		self.alpha = alpha

	def set_layer(self, layer):
		self.layer = layer

	def __call__(self, loss):
		y_pred = self.layer.get_output(True)   
		y_true = K.one_hot(indices=K.argmax(y_pred, axis=1), 
			num_classes=y_pred.get_shape().as_list()[1])


		L = y_true * K.square(K.maximum(np.array(0., dtype=K.floatx()), 0.9 - y_pred)) + \
			0.5 * (1 - y_true) * K.square(K.maximum(np.array(0., dtype=K.floatx()), y_pred - 0.1))

		return K.mean(K.sum(L, axis=-1), axis=-1)

	def get_config(self):
		config = super(SparseActivityRegularizer, self).get_config()
		config.update({"alpha": self.alpha})
		return config