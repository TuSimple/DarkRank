from __future__ import division
import sys
from itertools import permutations
sys.path.insert(0, "mxnet/python")
import mxnet as mx
import numpy as np


class MatchLoss(mx.operator.CustomOp):
    def __init__(self, grad_scale, eps=1e-12):
        super(MatchLoss, self).__init__()
        self.grad_scale = grad_scale
        self.eps = eps

    def forward(self, is_train, req, in_data, out_data, aux):
        """
        calculate the loss
        :param is_train: in training mode or test mode
        :param req: null, writeTo, writeInplace or addTo, specified by the caller, namely mx.Module or mx.Model
        :param in_data: (0: students, 1: teacher), scores from student, samples from teacher
        :param out_data: -log(p)
        :param aux: 
        :return: 
        """
        # student features and teacher features, l2 normalized
        # when use on gpu, call mx.nd.array.asnumpy before np.array.astype
        s_features = in_data[0].asnumpy().astype(float)  # type: np.ndarray
        t_features = in_data[1].asnumpy().astype(float)  # type: np.ndarray

        batch_size = t_features.shape[0]

        loss = 0
        for query_idx in range(batch_size // 2):
            loss += np.sum(np.square(s_features - t_features))
        self.assign(out_data[0], req[0], loss)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        """
        calculate the gradients
        :param req: 
        :param out_grad: 
        :param in_data: 
        :param out_data: 
        :param in_grad: 
        :param aux: 
        :return: 
        """
        s_features = in_data[0].asnumpy().astype(float)  # type: np.ndarray
        t_features = in_data[1].asnumpy().astype(float)  # type: np.ndarray

        batch_size = t_features.shape[0]
        grads = np.zeros_like(s_features, dtype=float)

        # In even_iter mode, the first half batch of samples are paired. So if we choose one as the query, what left
        # are one good response and (batch_size - 2) bad responses. If we choose one in the second half, all responses
        # will be bad.
        for query_idx in range(batch_size // 2):
            grads += 2 * (s_features - t_features)

        self.assign(in_grad[0], req[0], grads * self.grad_scale)


@mx.operator.register("matchLoss")
class MatchProp(mx.operator.CustomOpProp):
    def __init__(self, grad_scale=1.0):
        super(MatchProp, self).__init__(need_top_grad=False)
        self.grad_scale = float(grad_scale)

    def list_arguments(self):
        return ['data', 'teacher_label']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        label_shape = in_shape[1]
        return [data_shape, label_shape], [(1, )]

    def create_operator(self, ctx, shapes, dtypes):
        return MatchLoss(grad_scale=self.grad_scale)
