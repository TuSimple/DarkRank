from __future__ import division
import sys
from itertools import permutations
sys.path.insert(0, "mxnet/python")
import mxnet as mx
import numpy as np


class ListMSELoss(mx.operator.CustomOp):
    def __init__(self, power, scale, grad_scale, eps=1e-12):
        super(ListMSELoss, self).__init__()
        self.power = power
        self.scale = scale
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
        s_features *= np.sqrt(self.scale)
        t_features *= np.sqrt(self.scale)

        batch_size = t_features.shape[0]

        loss = 0
        for query_idx in range(batch_size // 2):
            t_distances = np.sum(np.square(t_features - t_features[query_idx]), axis=1)
            s_distances = np.sum(np.square(s_features - s_features[query_idx]), axis=1)
            s_scores = s_distances ** self.power
            t_scores = t_distances ** self.power
            loss += np.sum(np.square(s_scores - t_scores))
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
        s_features *= np.sqrt(self.scale)
        t_features *= np.sqrt(self.scale)

        batch_size = t_features.shape[0]
        grads = np.zeros_like(s_features, dtype=float)

        # In even_iter mode, the first half batch of samples are paired. So if we choose one as the query, what left
        # are one good response and (batch_size - 2) bad responses. If we choose one in the second half, all responses
        # will be bad.
        for query_idx in range(batch_size // 2):
            t_distances = np.sum(np.square(t_features - t_features[query_idx]), axis=1, keepdims=True)
            s_distances = np.sum(np.square(s_features - s_features[query_idx]), axis=1, keepdims=True)
            s_scores = s_distances ** self.power
            t_scores = t_distances ** self.power

            grads += 4 * np.sqrt(self.scale) * self.power * (s_scores - t_scores) * (
            s_features - s_features[query_idx]) * s_distances ** (self.power - 1)

        grads /= batch_size // 2
        self.assign(in_grad[0], req[0], grads * self.grad_scale)


@mx.operator.register("listmseLoss")
class ListMSEProp(mx.operator.CustomOpProp):
    def __init__(self, power=1.0, scale=1.0, grad_scale=1.0):
        super(ListMSEProp, self).__init__(need_top_grad=False)
        self.grad_scale = float(grad_scale)
        self.scale = float(scale)
        self.power = float(power)

    def list_arguments(self):
        return ['data', 'teacher_label']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        label_shape = in_shape[1]
        return [data_shape, label_shape], [(1, )]

    def create_operator(self, ctx, shapes, dtypes):
        return ListMSELoss(grad_scale=self.grad_scale, power=self.power, scale=self.scale)
