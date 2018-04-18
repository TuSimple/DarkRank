"""
diff from rank_loss.py:
- use the whole batch to compute gradient instead of the first sample
- remove bias term in score function, for that it is useless under exp non-linearity

diff from rank_loss1.py:
- only use partial of rank list to construct probability
- use ListNet
"""
from __future__ import division

from itertools import permutations

import mxnet as mx
import numpy as np


def _get_perm_prob(scores, perm):
    exp_score_sum = 0
    perm_prob = 1
    for idx in reversed(perm):
        exp_score = np.exp(scores[idx])
        exp_score_sum += exp_score
        perm_prob *= exp_score / exp_score_sum
    return perm_prob


def _get_all_perm_prob(scores, indices):
    return [_get_perm_prob(scores, r) for r in sorted(permutations(indices))]


def _get_perm_grad(features, query_idx, distances, scores, perm, power):
    exp_scores = np.zeros_like(scores)
    doc_probs = np.zeros_like(scores)
    grads = np.zeros_like(features)

    exp_score_sum = 0
    for idx in reversed(perm):
        exp_score = np.exp(scores[idx])
        exp_score_sum += exp_score
        exp_scores[idx] = exp_score
        doc_probs[idx] = exp_score / exp_score_sum

    for i, idx in enumerate(perm):
        grad_tmp = 0
        for j in range(i + 1):
            grad_tmp += exp_scores[idx] / exp_scores[perm[j]] * doc_probs[perm[j]]
        grads[idx] += (grad_tmp - 1) * (features[idx] - features[query_idx]) * -2 * \
                      (power * distances[idx] ** (power - 1))
    return grads


class ListNetLoss(mx.operator.CustomOp):
    def __init__(self, power, scale, grad_scale, list_length, eps=1e-12):
        super(ListNetLoss, self).__init__()
        self.power = power
        self.scale = scale
        self.grad_scale = grad_scale
        self.eps = eps
        self.list_length = list_length

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
        s_features *= np.sqrt(self.scale)
        t_features = in_data[1].asnumpy().astype(float)  # type: np.ndarray
        t_features *= np.sqrt(self.scale)

        batch_size = t_features.shape[0]

        loss = 0
        for query_idx in range(batch_size // 2):
            t_distances = np.sum(np.square(t_features - t_features[query_idx]), axis=1)
            s_distances = np.sum(np.square(s_features - s_features[query_idx]), axis=1)
            s_scores = -s_distances ** self.power
            t_scores = -t_distances ** self.power
            gt_perm = np.argsort(t_distances)[1:self.list_length+1].tolist()  # The ground-truth permutation
            t_probs = np.array(_get_all_perm_prob(t_scores, gt_perm))
            s_probs = np.array(_get_all_perm_prob(s_scores, gt_perm))
            loss += -np.sum(t_probs * np.log(s_probs))

        loss /= batch_size // 2
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
        s_features *= np.sqrt(self.scale)
        t_features = in_data[1].asnumpy().astype(float)  # type: np.ndarray
        t_features *= np.sqrt(self.scale)

        batch_size = t_features.shape[0]

        grads = np.zeros_like(s_features, dtype=float)
        doc_probs = np.empty(shape=(s_features.shape[0],), dtype=float)  # I_k, see the deduction

        # In even_iter mode, the first half batch of samples are paired. So if we choose one as the query, what left
        # are one good response and (batch_size - 2) bad responses. If we choose one in the second half, all responses
        # will be bad.
        for query_idx in range(batch_size // 2):
            t_distances = np.sum(np.square(t_features - t_features[query_idx]), axis=1)
            s_distances = np.sum(np.square(s_features - s_features[query_idx]), axis=1)
            gt_perm = np.argsort(t_distances)[1:self.list_length+1].tolist()
            s_scores = -s_distances ** self.power
            t_scores = -t_distances ** self.power
            t_probs = _get_all_perm_prob(t_scores, gt_perm)

            for i, perm in enumerate(sorted(permutations(gt_perm))):
                grads += np.sqrt(self.scale) * t_probs[i] * \
                         _get_perm_grad(s_features, query_idx, s_distances, s_scores, perm, self.power)

        grads /= batch_size // 2
        self.assign(in_grad[0], req[0], grads * self.grad_scale)


@mx.operator.register("listnetLoss")
class ListNetProp(mx.operator.CustomOpProp):
    def __init__(self, power=1.0, scale=1.0, grad_scale=1.0, list_length=3):
        super(ListNetProp, self).__init__(need_top_grad=False)
        self.grad_scale = float(grad_scale)
        self.scale = float(scale)
        self.power = float(power)
        self.list_length = int(list_length)

    def list_arguments(self):
        return ['data', 'teacher_label']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        label_shape = in_shape[1]
        assert label_shape[1] == data_shape[1], "wrong label shape {}".format(label_shape)
        return [data_shape, label_shape], [(1, )]

    def create_operator(self, ctx, shapes, dtypes):
        return ListNetLoss(grad_scale=self.grad_scale, power=self.power, scale=self.scale, list_length=self.list_length)
