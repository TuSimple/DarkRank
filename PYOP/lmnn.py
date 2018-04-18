import mxnet as mx
import numpy as np

eps = 1e-20


def relu(x):
    return np.maximum(x, 0)


class LMNNOp(mx.operator.CustomOp):
    """
       lmnn loss improved with focal loss (focal loss for dense object detection)
    """

    def __init__(self, **kwargs):
        super(LMNNOp, self).__init__()
        self.margin = float(kwargs.get('margin', 0.9))
        self.epsilon = float(kwargs.get('epsilon', 0.1))

    def forward(self, is_train, req, in_data, out_data, aux):
        X = in_data[0].asnumpy()
        L = np.zeros(shape=(X.shape[0], ))
        for i in range(X.shape[0] / 2):
            a = i  # anchor index
            p = (i + 1) if i % 2 == 0 else (i - 1)  # positive sample index
            n = list(set(range(X.shape[0])) - {a, p})  # negative sample indexes
            Xa, Xp, Xn = X[a], X[p], X[n, :]
            pdist2 = ((Xa - Xp) ** 2).sum()  # square of distance between anchor and positive sample
            ndist2 = ((Xa - Xn) ** 2).sum(axis=1, keepdims=True)  # square of distance between anchor and negative samples
            triplet = pdist2 - ndist2 + 2 * self.margin
            L[i] = 0.5 * (pdist2 + self.epsilon * relu(triplet).mean())
        self.assign(out_data[0], req[0], L)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        X = in_data[0].asnumpy()
        G = np.zeros_like(X)
        for i in range(X.shape[0] / 2):
            a = i  # anchor index
            p = (i + 1) if i % 2 == 0 else (i - 1)  # positive sample index
            n = list(set(range(X.shape[0])) - {a, p})  # negative sample indexes
            Xa, Xp, Xn = X[a], X[p], X[n, :]
            pdiff = Xa - Xp
            pdist2 = ((Xa - Xp) ** 2).sum()  # square of distance between anchor and positive sample
            ndist2 = ((Xa - Xn) ** 2).sum(axis=1, keepdims=True)  # square of distance between anchor and negative samples
            triplet = pdist2 - ndist2 + 2 * self.margin
            G[i] = pdiff + self.epsilon * ((Xn - Xp) * (triplet > 0)).mean(axis=0)
        self.assign(in_grad[0], req[0], G)


@mx.operator.register("LMNN")
class LMNNProp(mx.operator.CustomOpProp):
    def __init__(self, **kwargs):
        super(LMNNProp, self).__init__(need_top_grad=False)
        self._kwargs = kwargs

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        assert len(in_shape) == 1, "LMNN input data: [l2_embedding]"
        dshape = in_shape[0]
        assert len(dshape) == 2, "data shape should be (batch_size, embedding_dim), found %s" % dshape
        oshape = (dshape[0], )
        return [dshape], [oshape], []

    def create_operator(self, ctx, shapes, dtypes):
        self._kwargs['shape'] = shapes[0]
        return LMNNOp(**self._kwargs)


def grad_check():
    import random
    DELTA = 1e-3
    RTOL = 1e-2
    NUM_TEST_TRIAL = 1000

    X = mx.sym.var('data')
    random_margin = random.random()
    random_epsilon = random.random() * 10
    sym = mx.sym.Custom(data=X, margin=random_margin, epsilon=random_epsilon, op_type='LMNN')

    batch_size = 8
    num_dim = 10
    data_shape = (batch_size, num_dim)
    data = mx.nd.array(np.random.normal(size=data_shape), ctx=mx.cpu())
    data = mx.nd.L2Normalization(data)

    for i in range(NUM_TEST_TRIAL):
        random_batch_idx = random.randint(0, batch_size / 2 - 1)
        random_feature_idx = random.randint(0, num_dim - 1)
        data_p = data.copy()
        data_p[random_batch_idx][random_feature_idx] += DELTA
        data_m = data.copy()
        data_m[random_batch_idx][random_feature_idx] -= DELTA
        exe = sym.simple_bind(ctx=mx.cpu(), data=data_shape)

        exe.forward(data=data, is_train=True)
        exe.backward()
        L_sym = exe.outputs[0].asnumpy()[random_batch_idx]
        G_sym = exe.grad_arrays[0].asnumpy()[random_batch_idx, random_feature_idx]

        exe.forward(data=data_p, is_train=False)
        L_p = exe.outputs[0].asnumpy()[random_batch_idx]

        exe.forward(data=data_m, is_train=False)
        L_m = exe.outputs[0].asnumpy()[random_batch_idx]

        G_num = (L_p - L_m) / (2 * DELTA)

        print G_sym
        print G_num
        assert abs((G_num - G_sym) / G_sym) < RTOL, "gradient check failed at delta=%g, rtol=%g" % (DELTA, RTOL)


def speed_test(compare=False):
    """
    E5-2650v3, 1024d vector
    forward+backward 6000 samples/s
    backward only 9000 samples/s
    old implementation <1000 samples/s
    """
    import time

    X = mx.sym.var('data')
    if compare:
        sym = mx.sym.Custom(data=X, threshd=0.9, epsilon=0.1, op_type='lmnnLoss')
    else:
        sym = mx.sym.Custom(data=X, margin=0.9, epsilon=0.1, op_type='LMNN')

    batch_size = 8
    num_batch = 1000
    num_dim = 1024
    data_shape = (batch_size, num_dim)
    data = mx.nd.array(np.random.rand(*data_shape), ctx=mx.cpu())
    exe = sym.simple_bind(ctx=mx.cpu(), data=data_shape)

    tic = time.time()
    for i in range(num_batch):
        exe.forward(data=data, is_train=True)
        exe.backward()
        out = exe.outputs[0]
        _ = out.asnumpy()
    toc = time.time()
    print("Vanilla LMNN processed {} 1024d features per second".format(round(num_batch * batch_size / (toc - tic))))

if __name__ == '__main__':
    speed_test()