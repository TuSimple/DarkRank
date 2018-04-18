"""

NIN

Reference:

Lin, Min, Qiang Chen, and Shuicheng Yan. "Network in network." arXiv preprint arXiv:1312.4400 (2013).
"""

import mxnet as mx


def get_symbol(num_classes=1000):
    # data
    data = mx.symbol.Variable(name='data')
    # stage 1
    net = mx.symbol.Convolution(data=data, num_filter=96, kernel=(11, 11), stride=(4, 4), name='conv1', )
    net = mx.symbol.BatchNorm(data=net, name='bn_1')
    net = mx.symbol.Activation(data=net, act_type='relu', name='relu0')
    net = mx.symbol.Convolution(data=net, num_filter=96, kernel=(1, 1), name='cccp1', )
    net = mx.symbol.BatchNorm(data=net, name='bn_2')
    net = mx.symbol.Activation(data=net, act_type='relu', name='relu1')
    net = mx.symbol.Convolution(data=net, num_filter=96, kernel=(1, 1), name='cccp2', )
    net = mx.symbol.BatchNorm(data=net, name='bn_3')
    net = mx.symbol.Activation(data=net, act_type='relu', name='relu2')
    net = mx.symbol.Pooling(data=net, kernel=(3, 3), stride=(2, 2), pooling_convention='full', pool_type='max', name='pool0')
    # stage 2
    net = mx.symbol.Convolution(data=net, num_filter=256, kernel=(5, 5), pad=(2, 2), name='conv2', )
    net = mx.symbol.BatchNorm(data=net, name='bn_4')
    net = mx.symbol.Activation(data=net, act_type='relu', name='relu3')
    net = mx.symbol.Convolution(data=net, num_filter=256, kernel=(1, 1), name='cccp3', )
    net = mx.symbol.BatchNorm(data=net, name='bn_5')
    net = mx.symbol.Activation(data=net, act_type='relu', name='relu5')
    net = mx.symbol.Convolution(data=net, num_filter=256, kernel=(1, 1), name='cccp4', )
    net = mx.symbol.BatchNorm(data=net, name='bn_6')
    net = mx.symbol.Activation(data=net, act_type='relu', name='relu6')
    net = mx.symbol.Pooling(data=net, kernel=(3, 3), stride=(2, 2), pooling_convention='full', pool_type='max', name='pool2')
    # stage 3
    net = mx.symbol.Convolution(data=net, num_filter=384, kernel=(3, 3), pad=(1, 1), name='conv3', )
    net = mx.symbol.BatchNorm(data=net, name='bn_7')
    net = mx.symbol.Activation(data=net, act_type='relu', name='relu7')
    net = mx.symbol.Convolution(data=net, num_filter=384, kernel=(1, 1), name='cccp5', )
    net = mx.symbol.BatchNorm(data=net, name='bn_8')
    net = mx.symbol.Activation(data=net, act_type='relu', name='relu8')
    net = mx.symbol.Convolution(data=net, num_filter=384, kernel=(1, 1), name='cccp6', )
    net = mx.symbol.BatchNorm(data=net, name='bn_9')
    net = mx.symbol.Activation(data=net, act_type='relu', name='relu9')
    net = mx.symbol.Pooling(data=net, kernel=(3, 3), stride=(2, 2), pooling_convention='full', pool_type='max', name='pool3')
    net = mx.symbol.Dropout(data=net, name='drop')
    # stage 4
    net = mx.symbol.Convolution(data=net, num_filter=1024, kernel=(3, 3), pad=(1, 1), name='conv4_1024', )
    net = mx.symbol.BatchNorm(data=net, name='bn_10')
    net = mx.symbol.Activation(data=net, act_type='relu', name='relu10')
    net = mx.symbol.Convolution(data=net, num_filter=1024, kernel=(1, 1), name='cccp7_1024', )
    net = mx.symbol.BatchNorm(data=net, name='bn_11')
    net = mx.symbol.Activation(data=net, act_type='relu', name='relu11')
    net = mx.symbol.Convolution(data=net, num_filter=1024, kernel=(1, 1), name='cccp8_1024_new', )
    net = mx.symbol.BatchNorm(data=net, name='bn_12')
    net = mx.symbol.Activation(data=net, act_type='relu', name='relu12')

    return net


if __name__ == '__main__':
    sym = get_symbol()
    mx.viz.print_summary(sym, {'data': (1, 3, 224, 112)})
