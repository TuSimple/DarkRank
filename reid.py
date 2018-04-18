"""
A universal training file for
- Vanilla
- ListNet
- ListMLE
- ListMSE
- Match
- KD
- Focal Softmax
- Focal LMNN
"""
import datetime
import time
import importlib
import logging
import argparse

from mxnet_ext.kt_module import KTModule
from mxnet_ext.verify_iter import VerifyIter
from PYOP import lmnn, listnet_loss, listmle_loss

import mxnet as mx
from mxnet.optimizer import SGD
from mxnet.metric import CompositeEvalMetric, Accuracy, Loss
import six


def load_checkpoint(prefix, epoch):
    """
    Load model checkpoint from file.
    :param prefix: Prefix of model name.
    :param epoch: Epoch number of model we would like to load.
    :return: (arg_params, aux_params)
    arg_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's weights.
    aux_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's auxiliary states.
    """
    save_dict = mx.nd.load('%s-%04d.params' % (prefix, epoch))
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            aux_params[name] = v
    return arg_params, aux_params


def get_imrecord_iter(name, input_shape, batch_size, kv, shuffle=False, aug=False, even_iter=False):
    if even_iter:
        aug_params = {'resize': 256,
                      'rand_crop': aug,
                      'rand_mirror': aug,
                      'input_shape': input_shape,
                      'mean': 128.0}

        data_iter = EvenIter(
            lst_name='{}/{}.lst'.format(args.data_dir, name),
            batch_size=batch_size / 2,
            aug_params=aug_params,
            shuffle=shuffle)
    else:
        data_iter = mx.io.ImageRecordIter(
            path_imglist="{}/{}.lst".format(args.data_dir, name),
            path_imgrec="{}/{}.rec".format(args.data_dir, name),
            mean_r=128.0,
            mean_g=128.0,
            mean_b=128.0,
            rand_crop=aug,
            rand_mirror=aug,
            prefetch_buffer=4,
            preprocess_threads=3,
            shuffle=shuffle,
            label_width=1,
            round_batch=False,
            data_shape=input_shape,
            batch_size=batch_size / 2)

    return data_iter


def get_iterators(batch_size, input_shape, train, test, kv, gpus=1):
    train_even_iter = get_imrecord_iter(
        name='{}-even'.format(train),
        input_shape=input_shape,
        batch_size=batch_size,
        kv=kv,
        shuffle=args.even_iter,
        aug=True,
        even_iter=args.even_iter)
    train_rand_iter = get_imrecord_iter(
        name='{}-rand'.format(train),
        input_shape=input_shape,
        batch_size=batch_size,
        kv=kv,
        shuffle=True,
        aug=True)

    return VerifyIter(train_even_iter, train_rand_iter, use_lsoftmax=True, use_softmax=True, gpus=gpus)


def load_symbol(network_name, params_prefix, params_epoch, num_id, mode='student'):
    if mode == 'student' and args.from_scratch:
        print('training from scratch')
        arg_params, aux_params = None, None
    else:
        print('loading from %s-%d' % (params_prefix, params_epoch))
        arg_params, aux_params = \
            load_checkpoint('models/{}'.format(params_prefix), params_epoch)
    if mode == 'kd':
        symbol = importlib.import_module('symbol.symbol_' + network_name).get_symbol(num_id)
    else:
        symbol = importlib.import_module('symbol.symbol_' + network_name).get_symbol()

    if mode == 'student':
        symbol = build_network(symbol, num_class=num_id)
    print('loaded symbol_' + network_name)
    return symbol, arg_params, aux_params


def build_network(net, num_class):
    # classification loss
    pooling = mx.symbol.Pooling(data=net, kernel=(1, 1), global_pool=True, pool_type='avg',
                                name='global_pool')
    flatten = mx.symbol.Flatten(data=pooling, name='flatten')
    lsoftmax = mx.sym.LSoftmax(data=flatten, num_hidden=num_class, beta=1000, margin=3, scale=0.99999,
                               beta_min=5, name='lsoftmax')

    softmax = mx.sym.SoftmaxOutput(data=lsoftmax, name='softmax')

    # normalized for distance-based loss
    l2 = mx.symbol.L2Normalization(data=flatten, name='l2_norm')
    dropout = mx.symbol.Dropout(data=l2, name='dropout')
    lmnn = mx.sym.Custom(data=dropout, epsilon=0.1, threshd=0.9, op_type='LMNN', name='lmnn')

    outputs = [softmax, lmnn]

    if args.kd_temperature:
        logger.info("using kd")
        teacher_logit = mx.sym.Variable('kd_teacher_logit')
        soften_student_logit = lsoftmax / args.kd_temperature
        soften_teacher_logit = teacher_logit / args.kd_temperature
        soften_teacher_activation = mx.sym.SoftmaxActivation(data=soften_teacher_logit)
        kd = mx.sym.SoftmaxOutput(data=soften_student_logit, label=soften_teacher_activation,
                                  grad_scale=args.kd_temperature ** 2, name='kd')
        outputs.append(kd)

    if args.loss_weight_listnet:
        logger.info("using listnet")
        listnet = mx.sym.Custom(data=l2, power=args.score_power, scale=args.embedding_l2_norm,
                                list_length=args.list_length, grad_scale=args.loss_weight_listnet,
                                op_type='listnetLoss', name='listnet')
        outputs.append(listnet)

    if args.loss_weight_listmle:
        logger.info("using listmle")
        listmle = mx.sym.Custom(data=l2, power=args.score_power, scale=args.embedding_l2_norm,
                                grad_scale=args.loss_weight_listmle, op_type='listmleLoss', name='listmle')
        outputs.append(listmle)

    if args.loss_weight_listmse:
        logger.info("using listmse")
        listmse = mx.sym.Custom(data=l2, power=args.score_power, scale=args.embedding_l2_norm,
                                grad_scale=args.loss_weight_listmse, op_type='listmseLoss')
        outputs.append(listmse)

    if args.loss_weight_match:
        logger.info("using match")
        match = mx.sym.Custom(data=l2, grad_scale=args.loss_weight_match, op_type='matchLoss', name='match')
        outputs.append(match)

    return mx.symbol.Group(outputs)


def parse_args():
    parser = argparse.ArgumentParser(description='multi-task reid model')
    parser.add_argument('--gpus', type=str,
                        help='indices of gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--data-dir', type=str, default="data/Market-list",
                        help='directory containing rec and lst files')
    parser.add_argument('--num-examples', type=int,
                        help='number of training examples')
    parser.add_argument('--num-id', type=int,
                        help='number of training classes')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='mini-batch size')
    parser.add_argument('--lr', type=float,
                        help='initial learning rate')
    parser.add_argument('--num-epochs', type=int,
                        help='number of training epochs')
    parser.add_argument('--mode', type=str,
                        help='prefix of saved model params and log')
    parser.add_argument('--train-file', type=str,
                        help='train file name without extension')
    parser.add_argument('--kv-store', type=str, # default='device',
                        help='kvstore type')
    parser.add_argument('--teacher-network', type=str, nargs='+',
                        help='symbol file name of the teacher network')
    parser.add_argument('--teacher-params-epoch', type=int, nargs='+',
                        help='epoch number of the pre-trained params file')
    parser.add_argument('--teacher-params-prefix', type=str, nargs='+',
                        help='prefix of the pre-trained params file')
    parser.add_argument('--student-network', type=str,
                        help='symbol file name of the student network')
    parser.add_argument('--student-params-epoch', type=int,
                        help='epoch number of the pre-trained params file')
    parser.add_argument('--student-params-prefix', type=str, default="nin",
                        help='prefix of the pre-trained params file')
    parser.add_argument('--even-iter', action='store_true', default=False,
                        help='toggle even iterator')
    parser.add_argument('--even-iter1', action='store_true', default=False,
                        help='toggle even iterator 1')
    parser.add_argument('--score-power', type=float,
                        help='score power of listmle')
    parser.add_argument('--embedding-l2-norm', type=float,
                        help='control the l2 norm of embedding vector')
    parser.add_argument('--list-length', type=int,
                        help='the number of samples used to calculate probability')
    parser.add_argument('--loss-weight-match', type=float,
                        help='loss weight of match loss')
    parser.add_argument('--loss-weight-listmle', type=float,
                        help='loss weight of listmle loss')
    parser.add_argument('--loss-weight-listnet', type=float,
                        help='loss weight of listnet loss')
    parser.add_argument('--loss-weight-listmse', type=float,
                        help='loss weight of listmse loss')
    parser.add_argument('--kd-temperature', type=float,
                        help='temperature of kd loss')
    parser.add_argument('--eval-metric', type=str, default=['acc', 'avg'], nargs='+',
                        help='list of evaluation metrics')
    parser.add_argument('--from-scratch', action='store_true', default=False)
    parser.add_argument('--gamma', type=float,
                        help='gamma for focal softmax loss')
    parser.add_argument('--gamma1', type=float,
                        help='gamma for focal lmnn loss')
    parser.add_argument('--new-lmnn', action='store_true', default=False,
                        help='toggle the new implementation of LMNN loss')
    return parser.parse_args()


if __name__ == "__main__":
    # parse and print configuration
    args = parse_args()
    if args.even_iter:
        from mxnet_ext.even_iter import EvenIter
    elif args.even_iter1:
        from mxnet_ext.even_iter1 import EvenIter
    for k, v in sorted(six.iteritems(vars(args))):
        print(k, v)

    # specify logging utils
    timestr = datetime.datetime.now().__str__()
    timestamp = str(int(time.time()))
    print(timestamp)
    logging.basicConfig(filename='log/{}-{}.log'.format(args.mode, timestamp), level=logging.DEBUG)
    logging.info(args)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    prefix = 'models/{}-{}'.format(args.mode, timestamp)
    epoch_end_callback = mx.callback.do_checkpoint(prefix, 10)
    batch_end_callback = mx.callback.Speedometer(batch_size=args.batch_size)

    devices = [mx.gpu(int(i)) for i in args.gpus.split(',')]
    kv = None # mx.kvstore.create(args.kv_store)
    train = get_iterators(
        batch_size=args.batch_size,
        input_shape=(3, 224, 112),
        train=args.train_file,
        test=None,
        kv=kv,
        gpus=len(devices))

    init = mx.initializer.Xavier(rnd_type='gaussian', factor_type='in', magnitude=2)
    batch_per_epoch = int(args.num_examples * 2 / args.batch_size)
    lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(step=[batch_per_epoch * x for x in [50, 75]], factor=0.1)
    sgd = SGD(
        learning_rate=args.lr,
        momentum=0.9,
        wd=0.0001,
        clip_gradient=10,
        lr_scheduler=lr_scheduler,
        rescale_grad=1.0 / args.batch_size)

    # input symbols
    # Note: the order of label must be exactly the same as symbols'
    data_names = ['data']
    data_shapes = [('data', (args.batch_size, 3, 224, 112))]

    label_shapes = [('softmax_label', (args.batch_size, )),
                    ('lsoftmax_label', (args.batch_size, ))]
    label_names = ['softmax_label', 'lsoftmax_label']
    num_binded_loss = 0
    if args.kd_temperature:
        label_names.append('kd_teacher_logit')
        label_shapes.append(('kd_teacher_logit', (args.batch_size, args.num_id)))
    if args.loss_weight_listnet:
        label_names.append('listnet_teacher_label')
        label_shapes.append(('listnet_teacher_label', (args.batch_size, 1024)))
    if args.loss_weight_listmle:
        label_names.append('listmle_teacher_label')
        label_shapes.append(('listmle_teacher_label', (args.batch_size, 1024)))
    if args.loss_weight_listmse:
        label_names.append('listmse_teacher_label')
        label_shapes.append(('listmse_teacher_label', (args.batch_size, 1024)))
    if args.loss_weight_match:
        label_names.append('match_teacher_label')
        label_shapes.append(('match_teacher_label', (args.batch_size, 1024)))

    # load symbol file and pre-trained model and bind pre-trained weight to computational graph
    s_symbol, s_arg_params, s_aux_params = \
        load_symbol(args.student_network, args.student_params_prefix, args.student_params_epoch, args.num_id)

    t_modules = []
    if args.kd_temperature:
        t_label_names = ['lsoftmax_label']
        t_label_shapes = [('lsoftmax_label', (args.batch_size,))]
        t_symbol, t_arg_params, t_aux_params = load_symbol(
            network_name=args.teacher_network[num_binded_loss],
            params_prefix=args.teacher_params_prefix[num_binded_loss],
            params_epoch=args.teacher_params_epoch[num_binded_loss],
            num_id=args.num_id,
            mode='kd')
        t_module = mx.module.Module(symbol=t_symbol, context=devices, label_names=t_label_names)
        t_module.bind(data_shapes=data_shapes, label_shapes=t_label_shapes, for_training=False, grad_req='null')
        t_module.set_params(arg_params=t_arg_params, aux_params=t_aux_params)
        t_modules.append(t_module)
        num_binded_loss += 1

    for i in range(num_binded_loss, len(label_names) - 2):
        t_symbol, t_arg_params, t_aux_params = load_symbol(
            network_name=args.teacher_network[i],
            params_prefix=args.teacher_params_prefix[i],
            params_epoch=args.teacher_params_epoch[i],
            num_id=args.num_id,
            mode='metric')
        t_module = mx.module.Module(symbol=t_symbol, context=devices, label_names=[])
        t_module.bind(data_shapes=data_shapes, for_training=False, grad_req='null')
        t_module.set_params(arg_params=t_arg_params, aux_params=t_aux_params)
        t_modules.append(t_module)

    s_module = KTModule(
        symbol=s_symbol,
        context=devices,
        logger=logger,
        data_names=data_names,
        data_shapes=data_shapes,
        label_names=label_names,
        label_shapes=label_shapes,
        is_transfer=False,
        teacher_module=t_modules)

    # eval_metric = MultiMetric(loss_types=args.eval_metric)
    eval_metric = CompositeEvalMetric()
    eval_metric.add(Accuracy(output_names=["softmax_output"], label_names=["softmax_label"]))
    eval_metric.add(Loss(output_names=["lmnn_output"], label_names=[]))

    s_module.fit(
        train_data=train,
        eval_metric=eval_metric,
        kvstore=kv,
        initializer=init,
        optimizer=sgd,
        num_epoch=args.num_epochs,
        arg_params=s_arg_params,
        aux_params=s_aux_params,
        epoch_end_callback=epoch_end_callback,
        batch_end_callback=batch_end_callback,
        allow_missing=True)
