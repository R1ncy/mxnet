import math
import time
from collections import namedtuple
import mxnet as mx
import numpy as np
from data_load import *
 
CNNModel = namedtuple("CNNModel", ['cnn_exec', 'symbol', 'data', 'label', 'param_blocks'])
def make_text_cnn(sentence_size, num_embed, batch_size,
        num_label=2, filter_list=[3, 4, 5], num_filter=100,dropout=0.):
 
    input_x = mx.sym.Variable('data')
    input_y = mx.sym.Variable('softmax_label')
    conv_input = input_x
 
    pooled_outputs = []
    for i, filter_size in enumerate(filter_list):
        convi = mx.sym.Convolution(data=conv_input, kernel=(filter_size, num_embed), num_filter=num_filter)
        relui = mx.sym.Activation(data=convi, act_type='relu')
        pooli = mx.sym.Pooling(data=relui, pool_type='max', kernel=(sentence_size - filter_size + 1, 1), stride=(1,1))
        pooled_outputs.append(pooli)
 
    total_filters = num_filter * len(filter_list)
    concat = mx.sym.Concat(*pooled_outputs, dim=1)
    h_pool = mx.sym.Reshape(data=concat, target_shape=(batch_size, total_filters))
 
    if dropout > 0.0:
        h_drop = mx.sym.Dropout(data=h_pool, p=dropout)
    else:
        h_drop = h_pool
 
    cls_weight = mx.sym.Variable('cls_weight')
    cls_bias = mx.sym.Variable('cls_bias')
 
    fc = mx.sym.FullyConnected(data=h_drop, weight=cls_weight, bias=cls_bias, num_hidden=num_label)
    sm = mx.sym.SoftmaxOutput(data=fc, label=input_y, name='softmax')
    return sm
 
def setup_cnn_model(ctx, batch_size, sentence_size, num_embed,
        dropout=0.5, initializer=mx.initializer.Uniform(0.1)):
 
    cnn = make_text_cnn(sentence_size, num_embed, batch_size=batch_size, dropout=dropout)
    arg_names = cnn.list_arguments()
    input_shapes = {}
    input_shapes['data'] = (batch_size, 1, sentence_size, num_embed)
 
    arg_shape, out_shape, aux_shape = cnn.infer_shape(**input_shapes)
    arg_arrays = [mx.nd.zeros(s, ctx) for s in arg_shape]
    args_grad = {}
    for shape, name in zip(arg_shape, arg_names):
        if name in ['softmax_label', 'data']:
            continue
        args_grad[name] = mx.nd.zeros(shape, ctx)
 
    cnn_exec = cnn.bind(ctx=ctx, args=arg_arrays, args_grad=args_grad, grad_req='add')
 
    param_blocks = []
    arg_dict = dict(zip(arg_names, cnn_exec.arg_arrays))
    for i, name in enumerate(arg_names):
        if name in ['softmax_label', 'data']:
            continue
        initializer(name, arg_dict[name])
        param_blocks.append((i, arg_dict[name], args_grad[name], name))
 
    data = cnn_exec.arg_dict['data']
    label = cnn_exec.arg_dict['softmax_label']
    return CNNModel(cnn_exec=cnn_exec, symbol=cnn, data=data, label=label, param_blocks=param_blocks)
 
def train_cnn(model, X_train_batch, y_train_batch, X_dev_batch, y_dev_batch, batch_size,
        optimizer='rmsprop', max_grad_norm=5.0, learning_rate=0.1, epoch=200):
    m = model
    opt = mx.optimizer.create(optimizer)
    opt.lr = learning_rate
    updater = mx.optimizer.get_updater(opt)
 
    for iteration in range(epoch):
        tic = time.time()
        num_correct = 0
        num_total = 0
        for begin in range(0, X_train_batch.shape[0], batch_size):
            batchX = X_train_batch[begin:begin+batch_size]
            batchY = y_train_batch[begin:begin+batch_size]
 
            m.data[:] = batchX
            m.label[:] = batchY
            m.cnn_exec.forward(is_train=True)
            m.cnn_exec.backward()
 
            num_total += len(batchY)
            num_correct += sum(batchY == np.argmax(m.cnn_exec.outputs[0].asnumpy(), axis=1))
 
            norm = 0
            for idx, weight, grad, name in m.param_blocks:
                grad /= batch_size
                l2_norm = mx.nd.norm(grad).asscalar()
                norm += l2_norm * l2_norm
 
            norm = math.sqrt(norm)
            for idx, weight, grad, name in m.param_blocks:
                if norm > max_grad_norm:
                    grad *= (max_grad_norm / norm)
                updater(idx, grad, weight)
                grad[:] = 0.0
 
        opt.lr *= 0.5
        print("reset learning rate to %g" % opt.lr)
        toc = time.time()
        train_time = toc - tic
        train_acc = num_correct * 100 / float(num_total)
 
 
        num_correct = 0
        num_total = 0
        for begin in range(0, X_dev_batch.shape[0], batch_size):
            batchX = X_dev_batch[begin:begin + batch_size]
            batchY = y_dev_batch[begin:begin + batch_size]
            m.data[:] = batchX
            m.label[:] = batchY
            m.cnn_exec.forward(is_train=False)
             
            num_correct += sum(batchY == np.argmax(m.cnn_exec.outputs[0].asnumpy(), axis=1))
            num_total += len(batchY)
         
        test_acc = num_correct * 100 / float(num_total)
        print('Iter %d Train: Time: %.3fs, Training Accuracy: %.3f \
                        --- Dev Accuracy thus far: %.3f' % (iteration, train_time, train_acc, test_acc))
 
if __name__ == '__main__':
    train_x, train_y = load_data_with_word2vec("D:\\train1.xlsx","D:\\en_word_vec.txt", max_len=64)
    test_x, test_y = load_data_with_word2vec("D:\\test1.xlsx","D:\\en_word_vec.txt", max_len = 64)
    batch_size = 50
     
    if train_x.shape[0] % batch_size != 0:
        train_x = pad_data_with_batchsize(train_x, batch_size)
        train_y = pad_data_with_batchsize(train_y, batch_size)
    if test_x.shape[0] % batch_size != 0:
        test_x = pad_data_with_batchsize(test_x, batch_size)
        test_y = pad_data_with_batchsize(test_y, batch_size)
    train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1], train_x.shape[2]))
    test_x = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1], test_x.shape[2]))
    num_embed = train_x.shape[-1]
    sentence_size = train_x.shape[2]
 
    cnn_model = setup_cnn_model(mx.cpu(0), batch_size, sentence_size, num_embed, dropout=0.5)
    train_cnn(cnn_model, train_x, train_y, test_x, test_y, batch_size,epoch=20)