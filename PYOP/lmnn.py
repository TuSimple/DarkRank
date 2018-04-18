import mxnet as mx
import numpy as np


class LMNNLoss(mx.operator.CustomOp):
    '''
    LMNN Loss Layer = positive pairwise loss + triplet loss
    '''
    def __init__(self, epsilon, threshd, grad_scale):
        self.epsilon = epsilon      # epsilon is the trade-off parameter between positive pairwise and triplet loss(1: epsilon)
        self.threshd = threshd
        self.grad_scale = grad_scale

    
    def forward(self, is_train, req, in_data, out_data, aux):
        x = in_data[0]
        ctx = x.context
        y = mx.nd.zeros((x.shape[0], ), ctx=ctx)
        halfsize = x.shape[0] / 2
        
        for i in range(halfsize):
            pid = i + 1 if i % 2 == 0 else i - 1
            pdiff = x[i] - x[pid]
            pdist = 0.5 * mx.nd.sum(pdiff * pdiff)
            mask = mx.nd.ones((x.shape[0],), ctx=ctx)    # index mask for negative examples
            mask[i] = 0
            mask[pid] = 0
            ndiff = x[i] - x
            ndist = 0.5 * mx.nd.sum(ndiff * ndiff, axis=1)
            distdiff = (pdist - ndist + self.threshd) * mask  
            distdiff = mx.nd.sum(mx.nd.maximum(0, distdiff)) / mx.nd.sum(mask)
            y[i] = pdist + self.epsilon*distdiff   

        self.assign(out_data[0], req[0], y)                     
            
    
    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        x = in_data[0]
        ctx = x.context
        grad = in_grad[0]
        grad[:] = 0
        batchsize = x.shape[0]
                     
        for i in range(batchsize / 2):
            pid = i + 1 if i % 2 == 0 else i - 1
            grad[i] += (x[i] - x[pid]) 
            grad[pid] += (x[pid] - x[i]) 
                
            mask = np.ones((batchsize,))   # index mask for negative examples
            mask[i] = 0
            mask[pid] = 0   

            pdiff = x[i] - x[pid]
            pdist = 0.5 * mx.nd.sum(pdiff * pdiff)                  
            ndiff = x[i] - x
            ndist = 0.5 * mx.nd.sum(ndiff * ndiff, axis=1)
            distdiff = pdist - ndist + self.threshd
         
            index = np.zeros((batchsize, ))
            index[np.where(distdiff.asnumpy() > 0)] = 1
            index = index * mask
            index = mx.nd.array(index, ctx=ctx)
                                      
            ratio = index / (batchsize - 2)  #distdiff * index / (mx.nd.sum(distdiff * index) + 1e-6)
            ratio = mx.nd.Reshape(ratio, shape=(batchsize,1))
            ratio = mx.nd.broadcast_axis(ratio, axis=1, size=x.shape[1])
             
            grad[i] += mx.nd.sum((x - x[pid]) * ratio, axis=0) * self.epsilon
            grad[pid] += (x[pid] - x[i]) * (mx.nd.sum(index) / (batchsize - 2)) * self.epsilon   #(mx.nd.sum(distdiff * index) / (mx.nd.sum(distdiff * index) + 1e-6)) * self.epsilon
            grad += (x[i] - x) * ratio * self.epsilon
               
        self.assign(in_grad[0], req[0], grad * self.grad_scale)
           
           
@mx.operator.register("LMNNLoss")
class LMNNLossProp(mx.operator.CustomOpProp):
    def __init__(self, epsilon=0.1, threshd=0.9, grad_scale=1.0):
        super(LMNNLossProp, self).__init__(need_top_grad=False)
        self.epsilon = float(epsilon)
        self.threshd = float(threshd)
        self.grad_scale = float(grad_scale)
     
    def list_arguments(self):
        return ['data']  
    
    def list_outputs(self):
        return ['output']   

    def infer_shape(self, in_shape):
        data_shape = in_shape[0] 
        output_shape = (in_shape[0][0], )       
        return [data_shape], [output_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return LMNNLoss(self.epsilon, self.threshd, self.grad_scale)
