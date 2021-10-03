import torch
from torch.autograd import Function
from torch import nn
from torch.nn import functional as F
from NCE.alias_multinomial import AliasMethod
import math

class NCEFunction(Function):
    @staticmethod
    def forward(self, x, y, memory, idx, params):
        K = int(params[0].item()) # number of -ve samples
        T = params[1].item()
        Z = params[2].item()

        momentum = params[3].item()
        batchSize = x.size(0)
        outputSize = memory.size(0) #dataset size
        inputSize = memory.size(1) #128

        # sample positives & negatives
        #idx dim is batch_size x k+1-> +1 here for the +ve samples
        idx.select(1,0).copy_(y.data) #idx[:,0].copy_(y.data)

        # sample correspoinding weights
        weight = torch.index_select(memory, 0, idx.view(-1))# .view(-1)-> for flattenning
        weight.resize_(batchSize, K+1, inputSize) #values corresponds to the idx in the memory
        
        # print("weight sum")
        # print(weight.sum())

        # inner product
        x_norm = x.reshape(batchSize, inputSize, 1)
        # with torch.no_grad():
        
        weight_norm = F.normalize(weight, dim=2)
        x_norm = F.normalize(x_norm, dim=1).data
        
        out = torch.bmm(weight_norm, x_norm) #v.T * fi->output size should be (batxh_size,K+1,1)
        # print("out1 sum")
        # print(out.sum())
        # print("T")
        # print(T)
        out.div_(T).exp_() # batchSize * self.K+1 # out = exp((v.T * fi)/T)
        # print("out2 sum")
        # print(out.sum())
        # out.exp_()
        
        # print("out3 sum")
        # print(out.sum())
        x_norm = x_norm.reshape(batchSize, inputSize)

        if Z < 0:
            params[2] = out.mean() * outputSize
            Z = params[2].item()
            print("normalization constant Z is set to {:.1f}".format(Z))

        out.div_(Z).resize_(batchSize, K+1) #P(i|v) #out.div_(Z) 
        # print("out sum")
        # print(out.sum())

        self.save_for_backward(x_norm.data, memory, y, weight_norm, out, params) #Saves given tensors for a future call to backward()
        

        return out

    @staticmethod
    def backward(self, gradOutput):
        # gradOutput is the gradient output before this backward
        # grad_output is the gradient of the loss w.r.t. the layer output.
        # So if you have a layer l and do, say, y = l(x) ; loss = y.sum(); loss.backward(), you get the gradient of loss w.r.t. y.
        x, memory, y, weight, out, params = self.saved_tensors
        K = int(params[0].item())
        T = params[1].item()
        Z = params[2].item()
        momentum = params[3].item()
        batchSize = gradOutput.size(0)
        
        # gradients d Pm / d linear = exp(linear) / Z -> grad "out" wrt linear-> grad exp() is exp()
        gradOutput.data.mul_(out.data)
        # add temperature
        gradOutput.data.div_(T)#-> looks to me the derivative of P(i|v)!! (derivative of exp(x^2) wrt x is 2x*exp(x^2))

        gradOutput = gradOutput.reshape(batchSize, 1, K+1)
        
        # gradient of linear
        gradInput = torch.bmm(gradOutput.data.to(torch.float32), weight)#looks like it's a grad of the "out" wrt fi
        gradInput.resize_as_(x)

        # update the non-parametric data
        weight_pos = weight.select(1, 0).resize_as_(x)#weight[:,0].resize_as_(x)->+ve samples values 128 dim
        weight_pos.mul_(momentum) #old weights
        weight_pos.add_(torch.mul(x.data, 1-momentum)) # x is the new weights
        w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)#(x1^2 + x2^2 + .....xn^2)^0.5
        updated_weight = weight_pos.div(w_norm)
        memory.index_copy_(0, y, updated_weight) #only the +ve samples updated
        
        return gradInput, None, None, None, None

class NCEAverage(nn.Module):

    def __init__(self, inputSize, outputSize, K, T=0.07, momentum=0.5, Z=None):
        super(NCEAverage, self).__init__()
        self.nLem = outputSize
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams) # way to sample from discrete distribution
        self.multinomial.cuda()
        self.K = K #number of -ve samples

        self.register_buffer('params',torch.tensor([K, T, -1, momentum]));# register_buffer is of the parameters that are passed to the optimizer
        stdv = 1. / math.sqrt(inputSize/3)
        self.register_buffer('memory', torch.rand(outputSize, inputSize).mul_(2*stdv).add_(-stdv)) #stdv for init with random noise
 
    def forward(self, x, y): #x is the 128dim feature and y is the index in the dataset
        batchSize = x.size(0)
        idx = self.multinomial.draw(batchSize * (self.K+1)).view(batchSize, -1)
        out = NCEFunction.apply(x, y, self.memory, idx, self.params)
        
        return out

