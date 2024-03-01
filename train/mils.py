import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Softmax pooling based MIL
class softmil(nn.Module):
    def __init__(self, input_size, output_size):
        super(softmil, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.classifier =  nn.Linear(self.input_size, self.output_size)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x): 
        Y_probs = self.classifier(x)
        A = self.softmax(Y_probs)
        Y_prob = torch.sum(torch.mul(Y_probs, A))
        return Y_prob, A

# Linear Softmax pooling based MIL
class linsoftmil(nn.Module):
    def __init__(self, input_size, output_size):
        super(linsoftmil,self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.classifier =  nn.Linear(self.input_size, self.output_size)

    def forward(self, x):
        Y_probs = self.classifier(x)
        Y_prob = (Y_probs**2).sum() / Y_probs.sum()
        return Y_prob, Y_probs

# Log-Sum-Exp Pooling based MIL
class lsemil(nn.Module):
    def __init__(self, input_size,output_size, temperature=1.):
        super(lsemil, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.classifier =  nn.Linear(self.input_size, self.output_size)
        self.temperature = nn.Parameter(torch.tensor(temperature, dtype=torch.float64), requires_grad=True) 
    def forward(self, x): # N,D
        Y_probs = self.classifier(x)
        exp = torch.exp(torch.mul(Y_probs, self.temperature))
        Y_prob = torch.log(torch.mean(exp))/self.temperature
        return Y_prob, Y_probs



# Max pooling based MIL
class maxmil(nn.Module):
    def __init__(self, input_size, output_size):
        super(maxmil, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.classifier =  nn.Linear(self.input_size, self.output_size)      
        # self.classifier = nn.Sequential( nn.Linear(self.input_size, self.output_size), nn.Sigmoid())

    def forward(self, x): # N,D
        Y_probs = self.classifier(x)
        Y_prob = torch.amax(Y_probs)  # KxL
        return Y_prob, Y_probs


# Multi-class Max pooling based MIL
class mcmaxmil(nn.Module):
    def __init__(self, input_size, output_size):
        super(mcmaxmil, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.classifier =  nn.Linear(self.input_size, self.output_size)      
        # self.classifier = nn.Sequential( nn.Linear(self.input_size, self.output_size), nn.Sigmoid())

    def forward(self, x): # N,D
        Y_probs = self.classifier(x)
        Y_prob = Y_probs[torch.argmax(Y_probs[:,1:]).item()//(self.output_size-1)]
        return Y_prob, Y_probs


# Mean pooling based MIL
class meanmil(nn.Module):
    def __init__(self, input_size,output_size):
        super(meanmil, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        # self.classifier = nn.Sequential( nn.Linear(self.input_size, self.output_size), nn.Sigmoid())
        self.classifier =  nn.Linear(self.input_size, self.output_size)

    def forward(self, x): # N,D
        Y_probs = self.classifier(x)
        Y_prob = torch.mean(Y_probs) 
        return Y_prob, Y_probs


# Mean pooling based MIL
class mcmeanmil(nn.Module):
    def __init__(self, input_size,output_size):
        super(mcmeanmil, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        # self.classifier = nn.Sequential( nn.Linear(self.input_size, self.output_size), nn.Sigmoid())
        self.classifier =  nn.Linear(self.input_size, self.output_size)

    def forward(self, x): # N,D
        Y_probs = self.classifier(x)
        Y_prob = torch.mean(Y_probs, axis=0)
        return Y_prob, Y_probs


# Auto pooling based MIL
class automil(nn.Module):
    def __init__(self, input_size, output_size, alpha=1.):
        super(automil, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        # self.classifier = nn.Sequential( nn.Linear(self.input_size, self.output_size), nn.Sigmoid())
        self.classifier =  nn.Linear(self.input_size, self.output_size)
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=True)
        # self.register_parameter("alpha", nn.Parameter(torch.ones(1)))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x): 
        Y_probs = self.classifier(x)
        sig_alpha = torch.log(1.+ torch.exp(self.alpha)) # torch.relu(self.alpha)
        A = self.softmax(torch.mul(Y_probs, sig_alpha))
        Y_prob = torch.sum(torch.mul(Y_probs, A))
        return Y_prob, A


# Auto pooling based MIL
class mcautomil(nn.Module):
    def __init__(self, input_size, output_size, alpha=1.):
        super(mcautomil, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        # self.classifier = nn.Sequential( nn.Linear(self.input_size, self.output_size), nn.Sigmoid())
        self.classifier =  nn.Linear(self.input_size, self.output_size)
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=True)
        # self.register_parameter("alpha", nn.Parameter(torch.ones(1)))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x): 
        Y_probs = self.classifier(x)
        sig_alpha = torch.log(1.+ torch.exp(self.alpha)) # torch.relu(self.alpha)
        A = self.softmax(torch.mul(Y_probs, sig_alpha))
        Y_prob = torch.sum(torch.mul(Y_probs, A), axis=0)
        return Y_prob, A


# Attention pooling based MIL
class attenmil(nn.Module):
    def __init__(self, input_size, output_size=1):
        super(attenmil, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.classifier = nn.Sequential( nn.Linear(self.input_size, self.output_size), nn.Sigmoid())
        # self.classifier =  nn.Linear(self.input_size, self.output_size)      
        self.attention = nn.Linear(self.input_size, 1)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        Y_probs = self.classifier(x)
        A = self.softmax(self.attention(x))
        Y_prob = torch.sum(torch.mul(Y_probs, A))
        return Y_prob, A



# Multi-class Attention pooling based MIL
class mcattenmil(nn.Module):
    def __init__(self, input_size, output_size=1):
        super(mcattenmil, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        # self.classifier = nn.Sequential( nn.Linear(self.input_size, self.output_size), nn.Softmax(dim=0))
        self.classifier =  nn.Linear(self.input_size, self.output_size)      
        self.attention = nn.Linear(self.input_size, 1)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        Y_probs = self.classifier(x)
        A = self.softmax(self.attention(x))
        Y_prob = torch.sum(torch.mul(Y_probs, A),axis=0)
        return Y_prob, A

# Mixed Max Average Pooling based MIL
class mixmil(nn.Module):
    def __init__(self, input_size, output_size, alpha=0.5):
        super(mixmil, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.classifier =  nn.Linear(self.input_size, self.output_size)      
        # self.classifier = nn.Sequential( nn.Linear(self.input_size, self.output_size), nn.Sigmoid())
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=True)

    def forward(self, x):
        Y_probs = self.classifier(x)
        sig_alpha = torch.sigmoid(self.alpha)
        Y_prob = sig_alpha * torch.amax(Y_probs) + (1 - sig_alpha) * torch.mean(Y_probs)  
        return Y_prob, Y_probs 


# Mixed Max Average Pooling based MIL
class mcmixmil(nn.Module):
    def __init__(self, input_size, output_size, alpha=0.5):
        super(mcmixmil, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.classifier =  nn.Linear(self.input_size, self.output_size)      
        # self.classifier = nn.Sequential( nn.Linear(self.input_size, self.output_size), nn.Sigmoid())
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=True)

    def forward(self, x):
        Y_probs = self.classifier(x)
        sig_alpha = torch.sigmoid(self.alpha)
        y_max = Y_probs[torch.argmax(Y_probs[:,1:]).item()//(self.output_size-1)]
        Y_prob = sig_alpha * y_max + (1 - sig_alpha) * torch.mean(Y_probs, axis=0)
        return Y_prob, Y_probs 


# Learned Norm Pooling based MIL
class lnpmil(nn.Module):
    def __init__(self, input_size, output_size, norm=1.):
        super(lnpmil, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.p = nn.Parameter(torch.tensor(norm), requires_grad=True) 
        self.classifier =  nn.Linear(self.input_size, self.output_size)      
        # self.classifier = nn.Sequential( nn.Linear(self.input_size, self.output_size), nn.Sigmoid())

    def forward(self, x):
        Y_probs = torch.abs(self.classifier(x))
        norm_p= 1.+torch.log(1.+ torch.exp(self.p))
        Y_prob = torch.mean(Y_probs.pow(norm_p)).pow(1. / norm_p)
        return Y_prob, Y_probs 


# Multi-class Learned Norm Pooling based MIL
class mclnpmil(nn.Module):
    def __init__(self, input_size, output_size, norm=1.):
        super(mclnpmil, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.p = nn.Parameter(torch.tensor(norm), requires_grad=True) 
        self.classifier =  nn.Linear(self.input_size, self.output_size)      
        # self.classifier = nn.Sequential( nn.Linear(self.input_size, self.output_size), nn.Sigmoid())

    def forward(self, x):
        Y_probs = torch.abs(self.classifier(x))
        norm_p= 1.+torch.log(1.+ torch.exp(self.p))
        Y_prob = torch.mean(Y_probs.pow(norm_p), axis=0).pow(1. / norm_p)
        return Y_prob, Y_probs 


