import numpy as np
import torch
import torch.nn as nn
import torch.distributions as tdist
debug = 0
def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module
class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1)
class CNNBase2D(nn.Module):
    def __init__(self, num_inputs, num_outputs, paraslist):
        self.debugflag = True
        super(CNNBase2D, self).__init__()
        print('CNNBase2D',paraslist)
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('relu'))
        layers = []
        layer_inputs = num_inputs[2]
        if debug or self.debugflag: print('layer s :', num_inputs)
        for i,paras in enumerate(paraslist):
            para = paras.split(',')
            para = [int(parai) for parai in para]
            if para[7]==0: padding_mode = 'zeros'
            if para[7]==1: padding_mode = 'circular'#'reflect', 'replicate' or 'circular'
            layer = init_(nn.Conv2d(layer_inputs, para[6], kernel_size=(para[0],para[1]), stride=(para[2],para[3]), padding=(para[4],para[5]), padding_mode=padding_mode))
            layers.append(layer)
            layer_inputs = para[6]
            layers.append(nn.ReLU())
            num_inputs[0] = (num_inputs[0]+para[4]*2-(para[0]-para[2]))//para[2]
            num_inputs[1] = (num_inputs[1]+para[5]*2-(para[1]-para[3]))//para[3]
            if debug or self.debugflag: print('layer', i, ':', num_inputs[:2], layer_inputs)
        if debug or self.debugflag: print('layer l :', num_outputs)
        layers.append(Flatten())
        layers.append(init_(nn.Linear(layer_inputs*num_inputs[0]*num_inputs[1], num_outputs)))
        layers.append(nn.ReLU())
        self.main = nn.Sequential(*layers)
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        self.critic_linear = init_(nn.Linear(num_outputs, 1))
        self.num_outputs = num_outputs
        self.train()
    def forward(self, inputs):
        inputs = inputs.permute(0,4,1,2,3) # go to batch,channel,stack,width,height
        inputs = inputs.squeeze()
        x = self.main(inputs / 255.0)
        return self.critic_linear(x), x
class CNNBase3D(nn.Module):
    def __init__(self, num_inputs, num_outputs, paraslist):
        self.debugflag = True
        super(CNNBase3D, self).__init__()
        print('CNNBase3D',paraslist)
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('relu'))
        layers = []
        layer_inputs = num_inputs[3]
        if debug or self.debugflag: print('layer s :', num_inputs)
        for i,paras in enumerate(paraslist):
            para = paras.split(',')
            para = [int(parai) for parai in para]
            layer = init_(nn.Conv3d(layer_inputs, para[9], kernel_size=(para[2],para[0],para[1]), 
                                    stride=(para[5],para[3],para[4]), padding=(para[8],para[6],para[7])))
            layers.append(layer)
            layer_inputs = para[9]
            layers.append(nn.ReLU())
            num_inputs[0] = (num_inputs[0]+para[6]*2-(para[0]-para[3]))//para[3]
            num_inputs[1] = (num_inputs[1]+para[7]*2-(para[1]-para[4]))//para[4]
            num_inputs[2] = (num_inputs[2]+para[8]*2-(para[2]-para[5]))//para[5]
            if debug or self.debugflag: print('layer', i, ':', num_inputs[:3], layer_inputs) ###
        if debug or self.debugflag: print('layer l :', num_outputs) ###
        layers.append(Flatten())
        layers.append(init_(nn.Linear(layer_inputs*num_inputs[2]*num_inputs[0]*num_inputs[1], num_outputs)))
        layers.append(nn.ReLU())
        self.main = nn.Sequential(*layers)
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        self.critic_linear = init_(nn.Linear(num_outputs, 1))
        self.num_outputs = num_outputs
        self.train()
    def forward(self, inputs):
        inputs = inputs.permute(0,4,1,2,3) # go to batch,channel,stack,width,height
        x = self.main(inputs / 255.0)
        return self.critic_linear(x), x
class MLPBase(nn.Module):
    def __init__(self, num_inputs, paraslist):
        super(MLPBase, self).__init__()
        print('MLPBase',paraslist)
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        layers = []
        layer_inputs = num_inputs 
        for paras in paraslist:
            para = paras.split(',')
            layer = init_(nn.Linear(layer_inputs, int(para[0])))
            layers.append(layer)
            layers.append(nn.Tanh())#Tanh ReLU
            layer_inputs = int(para[0])
        self.actor = nn.Sequential(*layers)
        layers = []
        layer_inputs = num_inputs 
        for paras in paraslist:
            para = paras.split(',')
            layer = init_(nn.Linear(layer_inputs, int(para[0])))
            layers.append(layer)
            layers.append(nn.Tanh())#Tanh ReLU
            layer_inputs = int(para[0])
        self.critic= nn.Sequential(*layers)
        self.critic_linear = init_(nn.Linear(layer_inputs, 1))
        self.num_outputs = int(paraslist[-1])
        self.train()
    def forward(self, inputs):
        inputs = Flatten()(inputs)#inputs[:,0,:]
        hidden_critic = self.critic(inputs)
        hidden_actor = self.actor(inputs)
        #if not (hidden_critic==hidden_actor).all(): print('Two!!!')
        return self.critic_linear(hidden_critic), hidden_actor


log_prob_bernoulli = tdist.Bernoulli.log_prob
tdist.Bernoulli.log_probs = lambda self, actions: log_prob_bernoulli(self, actions).view(actions.size(0), -1).sum(-1).unsqueeze(-1)
bernoulli_entropy = tdist.Bernoulli.entropy
tdist.Bernoulli.entropy = lambda self: bernoulli_entropy(self).sum(-1)
tdist.Bernoulli.mode = lambda self: torch.gt(self.probs, 0.5).float()
class Bernoulli(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Bernoulli, self).__init__()
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        self.linear = init_(nn.Linear(num_inputs, num_outputs))
    def forward(self, x):
        x = self.linear(x)
        return tdist.Bernoulli(logits=x)

log_prob_cat = tdist.Categorical.log_prob
#tdist.Categorical.log_probs = lambda self, actions: log_prob_cat(self, actions).view(actions.size(0), -1).sum(-1).unsqueeze(-1)
tdist.Categorical.mode = lambda self: self.probs.argmax(dim=-1)
tdist.Categorical.log_probs = tdist.Categorical.log_prob#lambda self, actions: log_prob_cat(self, actions)#.squeeze(-1))#.view(actions.size(0), -1).sum(-1).unsqueeze(-1)
#tdist.Categorical.mode = lambda self: self.probs.argmax(dim=-1, keepdim=True)
#old_sample = tdist.Categorical.sample
#tdist.Categorical.sample = lambda self: old_sample(self).unsqueeze(-1)
class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=0.01)
        self.linear = init_(nn.Linear(num_inputs, num_outputs))
    def forward(self, x):
        x = self.linear(x)
        return tdist.Categorical(logits=x)

log_prob_normal = tdist.Normal.log_prob
tdist.Normal.log_probs = lambda self, actions: log_prob_normal(self, actions).sum(-1, keepdim=True)
normal_entropy = tdist.Normal.entropy
tdist.Normal.entropy = lambda self: normal_entropy(self).sum(-1)
tdist.Normal.mode = lambda self: self.mean
# Necessary for my KFAC implementation.
class AddBias(nn.Module):###can not change name!!!
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))
    def forward(self, x):
        if x.dim() == 2: bias = self._bias.t().view(1, -1)
        else:            bias = self._bias.t().view(1, -1, 1, 1)
        return x + bias
class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))
    def forward(self, x):
        action_mean = self.fc_mean(x)
        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda: zeros = zeros.cuda()
        action_logstd = self.logstd(zeros)
        return tdist.Normal(action_mean, action_logstd.exp())
