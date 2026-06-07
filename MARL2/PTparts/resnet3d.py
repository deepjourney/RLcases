import torch,math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from functools import partial
debug = 0
def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module
__all__ = ['ResNet3d', 'resnet3d10', 'resnet3d18', 'resnet3d34', 'resnet3d50', 'resnet3d101', 'resnet3d152', 'resnet3d200']
def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(out.size(0), planes-out.size(1), out.size(2), out.size(3), out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()
    out = Variable(torch.cat([out.data, zero_pads], dim=1))
    return out
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
class ResNet3D(nn.Module):
    def __init__(self, num_inputs, num_outputs, paraslist, block=BasicBlock, shortcut_type='B'):
        self.debugflag = True
        super(ResNet, self).__init__()
        para = paraslist[0].split(',')
        para = [int(parai) for parai in para]
        if debug or self.debugflag: print('init', num_inputs)
        self.conv1 = nn.Conv3d(num_inputs[3],para[9],kernel_size=(para[2],para[0],para[1]),stride=(para[5],para[3],para[4]),padding=(para[8],para[6],para[7]),bias=False)
        num_inputs[0] = (num_inputs[0]+para[6]*2-(para[0]-para[3]))//para[3]
        num_inputs[1] = (num_inputs[1]+para[7]*2-(para[1]-para[4]))//para[4]
        num_inputs[2] = (num_inputs[2]+para[8]*2-(para[2]-para[5]))//para[5]
        if debug or self.debugflag: print('init conv1', num_inputs)
        self.bn1 = nn.BatchNorm3d(para[9])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=para[10], padding=1)
        num_inputs[0] = (num_inputs[0]+2-(3-para[10]))//para[10]
        num_inputs[1] = (num_inputs[1]+2-(3-para[10]))//para[10]
        num_inputs[2] = (num_inputs[2]+2-(3-para[10]))//para[10]
        if debug or self.debugflag: print('init maxpool', num_inputs)
        self.inplanes = para[9]
        self.layers = []
        for i,paras in enumerate(paraslist[1:]):
            para = paras.split(',')
            para = [int(parai) for parai in para]
            layer = self._make_layer(block,  para[0], para[1], shortcut_type, stride=para[2])
            num_inputs[0] = (num_inputs[0]+2-(3-para[2]))//para[2]
            num_inputs[1] = (num_inputs[1]+2-(3-para[2]))//para[2]
            num_inputs[2] = (num_inputs[2]+2-(3-para[2]))//para[2]
            if debug or self.debugflag: print('init layer',i,':',num_inputs)
            self.layers.append(layer)
            self.num_filter = para[0]
        self.layers = nn.ModuleList(self.layers)
        if debug or self.debugflag: print('init before avgpool',num_inputs)
        self.avgpool = nn.AvgPool3d((num_inputs[2], num_inputs[0], num_inputs[1]), stride=1)
        self.fc = nn.Linear(self.num_filter*block.expansion, num_outputs)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                #m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.num_outputs = num_outputs
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('relu'))
        self.critic_linear = init_(nn.Linear(self.num_outputs, 1))
    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A': downsample = partial(downsample_basic_block,planes=planes*block.expansion,stride=stride)
            else: downsample = nn.Sequential(nn.Conv3d(self.inplanes,planes*block.expansion,kernel_size=1,stride=stride,bias=False), nn.BatchNorm3d(planes*block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = x.permute(0,4,1,2,3)/255.0 # go to batch,channel,stack,width,height
        if debug or self.debugflag: print(x.shape)
        x = self.conv1(x)
        if debug or self.debugflag: print('conv1',x.shape)
        x = self.bn1(x)
        if debug or self.debugflag: print('bn1',x.shape)
        x = self.relu(x)
        if debug or self.debugflag: print('relu',x.shape)
        x = self.maxpool(x)
        if debug or self.debugflag: print('maxpool',x.shape)
        for i,layer in enumerate(self.layers):
            x = layer(x)
            if debug or self.debugflag: print('layer',i,':',x.shape)
        x = self.avgpool(x)
        if debug or self.debugflag: print('avgpool',x.shape)
        x = x.view(x.size(0), -1)
        if debug or self.debugflag: print('view',x.shape)
        x = self.fc(x)
        if debug or self.debugflag: print('fc',x.shape)
        if debug: exit()
        self.debugflag = False
        return self.critic_linear(x), x

def resnet3d10(**kwargs):
    model = ResNet3D(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model
def resnet3d18(**kwargs):
    model = ResNet3D(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model
def resnet3d34(**kwargs):
    model = ResNet3D(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model
def resnet3d50(**kwargs):
    model = ResNet3D(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model
def resnet3d101(**kwargs):
    model = ResNet3D(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model
def resnet3d152(**kwargs):
    model = ResNet3D(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model
def resnet3d200(**kwargs):
    model = ResNet3D(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model
