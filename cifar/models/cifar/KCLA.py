import torch
from torch import nn
from torch.nn.parameter import Parameter
from math import sqrt
from math import log
import math
import torch.nn.functional as F

__all__ = ['kcla']

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

def torch_log_uniform_sequence(start, stop, num):
    start_log = torch.log(torch.tensor(float(start)))
    stop_log = torch.log(torch.tensor(float(stop)))
    log_values = torch.linspace(start_log, stop_log, num)
    return torch.exp(log_values)


class kcla_layer(nn.Module):
    def __init__(self,in_channel,u_channel,heads=None,init_layer=False,stride=1,padding=0,kernel_size=1,drop_path=0.2):
        super().__init__()

        if heads!=None:
            self.heads=heads
            self.head= in_channel//self.heads
        else:
            self.head=16
            self.heads=in_channel//self.head
        
        self.init_layer=init_layer
        self.u_channel=u_channel

        t = int(abs((log(in_channel, 2) + 1) / 2.))
        k_size = t if t % 2 else t + 1
        self.k_size = k_size

        if self.init_layer:
            self.adjust_channel=nn.Conv2d(u_channel,in_channel,kernel_size=1,stride=stride,padding=0,groups=self.heads,bias=False)
            self.u_Wk=nn.Conv1d(1,
                                1,
                                kernel_size=k_size,
                                padding=(k_size - 1) // 2,
                                bias=False)
            nn.init.xavier_uniform_(self.u_Wk.weight)
            self.u_bn=nn.BatchNorm2d(in_channel)
        self.drop_path=DropPath(drop_path)
        self._norm_fact = 1 / sqrt(self.head)
        self.bn_attention=nn.BatchNorm2d(in_channel)

        self.Wq = nn.Conv1d(1,
                            1,
                            kernel_size=k_size,
                            padding=(k_size - 1) // 2,
                            bias=False)
        self.Wk = nn.Conv1d(1,
                            1,
                            kernel_size=k_size,
                            padding=(k_size - 1) // 2,
                            bias=False)
        nn.init.xavier_uniform_(self.Wq.weight)
        nn.init.xavier_uniform_(self.Wk.weight)
        self.Wv = nn.Conv2d(in_channel,
                            in_channel,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            groups=in_channel,
                            bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.avg_pool2 = nn.AvgPool2d(kernel_size=2,stride=2)

        self.head_att=nn.Linear(self.heads,self.heads,bias=False)
        nn.init.xavier_uniform_(self.head_att.weight)

        sequence = torch_log_uniform_sequence(0.7, 1/0.7, self.heads)
        self.norm_d = nn.Parameter(1/sequence.view(1,self.heads,1,1,1),requires_grad=False)
    
    def forward(self,x,info=None):
        b,c,h,w=x.shape
        
        if info is None:
            u=None
            key_direction=None
            z=None
        else:
            u,key_direction,z=info
            if u is not None: 
                if len(u.shape) != 4:
                    _b,_g,_d,_h,_w=u.shape
                else:
                    _b,_g,_h,_w=u.shape
                    _d=1
                if self.init_layer:
                    u=(u/z).view(_b,-1,_h,_w)
                    if _d*_g!=c:
                        u = self.adjust_channel(u)
                    u=self.u_bn(u)
                    u_k = self.u_Wk(self.avg_pool(u).view(b,1,c)).view(b,self.heads,self.head,1,1) # K: [b, 1, c]
                    z=torch.exp(torch.clamp(torch.norm(u_k,dim=2,keepdim=True)/self.norm_d, max=10))
                    u=u.view(b,self.heads,self.head,h,w)*z
                    key_direction=None
                u = u.view(b,self.heads,self.head,h,w)
        q = self.avg_pool(x) # [b, c, 1, 1]
        Q = self.Wq(q.view(b,1,c)).view(b,self.heads,self.head,1,1)
        K = self.Wk(q.view(b,1,c)).view(b,self.heads,self.head,1,1) # K: [b, 1, c]

        if key_direction == None:
            key_direction = K
        else:
            key_direction = key_direction + K

        k_norm=torch.norm(K,dim=2,keepdim=True)
        cur_k_norm=torch.exp(torch.clamp(k_norm/self.norm_d, max=10))
        if u == None:
            u_next = cur_k_norm*x.view(b,self.heads,self.head,h,w)
            z = cur_k_norm
        else:
            u_next = u + cur_k_norm*x.view(b,self.heads,self.head,h,w)
            z = z + cur_k_norm 
        
        we_sequence=u_next/(z+1e-3)
        output = self.Wv(we_sequence.view(b,c,h,w))
        
        h_att = torch.einsum('bhcij,bhcij->bhij', Q, K)/k_norm.view(b,self.heads,1,1)
        h_att = self.head_att(h_att.view(b,self.heads)).view(b,self.heads,1,1,1)
        output = (F.sigmoid(h_att)*output.view(b,self.heads,self.head,h,w)).view(b,c,h,w)
        output = self.drop_path(F.relu(self.bn_attention(output)))

        return output,[u_next,key_direction,z]

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, init_cell=False, drop_path = 0.0,block_idx=None):
        super(Bottleneck, self).__init__()
        self.block_idx = block_idx
        self.drop_path = DropPath(0.1) #if drop_path > 0. else nn.Identity()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.kcla=kcla_layer(planes * 4,planes * 2,heads=None,init_layer=init_cell,stride=2,padding=0,kernel_size=1,drop_path=0.2)
        self.avgpool2= nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x,info=None):
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
            residual = self.downsample(residual)
        out = self.relu(out + residual)
        att,info=self.kcla(out,info)
        out = out + att
        return out,info

class ResNet(nn.Module):

    def __init__(self, depth, num_classes=1000, block_name='BasicBlock', drop_path =0.0):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        if block_name.lower() == 'basicblock':
            assert (depth - 2) % 6 == 0, 'When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202'
            n = (depth - 2) // 6
            block = BasicBlock
        elif block_name.lower() == 'bottleneck':
            assert (depth - 2) % 9 == 0, 'When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199'
            n = (depth - 2) // 9
            block = Bottleneck
        else:
            raise ValueError('block_name shoule be Basicblock or Bottleneck')
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.head=16
        self.heads=8

        self.relu = nn.ReLU(inplace=True)
        
        stages = [None]*3
        stages[0] = self._make_layer(block, 16, n, stride=1,init_cell = True, drop_path=drop_path,block_idx=1)
        stages[1] = self._make_layer(block, 32, n, stride=2,init_cell=True, drop_path=drop_path,block_idx=2)
        stages[2] = self._make_layer(block, 64, n, stride=2,init_cell=True, drop_path=drop_path,block_idx=3)
        self.stages = nn.ModuleList(stages)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, init_cell=False, drop_path=0.0,block_idx=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, init_cell, drop_path = drop_path,block_idx=block_idx))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes,planes, drop_path = drop_path,block_idx=block_idx))

        return nn.ModuleList(layers)

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        info = None
        for index,layers in enumerate(self.stages):
            for layer in layers:
                x,info=layer(x,info)

        return x
        
    def forward(self, x):
        x = self.forward_features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def kcla(**kwargs):
    """
    Constructs a ResNet model.
    """
    print('--------Constructing KCLA--------')
    return ResNet(**kwargs)


if __name__ == '__main__':
    from thop import profile
    input_ = torch.randn(1,3,32,32)
    model = ResNet(depth=56,num_classes=100,block_name='Bottleneck',drop_path=0.2)
    flops, params = profile(model, inputs=(input_,))
    print(f"FLOPs: {flops/1e9:.2f}G, Params: {params/1e6:.2f}M")
