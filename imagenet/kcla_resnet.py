import torch
import torch.nn as nn
import torch.nn.functional as F

from math import log
from math import sqrt
import math
import time

__all__ = [
           'resnet50_kcla', 
           'resnet101_kcla', 
           ]

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
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
            self.head=64
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



def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
    
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, 
                 stride=1, downsample=None, 
                 SE=False, ECA_size=None, 
                 groups=1, base_width=64, dilation=1, 
                 norm_layer=nn.BatchNorm2d, drop_path=0.0):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            # norm_layer = nn.SyncBatchNorm 
            # # support for multi gpus
            # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
            
        width = int(planes * (base_width / 64.)) * groups
        
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = downsample
        self.stride = stride
        
        # channel attention modules
        self.se = None
        if SE:
            self.se = se_layer(planes * self.expansion, reduction=16)
        self.eca = None
        if ECA_size != None:
            self.eca = eca_layer(planes * self.expansion, int(ECA_size))
            

        
    def forward(self, x):
        identity = x
        
        # res block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        # channel attention
        if self.se != None:
            out = self.se(out)
        if self.eca != None:
            out = self.eca(out) 
        # downsampling for short cut    
        if self.downsample is not None:
            identity = self.downsample(identity)
            
        out += identity 
        out = self.relu(out)  
        return out, identity

class Attention(nn.Module):
    def __init__(self, ModuleList, block_idx):
        super(Attention, self).__init__()
        self.layers, self.attention = ModuleList
    def forward(self, x,info):
        for idx, (layer, attention) in enumerate(zip(self.layers, self.attention)):
            x, org = layer(x) 
            out,info = attention(x,info)
            x=x+out
        return x,info

class ResNet(nn.Module):
    def __init__(self, block, 
                layers, 
                num_classes=1000, 
                SE=False, 
                ECA=None, 
                zero_init_last_bn=True, #zero_init_residual=False,
                groups=1, 
                width_per_group=64, 
                replace_stride_with_dilation=None,
                norm_layer=nn.BatchNorm2d, 
                drop_rate=0.0, 
                drop_path=0.0
                ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            # norm_layer = nn.SyncBatchNorm
        self._norm_layer = norm_layer
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.drop_path = drop_path
        self.layer_attention = kcla_layer
        
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        
        if ECA is None:
            ECA = [None] * 4
        elif len(ECA) != 4:
            raise ValueError("argument ECA should be a 4-element tuple, got {}".format(ECA))
    
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool2= nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.layer1 = Attention( self._make_layer(block, 64, layers[0], SE=SE, ECA_size=ECA[0]),1)
        self.layer2 = Attention( self._make_layer(block, 128, layers[1], SE=SE, ECA_size=ECA[1], stride=2, dilate=replace_stride_with_dilation[0]),2)
        self.layer3 = Attention( self._make_layer(block, 256, layers[2], SE=SE, ECA_size=ECA[2], stride=2, dilate=replace_stride_with_dilation[1]),3)
        self.layer4 = Attention( self._make_layer(block, 512, layers[3], SE=SE, ECA_size=ECA[3], stride=2, dilate=replace_stride_with_dilation[2]),4)
        
        # classifier head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
            
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            # elif isinstance(m, (nn.SyncBatchNorm, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        # Zero-initialize the last BN in each residual branch
        if zero_init_last_bn:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, SE, ECA_size, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            ) # downsampling and change channels for x(identity)

        layers = []
        kcla=[]
        layers.append(block(self.inplanes, planes, stride, downsample, 
                            SE=SE, ECA_size=ECA_size, groups=self.groups,
                            base_width=self.base_width, dilation=previous_dilation, norm_layer=norm_layer, drop_path=self.drop_path))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, 
                                SE=SE, ECA_size=ECA_size, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, drop_path=self.drop_path))
        for i in range(0, blocks):
            init_cell=True if (i==0 or i==11) else False
            kcla.append(self.layer_attention(self.inplanes,self.inplanes//2,heads=None,init_layer=init_cell,stride=2,padding=0,kernel_size=1,drop_path=0.2))
        return nn.Sequential(*layers), nn.Sequential(*kcla)

    def forward_features(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        info = None

        x,info = self.layer1(x,info)

        x,info = self.layer2(x,info)

        x,info = self.layer3(x,info)

        x,info = self.layer4(x,info)
      
        return x
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if self.drop_rate:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        x = self.fc(x)
        return x



def resnet50_kcla(**kwargs):
    print("Constructing resnet50_kcla......")
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model

def resnet101_kcla(**kwargs):
    print("Constructing resnet101_kcla......")
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model

def resnet18_kcla(**kwargs):
    print("Constructing resnet18_kcla......")
    model = ResNet(Bottleneck, [1, 1, 1, 1], **kwargs)
    return model

if __name__ == '__main__':
    model = resnet101_kcla().cuda()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params / 1e6:.2f}M")
    
