import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log
from math import sqrt



__all__ = [
           'resnet50_kcla', 
           'resnet101_kcla', 
           'resnet18_kcla',
           ]
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
    def __init__(self,in_channel,sv_channel,heads=None,init_layer=False,stride=1,padding=0,kernel_size=1,drop_path=0.2,cross_layer=True,conv_adjust=True,usegate=True):
        super().__init__()

        if heads!=None:
            self.heads=heads
            self.dkeysize= in_channel//self.heads
        else:
            self.dkeysize= 16
            self.heads=in_channel//self.dkeysize
        
        self.init_layer=init_layer
        self.cross_layer=cross_layer
        self.conv_adjust=conv_adjust
        self.usegate=usegate
        self.sv_channel=sv_channel
        if self.init_layer and self.sv_channel!=None:
            if self.conv_adjust:
                self.adjust_channel=nn.Sequential(
                    nn.Conv2d(sv_channel,in_channel,kernel_size=1,stride=stride,padding=0,groups=1),
                    nn.BatchNorm2d(in_channel)
                )

        self.drop_path=DropPath(drop_path)
        self._norm_fact = 1 / sqrt(self.dkeysize)
        self.bn_attention=nn.BatchNorm2d(in_channel)
        t = int(abs((log(in_channel, 2) + 1) / 2.))
        k_size = t if t % 2 else t + 1
        self.k_size = k_size
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
        self.Wv = nn.Conv2d(in_channel,
                            in_channel,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            groups=in_channel,
                            bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.avg_pool2 = nn.AvgPool2d(kernel_size=2,stride=2)
        sequence = torch_log_uniform_sequence(0.7, 1/0.7, self.heads)
        print(sequence)
        self.norm_d = nn.Parameter(1/sequence.view(1,self.heads,1,1,1),requires_grad=False)

    def forward(self,x,info=None,normd=None):
        if info is None:
            kvpre=None
            pre_k_norm=None
            mem_norm=None
        else:
            kvpre,pre_k_norm,mem_norm=info

        b,c,h,w=x.shape
        # 如果输入的x的宽度与kvpre的宽度不一致，则需要对mem进行处理
        if self.init_layer and kvpre is not None:
            if self.cross_layer:
                _b,_g,_head,_h,_w=kvpre.shape
                kvpre=(kvpre/mem_norm).view(_b,_g*_head,_h,_w)
                if self.conv_adjust:
                    kvpre=self.adjust_channel(kvpre)
                else:
                    kvpre=self.avg_pool2(kvpre).repeat_interleave(2,dim=1)
                pre_k=self.Wk(self.avg_pool(kvpre).view(b,1,c)).view(_b,self.heads,self.dkeysize,1,1)
                pre_k_norm=torch.exp(torch.clamp(torch.norm(pre_k,p=2,dim=2,keepdim=True)*self.norm_d, max=10))
                mem_norm=pre_k_norm
                kvpre=kvpre.view(b,self.heads,self.dkeysize,h,w)*pre_k_norm
            else:
                kvpre=None
                pre_k_norm=None
                mem_norm=None

        q = self.avg_pool(x) # [b, c, 1, 1]
        Q = self.Wq(q.view(b,1,c)) # Q: [b, 1, c]
        K = self.Wk(q.view(b,1,c)) # K: [b, 1, c]

        #计算当前的k的norm
        cur_k=K.view(b,self.heads,-1,1,1)
        k_norm=torch.norm(cur_k, p=2, dim=2, keepdim=True)
        cur_k_norm=torch.exp(torch.clamp(k_norm*self.norm_d, max=10))
    
        if kvpre!=None:  
            #lambda_= pre_k_norm/(cur_k_norm)
            kvbinding = kvpre+cur_k_norm*x.view(b,self.heads,self.dkeysize,h,w)
            mem_norm = mem_norm+cur_k_norm
            token_x=(kvbinding/mem_norm).view(b,c,h,w)
        else:
            token_x=x
            kvbinding=cur_k_norm*x.view(b,self.heads,self.dkeysize,h,w)
            mem_norm=cur_k_norm

        V=self.Wv(token_x)
        Q=Q.view(b,c//self.dkeysize,1,self.dkeysize) #[b, g, c/g, 1, 1, 1]
        K=K.view(b,c//self.dkeysize,1,self.dkeysize) #[b, g, c/g, 1, 1, 1]
        V=V.view(b,c//self.dkeysize,self.dkeysize,h,w)  # (b,g,1,c/g or dkey,h,w)   

        attn = torch.einsum('... i d, ... j d -> ... i j', Q, K)/k_norm.view(b,self.heads,1,1)
        attn = F.sigmoid(attn) # [b, g, 1, 1, 1]
        output = V * attn.view(b, c//self.dkeysize, 1, 1, 1).expand_as(V) # [b, g, c/g, h, w]
        output = output.view(b, c, h, w)
        output = self.drop_path(self.bn_attention(output))

        return output,[kvbinding,cur_k_norm,mem_norm]


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
    
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, SE=False, ECA_size=None, groups=1, base_width=64, dilation=1, norm_layer=nn.BatchNorm2d, drop_path=0.0):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)   
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1) 
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):  
        identity = x  
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None: 
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



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
        # self.se = None
        # if SE:
        #     self.se = se_layer(planes * self.expansion, reduction=16)
        # self.eca = None
        # if ECA_size != None:
        #     self.eca = eca_layer(planes * self.expansion, int(ECA_size))
               
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
            x=layer(x) 
            out,info = attention(x,info)
            x=x+out
            
        return x,info

    

class ResNet(nn.Module):
    def __init__(self, block, 
                layers, 
                num_classes=1000, 
                SE=False, 
                ECA=None, 
                zero_init_last_bn=True,
                groups=1, 
                width_per_group=64, 
                replace_stride_with_dilation=None,
                norm_layer=nn.BatchNorm2d, 
                drop_rate=0.0, 
                drop_path=0.0,
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
        
        # self.inplanes = 64
        self.inplanes = 16
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
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.bn2 = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.avgpool2= nn.AvgPool2d(kernel_size=2, stride=2)

        self.layer1 = Attention(self._make_layer(block, 32, layers[0], SE=SE, ECA_size=ECA[0],stride=2),1)  # 128/4
        self.layer2 = Attention(self._make_layer(block, 64, layers[1], SE=SE, ECA_size=ECA[1], stride=2, dilate=replace_stride_with_dilation[0]),2)  # 256/4
        self.layer3 = Attention(self._make_layer(block, 128, layers[2], SE=SE, ECA_size=ECA[2], stride=2, dilate=replace_stride_with_dilation[1]),3)  # 512/4
        self.layer4 = Attention(self._make_layer(block, 128, layers[3], SE=SE, ECA_size=ECA[3], stride=1, dilate=replace_stride_with_dilation[2]),4)  # 512/4

        self.layer_out=[]
            
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

        inplanes=self.inplanes
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
            init_cell=True if i==0 else False
            kcla.append(self.layer_attention(self.inplanes,inplanes,heads=None,init_layer=init_cell,stride=stride,padding=0,kernel_size=1,drop_path=0.2,cross_layer=True,conv_adjust=True,usegate=False))
        return nn.Sequential(*layers), nn.Sequential(*kcla)

    def forward_features(self, x):
        info=None
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.maxpool(x)
        self.layer_out.append(x)
        
        x,info = self.layer1(x,info)
        self.layer_out.append(x)

        x,info = self.layer2(x,info)
        self.layer_out.append(x)

        x,info = self.layer3(x,info)
        self.layer_out.append(x)

        x,info = self.layer4(x,info)
        self.layer_out.append(x)


        return x
    
    def forward(self, x):
        self.layer_out=[]
        x = self.forward_features(x)
        return self.layer_out



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
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model

if __name__ == "__main__":
    from thop import profile
    from thop import clever_format
    model = resnet18_kcla()

    input_size=(1, 3, 256, 256)
    input = torch.randn(input_size)
    
    flops, params = profile(model, inputs=(input,))
    flops, params = clever_format([flops, params], "%.3f")
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Input size: {input_size}")
    print(f"FLOPs: {flops}")
    print(f"Params: {params}")

