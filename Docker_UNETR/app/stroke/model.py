import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange

"""v0版本(瓶颈层的分辨率低高)：skip-connect是4次, 上下采样均为4次"""
class BaseFeatureExtraction(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 ffn_expansion_factor=1.,  
                 qkv_bias=False,):
        super(BaseFeatureExtraction, self).__init__()
        self.norm1 = LayerNorm(dim, 'WithBias')
        self.attn = AttentionBase(dim, num_heads=num_heads, qkv_bias=qkv_bias,)
        self.norm2 = LayerNorm(dim, 'WithBias')
        self.mlp = Mlp(in_features=dim,
                       ffn_expansion_factor=ffn_expansion_factor,)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
    
class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
    
import numbers
##########################################################################
## Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)    

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias
    
    
class AttentionBase(nn.Module):
    def __init__(self,
                 dim,   
                 num_heads=8,
                 qkv_bias=False,):
        super(AttentionBase, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv1 = nn.Conv2d(dim, dim*3, kernel_size=1, bias=qkv_bias)
        self.qkv2 = nn.Conv2d(dim*3, dim*3, kernel_size=3, padding=1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        b, c, h, w = x.shape
        qkv = self.qkv2(self.qkv1(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.proj(out)
        return out    
 
class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, 
                 in_features, 
                 hidden_features=None, 
                 ffn_expansion_factor = 2,
                 bias = False):
        super().__init__()
        hidden_features = int(in_features*ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            in_features, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, in_features, kernel_size=1, bias=bias)
    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

       
class DetailFeatureExtraction(nn.Module):
    def __init__(self, dim, num_layers=3):
        super(DetailFeatureExtraction, self).__init__()
        INNmodules = [DetailNode(dim) for _ in range(num_layers)]
        self.net = nn.Sequential(*INNmodules)
    def forward(self, x):
        z1, z2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:x.shape[1]]

        for layer in self.net:
            z1, z2 = layer(z1, z2)
        return torch.cat((z1, z2), dim=1)    
    
class DetailNode(nn.Module):
    def __init__(self, dim):
        super(DetailNode, self).__init__()
        # Scale is Ax + b, i.e. affine transformation
        self.theta_phi = InvertedResidualBlock(inp=dim//2, oup=dim//2, expand_ratio=2)
        self.theta_rho = InvertedResidualBlock(inp=dim//2, oup=dim//2, expand_ratio=2)
        self.theta_eta = InvertedResidualBlock(inp=dim//2, oup=dim//2, expand_ratio=2)
        self.shffleconv = nn.Conv2d(dim, dim, kernel_size=1,
                                    stride=1, padding=0, bias=True)
    def separateFeature(self, x):
        z1, z2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:x.shape[1]]
        return z1, z2
    def forward(self, z1, z2):
        z1, z2 = self.separateFeature(
            self.shffleconv(torch.cat((z1, z2), dim=1)))
        z2 = z2 + self.theta_phi(z1)
        z1 = z1 * torch.exp(self.theta_rho(z2)) + self.theta_eta(z2)
        return z1, z2
    
class InvertedResidualBlock(nn.Module):
    def __init__(self, inp, oup, expand_ratio):
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = int(inp * expand_ratio)
        self.bottleneckBlock = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, 1, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # dw
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_dim, hidden_dim, 3, groups=hidden_dim, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            # nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True)
        )
    def forward(self, x):
        return self.bottleneckBlock(x)    
    

class Encoder(nn.Module):
    def __init__(self, n_channels, bilinear=True):
        super(Encoder, self).__init__()
        self.n_channels = n_channels

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        
        self.Base = BaseFeatureExtraction(dim=512, num_heads=8)
        self.Detail = DetailFeatureExtraction(dim=512,num_layers=3)
        
        
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        base_feature = self.Base(x5)
        detail_feature = self.Detail(x5)
        
        x5 = torch.cat((base_feature, detail_feature), dim=1)
        
        return x5,x4,x3,x2,x1,base_feature,detail_feature


class Decoder(nn.Module):
    def __init__(self, n_classes, bilinear=True):
        super(Decoder, self).__init__()
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.up1 = Up(1536, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x5, x4, x3, x2, x1):
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)

        return torch.sigmoid(x)

    

    
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

"""抽取每个模态的鉴别性特征和位置""" 
"""Step 1.计算得分每个模态鉴别性特征的重要性score, 原始的三个模态的数据分布不同并且编码器不共享, 那么得到的特征的分布是否不相同, 者是否会有影响。
   得到的模态特异的特征的分布是相似的, 因为detail_layer最后有ReLu6"""

#1. 计算每个模态的得分Mask
class Score_PredictorConv(nn.Module):
    def __init__(self, embed_dim=512, num_modals=3):
        super().__init__()
        self.num_modals = num_modals
        self.score_nets = nn.ModuleList([nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, groups=(embed_dim)),
            nn.Conv2d(embed_dim, 1, 1),
            nn.Sigmoid()
        )for _ in range(num_modals)])
        
    def forward(self, x):
        B, C, H, W = x[0].shape
        x_ = [torch.zeros((B, 1, H, W)) for _ in range(self.num_modals)]
        for i in range(self.num_modals):
            x_[i] = self.score_nets[i](x[i])
        return x_
    
#2. 通过利用mask和跨模态比较得到最终的鉴别性mask, 加法的时候用平均表示torch.mean, 以至于最后能得到位置-level的mask
def Dis_mask_Select(x_ext, module):    
        x_scores = module(x_ext) 
        x_value = []
        for i in range(len(x_ext)):
            x_value.append(x_scores[i] * torch.mean(x_ext[i], dim=1, keepdim=True) + torch.mean(x_ext[i], dim=1, keepdim=True))
        
        """stack操作会增加一个新的维度, 比如原始的三个tensor为2,1,14,14那么选择堆叠的维度在dim=2, 则堆叠后的维度为2,1,3,14,14
           然后求argmax也保持dim=2, 则比较的是原始的三个tensor中, 对于两个batchsize中, 在每个对应的batch/sample上,
           三个模态的14*14范围内的每个坐标进行比较, 并返回三个模态中最大值的位置索引"""
        x_f = torch.stack([x_value[0], x_value[1], x_value[2]], dim=2).argmax(dim=2)
        
        """然后将argmax的结果转变为one-hot形式, 表明每个模态的不同贡献"""
        mask_a = torch.where(x_f == 0, 1, 0)
        mask_d = torch.where(x_f == 1, 1, 0)
        mask_f = torch.where(x_f == 2, 1, 0)
        
        """得到每个模态中的每个图中位置最具有鉴别性的位置, 相当于将一张图三个区域, 分为指示了adc, dwi 和flair的鉴别性的位置区域"""
        return [mask_a,mask_d,mask_f]
    
"""瓶颈层的特征图大小: B, C(512), W(14), H(14)"""

"""Step 2.利用cross-attention从多模态融合图像中得到"""
class CrossAttention(nn.Module):
    def __init__(self,
                 dim,   
                 num_heads=8,
                 qkv_bias=False,):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv1 = nn.Conv2d(dim, dim*3, kernel_size=1, bias=qkv_bias)
        self.qkv2 = nn.Conv2d(dim*3, dim*3, kernel_size=3, padding=1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
    
    #交差注意力, q为融合前的单模态特征, kv为融合后的多模态特征
    def forward(self, q, kv):
        # [batch_size, num_patches + 1, total_embed_dim]
        b, c, h, w = q.shape
        q = q
        k = kv
        v = kv
#         qkv = self.qkv2(self.qkv1(x))
#         print(qkv.shape)
#         q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.proj(out)
        return out   

"""调用上面两个Step: 入参是融合前的单模态特征F_t_list=[], 和融合后的多模态特征"""
def Generate_CorFeature_and_Return_CorScore(F_t_list, F_t_fusion):

        channel_dim = F_t_fusion.shape[1]
        
        #定义模态得分计算器，并计算得分mask
        Score_Pred = Score_PredictorConv(channel_dim, num_modals = 3)
        Score_Pred = Score_Pred.cuda()
        Score_Pred = nn.DataParallel(Score_Pred, device_ids=[0])  
        discri_mask = Dis_mask_Select(F_t_list,Score_Pred)
        
        #从多模态融合特征中提取对应的模态特征
        Cross_Attention = CrossAttention(channel_dim,num_heads = 4)
        Cross_Attention = Cross_Attention.cuda()
        Cross_Attention = nn.DataParallel(Cross_Attention, device_ids=[0]) 
        content = []
        for i in range(0,len(F_t_list)):
            content.append(Cross_Attention(F_t_list[i], F_t_fusion)) 
        
        #mask*抽取的单模态特征: 计算鉴别性特征
        discri_content = []
        for i in range(0,len(F_t_list)):
            discri_content.append(discri_mask[i]*content[i]) 
        
        #返回每个模态的鉴别性特征 和 每个模态对应的鉴别位置(cs,1,W,H)
        return discri_content, discri_mask    

"""分别计算不同模态在, 跨模态注意力的提取的特征"""
def Generate_CorFeature(F_t_list, F_t_Fusion):
  
        #从多模态融合特征中提取对应的模态特征
        Cross_Attention = CrossAttention(F_t_Fusion.shape[1],num_heads = 4)
        Cross_Attention = Cross_Attention.cuda()
        Cross_Attention = nn.DataParallel(Cross_Attention, device_ids=[0]) 
        content = []
        for i in range(0,len(F_t_list)):
            content.append(Cross_Attention(F_t_list[i], F_t_Fusion)) 
        
        return content    
    
"""投影头, 推理时输入所有模态的特征, 是一个list, 然后分别推理然后打包成list返回"""
class Projection_Head(nn.Module):
    def __init__(self,):
        super().__init__()

        self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 128)
            )

#         self.apply(InitWeights_He(4e-3))

    def forward(self, x_list):
        
        rep_list = []
        for x in x_list:
            rep_list.append(F.normalize(self.head(x), dim=1))

        return rep_list 
    
# if __name__ == '__main__':

# #     print(config[
#     input = torch.randn(16,1,256,256)

#     encoder = Encoder(1)
#     decoder = Decoder(1)
    
#     x5,x4,x3,x2,x1,_,_ = encoder(input)
#     out = decoder(x5,x4,x3,x2,x1)
#     print(out.shape)
    