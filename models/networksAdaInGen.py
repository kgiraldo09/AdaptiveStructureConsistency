from torch import nn
import torch
import torch.nn.functional as F

class SimpleAdaInGen(nn.Module):
    def __init__(self, input_dim, params, dim_=2):
        super(SimpleAdaInGen, self).__init__()
        dim_=dim_
        dim = params['dim']
        style_dim = params['style_dim']
        n_downsample = params['n_downsample']
        n_res = params['n_res']
        activ = params['activ']
        pad_type = params['pad_type']
        mlp_dim = params['mlp_dim']

        self.enc = AdaEncoder(n_downsample, n_res, input_dim, dim, 'in', activ, pad_type=pad_type, dim_=dim_)
        self.dec = Decoder(n_downsample, n_res, self.enc.output_dim, input_dim, res_norm='adain', activ='sigmoid',
                           pad_type=pad_type, dim_=dim_)

        self.mlp_enc = MLP(style_dim, self.get_num_adain_params(self.enc), mlp_dim, 3, norm='none', activ=activ)
        self.mlp_dec = MLP(style_dim, self.get_num_adain_params(self.dec), mlp_dim, 3, norm='none', activ=activ)

    def forward(self, images, bvec):
        adain_params_enc = self.mlp_enc(bvec)
        self.assign_adain_params(adain_params_enc, self.enc)
        content = self.enc(images)
        adain_params_dec = self.mlp_dec(bvec)
        self.assign_adain_params(adain_params_dec, self.dec)
        images_recon = self.dec(content)
        return images_recon

    def assign_adain_params(self, adain_params, model):
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2*m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2*m.num_features:
                    adain_params = adain_params[:, 2*m.num_features:]

    def get_num_adain_params(self, model):
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2*m.num_features
        return num_adain_params

class ResAdaInGen(SimpleAdaInGen):
    def __init__(self, input_dim, output_dim, params):
        super(SimpleAdaInGen, self).__init__()
        dim = params['dim']
        style_dim = params['style_dim']
        n_downsample = params['n_downsample']
        n_res = params['n_res']
        activ = params['activ']
        pad_type = params['pad_type']
        mlp_dim = params['mlp_dim']
        input_domain = params['input_domain']

        self.enc = AdaEncoder(n_downsample, n_res, input_dim, dim, 'in', activ, pad_type=pad_type)  # "in -> adain"

        self.dec = Decoder(n_downsample, n_res, self.enc.output_dim * input_domain, output_dim, res_norm='in', activ='none',
                           pad_type=pad_type)

        self.mlp_enc = MLP(style_dim, self.get_num_adain_params(self.enc), mlp_dim, 3, norm='none', activ=activ)
        self.mlp_dec = MLP(style_dim, self.get_num_adain_params(self.dec), mlp_dim, 3, norm='none', activ=activ)

        self.multi_modal_attention = MultiModalAttentionLayer2(input_domain)
        self.dim_reduce_conv = nn.Sequential(
            nn.Conv2d(dim * 4 * input_domain, dim * 4, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):#, bvec):
        #adain_params_enc = self.mlp_enc(bvec)
        #self.assign_adain_params(adain_params_enc, self.enc)
        #b0, t2, t1 = torch.split(input, 1, dim=1)
        content_b0 = self.enc(input)
        #content_t2 = self.enc(t2)
        #content_t1 = self.enc(t1)
        #content_all = self.multi_modal_attention([content_b0, content_t2, content_t1])SS
        content_all = self.multi_modal_attention(content_b0)
        #adain_params_dec = self.mlp_dec(bvec)
        #self.assign_adain_params(adain_params_dec, self.dec)
        dwi = self.dec(content_all)
        return dwi


class ResAdaInGen2(SimpleAdaInGen):
    def __init__(self, input_dim, output_dim, params, dim_=2):
        super(SimpleAdaInGen, self).__init__()
        dim_=dim_
        dim = params['dim']
        style_dim = params['style_dim']
        n_downsample = params['n_downsample']
        n_res = params['n_res']
        activ = params['activ']
        pad_type = params['pad_type']
        mlp_dim = params['mlp_dim']
        input_domain = params['input_domain']

        self.enc = AdaEncoder(n_downsample, n_res, input_dim, dim, 'in', activ, pad_type=pad_type, dim_=dim_)  # "in -> adain"

        self.dec = Decoder(n_downsample, n_res, self.enc.output_dim * input_domain, output_dim, res_norm='in', activ='none',
                           pad_type=pad_type, dim_ = dim_)
        self.decA = Decoder(n_downsample, n_res, self.enc.output_dim * input_domain, output_dim, res_norm='in', activ='none',
                           pad_type=pad_type, sigmoid=True, dim_ = dim_)

        self.mlp_enc = MLP(style_dim, self.get_num_adain_params(self.enc), mlp_dim, 3, norm='none', activ=activ)
        self.mlp_dec = MLP(style_dim, self.get_num_adain_params(self.dec), mlp_dim, 3, norm='none', activ=activ)

        self.multi_modal_attention = MultiModalAttentionLayer2(dim_=dim_)#MultiModalAttentionLayer2(input_domain)
        self.dim_reduce_conv = nn.Sequential(
            nn.Conv2d(dim * 4 * input_domain, dim * 4, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, input, outA_bool = False):#, bvec):
        #adain_params_enc = self.mlp_enc(bvec)
        #self.assign_adain_params(adain_params_enc, self.enc)
        #b0, t2, t1 = torch.split(input, 1, dim=1)
        content_b0 = self.enc(input)
        #content_t2 = self.enc(t2)
        #content_t1 = self.enc(t1)
        #content_all = self.multi_modal_attention([content_b0, content_t2, content_t1])SS
        content_all = self.multi_modal_attention(content_b0)
        #adain_params_dec = self.mlp_dec(bvec)
        #self.assign_adain_params(adain_params_dec, self.dec)
        dwiI = self.dec(content_all)
        if outA_bool:
            dwiA = self.decA(content_all)
            return dwiI, dwiA
        return dwiI





class AdaEncoder(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type, dim_=2):
        super(AdaEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type, dim_ = dim_)]
        for i in range(n_downsample):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type, dim_ = dim_)]
            self.model.append(SingleModalAttentionLayer(2 * dim, dim_ = dim_))
            dim *= 2
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)

class Decoder(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, res_norm='adain', activ='tanh', pad_type='zero', sigmoid=False, dim_ = 2):
        super(Decoder, self).__init__()

        self.model = []
        self.model += [ResBlocks(n_res, dim, res_norm, activation='relu', pad_type=pad_type, dim_ = dim_)]
        self.model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type, dim_ = dim_)]

        for i in range(n_upsample):
            self.model += [nn.Upsample(scale_factor=2, mode='nearest'),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation='relu', pad_type=pad_type, dim_ = dim_)]
            self.model.append(SingleModalAttentionLayer(dim // 2, dim_ = dim_))
            dim //= 2

        self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation=activ, pad_type=pad_type, sigmoid=sigmoid, dim_ = dim_)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero', dim_ = 2):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type, dim_ = dim_)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim, n_blk, norm='none', activ='relu'):

        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(input_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model = nn.Sequential(*self.model)
        self.final = LinearBlock(dim, output_dim, norm='none', activation='none') # no output activations

    def forward(self, x):
        self.feature = self.model(x.view(x.size(0), -1))
        out = self.final(self.feature)
        return out


class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero', dim_ = 2):
        super(ResBlock, self).__init__()
        model = []
        model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type, dim_=dim_)]
        model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type, dim_=dim_)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out = out + residual
        return out

class SingleModalAttentionLayer(nn.Module):
    def __init__(self, channel, reduction=16, dim_ = 2):
        super(SingleModalAttentionLayer, self).__init__()
        if dim_ ==3:
            self.avg_pool = nn.AdaptiveAvgPool3d(1)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dim_ = dim_
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        if self.dim_ ==3:
            b, c, _, _, _ = x.size()
            y = self.avg_pool(x).view(b, c)
            y = self.fc(y).view(b, c, 1, 1, 1)
        else:
            b, c, _, _ = x.size()
            y = self.avg_pool(x).view(b, c)
            y = self.fc(y).view(b, c, 1, 1)
        return x + x * y.expand_as(x)

class MultiModalAttentionLayer(nn.Module):
    def __init__(self, modal_num=1, input_size=64, reduction=16):
        super(MultiModalAttentionLayer, self).__init__()
        self.modal_num = modal_num
        input_channel = input_size**2 * modal_num

        self.fc = nn.Sequential(
            nn.Linear(input_channel, input_channel//reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(input_channel//reduction, modal_num, bias=False),
            nn.Sigmoid()
        )
        self.sm = nn.Softmax(dim=2)
        self.l1 = nn.Linear(256 * 3, 256)

    def forward(self, input):#, feature_list):
        b, c, h, w = input.size()
        px = input.view(b, c, -1)
        #px = torch.cat(px, dim=-1)
        attention = self.fc(px)
        attention = self.sm(attention)
        attention = torch.transpose(attention, 1, 2)

        x = input
        out = x * attention[:, :, :, None, None]
        out = torch.sum(out, dim=1)

        input = input * 0.9 + out * 0.1

        return input


class MultiModalAttentionLayer2(nn.Module):
    """
    Single-modal spatial attention (CBAM-style), robust to any batch size.
    Replaces the previous buggy implementation which assumed specific shapes.
    Input:  (B, C, H, W)
    Output: (B, C, H, W)  (same shape)
    """

    def __init__(self, dim_ = 2):
        super(MultiModalAttentionLayer2, self).__init__()
        # 7x7 conv as in CBAM spatial attention
        if dim_ ==3:
            self.spatial_att = nn.Sequential(
                nn.Conv3d(2, 1, kernel_size=7, stride=1, padding=3, bias=False),
                nn.Sigmoid()
            )
        else:
            self.spatial_att = nn.Sequential(
                nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False),
                nn.Sigmoid()
            )

    def forward(self, x):
        # x: (B, C, H, W)
        # 1) channel-wise pooling -> two maps (avg, max): (B, 1, H, W)
        avg_pool = torch.mean(x, dim=1, keepdim=True)          # (B,1,H,W)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)        # (B,1,H,W)

        # 2) concat -> conv -> sigmoid -> spatial attention map (B,1,H,W)
        att_map = self.spatial_att(torch.cat([avg_pool, max_pool], dim=1))  # (B,1,H,W)

        # 3) apply attention per-channel and mix with original input (keeping your 0.9/0.1 blend)
        attended = x * att_map    # broadcasting to (B,C,H,W)
        out = x * 0.9 + attended * 0.1
        return out

class MultiModalAttentionLayer3(nn.Module):
    """
    Single-modal spatial attention (CBAM-style), robust to any batch size.
    Replaces the previous buggy implementation which assumed specific shapes.
    Input:  (B, C, H, W)
    Output: (B, C, H, W)  (same shape)
    """
    def __init__(self, channels, r):
        super(MultiModalAttentionLayer3, self).__init__()
        self.r = r
        self.sam = SAM(bias=False)
        self.cam = CAM(channels=self.channels, r=self.r)

    def forward(self, x):
        output = self.cam(x)
        output = self.sam(output)
        return output + x



class SAM(nn.Module):
    def __init__(self, bias=False):
        super(SAM, self).__init__()
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1, bias=self.bias)

    def forward(self, x):
        max = torch.max(x,1)[0].unsqueeze(1)
        avg = torch.mean(x,1).unsqueeze(1)
        concat = torch.cat((max,avg), dim=1)
        output = self.conv(concat)
        output = F.sigmoid(output) * x 
        return output 

class CAM(nn.Module):
    def __init__(self, channels, r):
        super(CAM, self).__init__()
        self.channels = channels

        self.linear = nn.Sequential(
            nn.Linear(in_features=self.channels, out_features=self.channels//self.r, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.channels//self.r, out_features=self.channels, bias=True))

    def forward(self, x):
        max = F.adaptive_max_pool2d(x, output_size=1)
        avg = F.adaptive_avg_pool2d(x, output_size=1)
        b, c, _, _ = x.size()
        linear_max = self.linear(max.view(b,c)).view(b, c, 1, 1)
        linear_avg = self.linear(avg.view(b,c)).view(b, c, 1, 1)
        output = linear_max + linear_avg
        output = F.sigmoid(output) * x
        return output
    
class CBAM(nn.Module):
    def __init__(self, channels, r):
        super(CBAM, self).__init__()
        self.channels = channels
        self.r = r
        self.sam = SAM(bias=False)
        self.cam = CAM(channels=self.channels, r=self.r)

    def forward(self, x):
        output = self.cam(x)
        output = self.sam(output)
        return output + x







class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero', sigmoid=False, dim_=2):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            if dim_ == 3:
                self.pad = nn.ReflectionPad3d(padding)
            else:
                self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            if dim_ == 3:
                self.pad = nn.ReplicationPad3d(padding)
            else:
                self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            if dim_ == 3:   
                self.pad = nn.ZeroPad3d(padding)
            else:
                self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            if dim_ == 3:
                self.norm = nn.BatchNorm3d(norm_dim)
            else:
                self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            if dim_ == 3:
                self.norm = nn.InstanceNorm3d(norm_dim)
            else:
                self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        elif activation == 'softmax':
            self.activation = nn.Softmax(dim=1)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            assert 0, "Unsupported activation: {}".format(activation)
        if sigmoid:
            self.activation = nn.Sigmoid()
        if dim_ == 3:
            self.conv = nn.Conv3d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out

class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = None
        self.bias = None
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'

class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        if x.size(0) == 1:
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x