import torch
import torch.nn as nn
import torch.nn.functional as F


def sequential(*mods):
    return nn.Sequential(*mods)


def conv(
    in_channels=64,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=True,
    mode="CBR",
    negative_slope=0.2,
):
    layers = []
    for token in mode:
        if token == "C":
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias))
        elif token == "T":
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias))
        elif token == "B":
            layers.append(nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-4, affine=True))
        elif token == "I":
            layers.append(nn.InstanceNorm2d(out_channels, affine=True))
        elif token == "R":
            layers.append(nn.ReLU(inplace=True))
        elif token == "r":
            layers.append(nn.ReLU(inplace=False))
        elif token == "L":
            layers.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        elif token == "l":
            layers.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=False))
        elif token in {"2", "3", "4"}:
            layers.append(nn.PixelShuffle(upscale_factor=int(token)))
        elif token == "U":
            layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
        elif token == "u":
            layers.append(nn.Upsample(scale_factor=3, mode="nearest"))
        elif token == "v":
            layers.append(nn.Upsample(scale_factor=4, mode="nearest"))
        elif token == "M":
            layers.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        elif token == "A":
            layers.append(nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        else:
            raise NotImplementedError(f"Undefined layer mode: {token}")
    return sequential(*layers)


def downsample_strideconv(in_channels=64, out_channels=64, mode="2", bias=True):
    scale = int(mode[0])
    return conv(in_channels, out_channels, kernel_size=scale, stride=scale, padding=0, bias=bias, mode="C")


def upsample_pixelshuffle(in_channels=64, out_channels=3, mode="2", bias=True):
    scale = int(mode[0])
    return conv(
        in_channels,
        out_channels * scale**2,
        kernel_size=1,
        stride=1,
        padding=0,
        bias=bias,
        mode="C" + mode,
    )


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        y = (x - mean) / torch.sqrt(var + eps)
        ctx.save_for_backward(y, var, weight)
        return weight.view(1, -1, 1, 1) * y + bias.view(1, -1, 1, 1)

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        y, var, weight = ctx.saved_tensors
        grad = grad_output * weight.view(1, -1, 1, 1)
        mean_grad = grad.mean(dim=1, keepdim=True)
        mean_grad_y = (grad * y).mean(dim=1, keepdim=True)
        grad_x = (grad - y * mean_grad_y - mean_grad) / torch.sqrt(var + eps)
        grad_weight = (grad_output * y).sum(dim=(0, 2, 3))
        grad_bias = grad_output.sum(dim=(0, 2, 3))
        return grad_x, grad_weight, grad_bias, None


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.0):
        super().__init__()
        dw_channel = c * DW_Expand
        ffn_channel = c * FFN_Expand

        self.conv1 = nn.Conv2d(c, dw_channel, kernel_size=1, padding=0, stride=1, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, kernel_size=1, padding=0, stride=1, bias=True)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, kernel_size=1, padding=0, stride=1, bias=True),
        )
        self.sg = SimpleGate()
        self.conv4 = nn.Conv2d(c, ffn_channel, kernel_size=1, padding=0, stride=1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, kernel_size=1, padding=0, stride=1, bias=True)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma


class MultiExposureFusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.att11 = nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=3, padding=1, bias=True)
        self.att12 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1, bias=True)
        self.att31 = nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=3, padding=1, bias=True)
        self.att32 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1, bias=True)
        self.relu = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(in_channels * 3, in_channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x1, x2, x3):
        f1_att = torch.sigmoid(self.att12(self.relu(self.att11(torch.cat((x1, x2), dim=1)))))
        f3_att = torch.sigmoid(self.att32(self.relu(self.att31(torch.cat((x3, x2), dim=1)))))
        return self.conv2(torch.cat((x1 * f1_att, x2, x3 * f3_att), dim=1))


class NAFNet(nn.Module):
    def __init__(
        self,
        img_channel=6,
        width=64,
        middle_blk_num=1,
        enc_blk_nums=None,
        dec_blk_nums=None,
        down_scales=None,
        channels=None,
        seg_head_channels=16,
        seg_output_channels=1,
        use_mask_head=True,
        freeze_backbone=True,
    ):
        super().__init__()
        enc_blk_nums = [2, 4, 4, 8] if enc_blk_nums is None else enc_blk_nums
        dec_blk_nums = [2, 2, 2, 2] if dec_blk_nums is None else dec_blk_nums
        down_scales = [2, 2, 2, 4] if down_scales is None else down_scales
        channels = [64, 128, 256, 512, 1024] if channels is None else channels

        assert len(down_scales) == len(enc_blk_nums) == len(dec_blk_nums)
        assert len(channels) == len(down_scales) + 1

        self.use_mask_head = use_mask_head
        self.seg_head_channels = seg_head_channels
        self.seg_output_channels = seg_output_channels
        self.intro_0 = nn.Conv2d(img_channel, channels[0], kernel_size=3, padding=1, stride=1, bias=True)
        self.intro_1 = nn.Conv2d(img_channel, channels[0], kernel_size=3, padding=1, stride=1, bias=True)
        self.intro_2 = nn.Conv2d(img_channel, channels[0], kernel_size=3, padding=1, stride=1, bias=True)
        self.ending = nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=1, stride=1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = channels[0]
        for i, num_blocks in enumerate(enc_blk_nums):
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num_blocks)]))
            next_chan = channels[i + 1]
            scale = down_scales[i]
            self.downs.append(downsample_strideconv(chan, next_chan, mode=str(scale)))
            chan = next_chan

        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])

        for i, num_blocks in enumerate(dec_blk_nums):
            scale = down_scales[::-1][i]
            next_chan = channels[-2 - i]
            self.ups.append(upsample_pixelshuffle(chan, next_chan, mode=str(scale), bias=False))
            chan = next_chan
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num_blocks)]))

        self.padder_size = 1
        for scale in down_scales:
            self.padder_size *= scale

        self.fusion = MultiExposureFusion(channels[0])
        self.outconv = nn.Conv2d(channels[0], 3, kernel_size=3, padding=1, stride=1, bias=True)
        if self.use_mask_head:
            self.encoder_adapt_convs = nn.ModuleList([
                nn.Conv2d(c, seg_head_channels, kernel_size=1, padding=0, bias=True)
                for c in channels[:-1]
            ])
            self.seg_conv_layers = nn.Sequential(
                nn.Conv2d(seg_head_channels * len(enc_blk_nums), seg_head_channels * 2, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(seg_head_channels * 2, seg_head_channels, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(seg_head_channels, seg_output_channels, kernel_size=1, padding=0, bias=True),
            )
            if freeze_backbone:
                self.freeze_pretrained_layers()

    def freeze_pretrained_layers(self):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.encoder_adapt_convs.parameters():
            param.requires_grad = True
        for param in self.seg_conv_layers.parameters():
            param.requires_grad = True

    def forward(self, x1, x2, x3):
        _, _, height, width = x2.shape
        x1 = self.check_image_size(x1)
        x2 = self.check_image_size(x2)
        x3 = self.check_image_size(x3)

        f1 = self.intro_0(x1)
        f2 = self.intro_1(x2)
        f3 = self.intro_2(x3)
        x = self.fusion(f1, f2, f3)

        encoder_features = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encoder_features.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, skip in zip(self.decoders, self.ups, reversed(encoder_features)):
            x = decoder(up(x) + skip)

        x = torch.sigmoid(self.outconv(self.ending(x) + f2))
        x = x[:, :, :height, :width]
        if not self.use_mask_head:
            return x
        seg_output = self.forward_segmentation_head(encoder_features, f2.shape[-2:])
        return x, seg_output[:, :, :height, :width]

    def forward_segmentation_head(self, encoder_features, output_size):
        seg_features = []
        for enc_feat, adapt_conv in zip(encoder_features, self.encoder_adapt_convs):
            feat = adapt_conv(enc_feat)
            feat = F.interpolate(feat, size=output_size, mode="bilinear", align_corners=False)
            seg_features.append(feat)
        return torch.sigmoid(self.seg_conv_layers(torch.cat(seg_features, dim=1)))

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h))

    def get_visual_embedding(self, x1, x2, x3):
        _, _, height, width = x2.shape
        x1 = self.check_image_size(x1)
        x2 = self.check_image_size(x2)
        x3 = self.check_image_size(x3)

        f1 = self.intro_0(x1)
        f2 = self.intro_1(x2)
        f3 = self.intro_2(x3)
        x = self.fusion(f1, f2, f3)
        encoder_features = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encoder_features.append(x)
            x = down(x)
        x = self.middle_blks(x)
        visual_embedding = torch.flatten(x, -2, -1).permute(0, 2, 1)
        if not self.use_mask_head:
            return visual_embedding

        for decoder, up, skip in zip(self.decoders, self.ups, reversed(encoder_features)):
            x = decoder(up(x) + skip)

        gain_map = torch.sigmoid(self.outconv(self.ending(x) + f2))
        gain_map = gain_map[:, :, :height, :width]
        mask = self.forward_segmentation_head(encoder_features, f2.shape[-2:])
        mask = mask[:, :, :height, :width]
        return visual_embedding, gain_map * 2 - 1, mask
