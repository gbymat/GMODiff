import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'diffusion'))

import torch
import torch.nn as nn
from diffusers import DDPMScheduler
from models.autoencoder_kl import AutoencoderKL
from models.unet_2d_condition import UNet2DConditionModel
from peft import LoraConfig
from model import make_1step_sched, my_vae_encoder_fwd, my_vae_decoder_fwd


def initialize_vae(args):
    vae = AutoencoderKL.from_pretrained(args.pretrained_model, subfolder="vae")
    vae.requires_grad_(False)
    vae.train()

    vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
    vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
    vae.decoder.skip_conv_1 = DRB(512, 512).cuda()
    vae.decoder.skip_conv_2 = DRB(256, 512).cuda()
    vae.decoder.skip_conv_3 = DRB(128, 512).cuda()
    vae.decoder.skip_conv_4 = DRB(128, 256).cuda()
    vae.decoder.ignore_skip = False
    torch.nn.init.constant_(vae.decoder.skip_conv_1.conv1.weight, 1e-5)
    torch.nn.init.constant_(vae.decoder.skip_conv_2.conv1.weight, 1e-5)
    torch.nn.init.constant_(vae.decoder.skip_conv_3.conv1.weight, 1e-5)
    torch.nn.init.constant_(vae.decoder.skip_conv_4.conv1.weight, 1e-5)


    l_target_modules_encoder = []
    l_grep = ["conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
                "skip_conv_1", "skip_conv_2", "skip_conv_3", "skip_conv_4",
                "to_k", "to_q", "to_v", "to_out.0",
            ]
    for n, p in vae.named_parameters():
        if "bias" in n or "norm" in n:
            continue
        for pattern in l_grep:
            if pattern in n and ("encoder" in n):
                l_target_modules_encoder.append(n.replace(".weight", ""))
            elif ('quant_conv' in n) and ('post_quant_conv' not in n):
                l_target_modules_encoder.append(n.replace(".weight", ""))

    lora_conf_encoder = LoraConfig(r=args.lora_rank, init_lora_weights="gaussian",
                                   target_modules=l_target_modules_encoder)
    vae.add_adapter(lora_conf_encoder, adapter_name="default_encoder")

    l_target_modules_decoder = []
    for n, p in vae.named_parameters():
        if "bias" in n or "norm" in n:  # 忽略偏置和归一化层（通常不做 LoRA）
            continue
        for pattern in l_grep:  # 复用相同的关键词（卷积层、注意力层等）
            if pattern in n and ("decoder" in n):  # 筛选 decoder 中的层
                l_target_modules_decoder.append(n.replace(".weight", ""))


    lora_conf_decoder = LoraConfig(r=args.lora_rank, init_lora_weights="gaussian",
                                   target_modules=l_target_modules_decoder)
    vae.add_adapter(lora_conf_decoder, adapter_name="default_decoder")  # 使用不同的 adapter 名称

    return vae, l_target_modules_encoder, l_target_modules_decoder

def initialize_vae1(args):
    vae = AutoencoderKL.from_pretrained(args.pretrained_model, subfolder="vae")
    vae.requires_grad_(False)
    vae.train()

    vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
    vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
    vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
    vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
    vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
    vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
    vae.decoder.ignore_skip = False

    torch.nn.init.constant_(vae.decoder.skip_conv_1.weight, 1e-5)
    torch.nn.init.constant_(vae.decoder.skip_conv_2.weight, 1e-5)
    torch.nn.init.constant_(vae.decoder.skip_conv_3.weight, 1e-5)
    torch.nn.init.constant_(vae.decoder.skip_conv_4.weight, 1e-5)


    l_target_modules_encoder = []
    l_grep = ["conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
                "skip_conv_1", "skip_conv_2", "skip_conv_3", "skip_conv_4",
                "to_k", "to_q", "to_v", "to_out.0",
            ]
    for n, p in vae.named_parameters():
        if "bias" in n or "norm" in n:
            continue
        for pattern in l_grep:
            if pattern in n and ("encoder" in n):
                l_target_modules_encoder.append(n.replace(".weight", ""))
            elif ('quant_conv' in n) and ('post_quant_conv' not in n):
                l_target_modules_encoder.append(n.replace(".weight", ""))

    lora_conf_encoder = LoraConfig(r=args.lora_rank, init_lora_weights="gaussian",
                                   target_modules=l_target_modules_encoder)
    vae.add_adapter(lora_conf_encoder, adapter_name="default_encoder")


    return vae, l_target_modules_encoder


def initialize_unet(args):
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model, subfolder="unet")
    unet.requires_grad_(False)
    unet.train()

    original_conv_in = unet.conv_in
    out_channels = original_conv_in.out_channels  # 320（固定）
    kernel_size = original_conv_in.kernel_size
    padding = original_conv_in.padding
    stride = original_conv_in.stride

    new_conv_in = torch.nn.Conv2d(
        in_channels=8,  # 修改为原通道数的2倍
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding
    )

    with torch.no_grad():
        new_conv_in.weight[:, :4, :, :] = original_conv_in.weight.clone()
        new_conv_in.weight[:, 4:, :, :] = torch.randn_like(
            new_conv_in.weight[:, 4:, :, :]) * original_conv_in.weight.std()
        if original_conv_in.bias is not None:
            new_conv_in.bias = torch.nn.Parameter(original_conv_in.bias.clone())  # 这里是修改点

    unet.conv_in = new_conv_in

    l_target_modules_encoder, l_target_modules_decoder, l_modules_others = [], [], []
    l_grep = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_in", "conv_shortcut", "conv_out",
              "proj_out", "proj_in", "ff.net.2", "ff.net.0.proj", "downsamplers.0.conv", "upsamplers.0.conv", ]
    for n, p in unet.named_parameters():
        if "bias" in n or "norm" in n:
            continue
        for pattern in l_grep:
            if pattern in n and ("down_blocks" in n or "conv_in" in n):
                l_target_modules_encoder.append(n.replace(".weight", ""))
                break
            elif pattern in n and ("up_blocks" in n or "conv_out" in n):
                l_target_modules_decoder.append(n.replace(".weight", ""))
                break
            elif pattern in n:
                l_modules_others.append(n.replace(".weight", ""))
                break

    lora_conf_encoder = LoraConfig(r=args.lora_rank, init_lora_weights="gaussian",
                                   target_modules=l_target_modules_encoder)
    lora_conf_decoder = LoraConfig(r=args.lora_rank, init_lora_weights="gaussian",
                                   target_modules=l_target_modules_decoder)
    lora_conf_others = LoraConfig(r=args.lora_rank, init_lora_weights="gaussian", target_modules=l_modules_others)
    unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
    unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
    unet.add_adapter(lora_conf_others, adapter_name="default_others")

    return unet, l_target_modules_encoder, l_target_modules_decoder, l_modules_others

class DRB(nn.Module):
    """多曝光特征融合模块，强调参考帧权重"""

    def __init__(self, in_channels=32, out_channels=32):
        super(DRB, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)  # 1x1卷积无需padding

        self.mlp = nn.Sequential(
            nn.Linear(in_features=1024, out_features=out_channels),  # 假设p的最后一维是in_channels
            nn.ReLU(),  # 增加非线性
            nn.Linear(in_features=out_channels,out_features=out_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, p):

        skip_in = self.conv1(x1)  # [b, out_channels, h, w]

        p_agg = p.mean(dim=1)  # 全局平均池化n维度，也可用max(dim=1)
        p_weight = self.mlp(p_agg)  # [b, out_channels]
        p_weight = p_weight.unsqueeze(2).unsqueeze(3)  # [b, out_channels, 1, 1]
        p_weight = self.sigmoid(p_weight)  # 归一化到[0,1]

        x2 = (skip_in * p_weight)  # 逐通道加权，强调参考帧权重

        return x2


class MaskVisualFusionModule(nn.Module):
    def __init__(self, visual_embedding_dim=1024, mlp_hidden_dim=1024, mlp_output_dim=1024):
        """
        融合mask和visual_embedding的模块
        Args:
            visual_embedding_dim: visual_embedding的最后一维维度（默认512）
            mlp_hidden_dim: MLP隐藏层维度（默认2048）
            mlp_output_dim: MLP最终输出维度（默认1024）
        """
        super(MaskVisualFusionModule, self).__init__()

        self.mask_encoder = nn.Sequential(
            nn.PixelUnshuffle(downscale_factor=8),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        self.flatten_permute = lambda x: torch.nn.Flatten(-2, -1)(x).permute(0, 2, 1)

        self.mlp = nn.Sequential(
            nn.Linear(1024 + visual_embedding_dim, mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_dim, mlp_output_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, visual_embedding,mask):
        """
        前向传播
        Args:
            mask: 输入mask，shape=[batch_size, 1, 256, 256] (示例: [3,1,256,256])
            visual_embedding: 视觉嵌入，shape=[batch_size, visual_embedding_dim] 或 [batch_size, L, visual_embedding_dim]
                              示例: [3,512] 或 [3,64,512]
        Returns:
            output: MLP输出，shape=[batch_size, L, mlp_output_dim] (L=64) 或 [batch_size, mlp_output_dim]
        """
        mask_feat = self.mask_encoder(mask)  # [3,1024,8,8]

        mask_vec = torch.nn.Flatten(-2, -1)(mask_feat)
        mask_vec = mask_vec.permute(0, 2, 1)

        fused_feat = torch.cat([mask_vec, visual_embedding], dim=-1)

        output = self.mlp(fused_feat)

        return output

class GMODiff_train(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model, subfolder="scheduler")
        self.noise_scheduler.set_timesteps(1, device="cuda")
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.cuda()
        self.args = args

        self.vae, self.lora_vae_modules_encoder, self.lora_vae_modules_decoder = initialize_vae(self.args)
        self.vae.decoder.gamma = 1
        self.vae_c, self.lora_vae_c_modules_encoder = initialize_vae1(self.args)
        self.unet, self.lora_unet_modules_encoder, self.lora_unet_modules_decoder, self.lora_unet_others = initialize_unet(
            self.args)
        self.lora_rank_unet = self.args.lora_rank
        self.lora_rank_vae = self.args.lora_rank
        self.lora_rank_vae_c = self.args.lora_rank
        self.proj = MaskVisualFusionModule().to("cuda")

        if args.gmodiff_path:
            gmodiff = torch.load(args.gmodiff_path, map_location='cpu')
            self.load_ckpt(gmodiff)
        else:
            print("No GMODiff checkpoint provided. Training will start from the initialized SD/DaReg adapters.")

        self.unet.to("cuda")
        self.vae.to("cuda")
        self.vae_c.to("cuda")
        self.timesteps = torch.tensor([499], device="cuda").long()


        self.proj.requires_grad_(True)
        self.vae.set_adapter(['default_encoder', 'default_decoder'])
        self.vae_c.set_adapter(['default_encoder'])
        self.unet.set_adapter(['default_encoder', 'default_decoder', 'default_others'])
        self.set_train()

    def set_train(self):
        self.unet.train()
        self.vae.train()
        self.vae_c.train()
        self.proj.requires_grad_(True)
        for n, _p in self.unet.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.unet.conv_in.requires_grad_(True)

        for n, _p in self.vae_c.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        for n, _p in self.vae.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.vae.decoder.skip_conv_1.requires_grad_(True)
        self.vae.decoder.skip_conv_2.requires_grad_(True)
        self.vae.decoder.skip_conv_3.requires_grad_(True)
        self.vae.decoder.skip_conv_4.requires_grad_(True)

    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        self.vae_c.eval()
        for n, _p in self.unet.named_parameters():
            if "lora" in n:
                _p.requires_grad = False
        self.unet.conv_in.requires_grad_(False)
        for n, _p in self.vae.named_parameters():
            if "lora" in n:
                _p.requires_grad = False

        for n, _p in self.vae_c.named_parameters():
            if "lora" in n:
                _p.requires_grad = False

    @torch.no_grad()
    def eval(self, lq, visual_embedding):
        lq_latent = self.vae.encode(lq).latent_dist.sample() * self.vae.config.scaling_factor
        model_pred = self.unet(lq_latent, self.timesteps,
                               encoder_hidden_states=self.proj(visual_embedding.to(torch.float32))).sample
        x_denoised = self.noise_scheduler.step(model_pred, self.timesteps, lq_latent, return_dict=True).prev_sample
        output_image = (
            self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)

        return output_image

    def forward(self, c_t, visual_embedding,condition):

        condition = self.vae_c.encode(condition).latent_dist.sample() * self.vae_c.config.scaling_factor

        encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor  # b, 4, 16, 16


        visual_embeds = self.proj(visual_embedding[0],visual_embedding[1])

        model_pred = self.unet(torch.cat((encoded_control, condition), dim=1), self.timesteps,
                               encoder_hidden_states=visual_embeds.to(torch.float32), ).sample

        x_denoised = self.noise_scheduler.step(model_pred, self.timesteps, encoded_control,
                                               return_dict=True).prev_sample
        self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
        self.vae.decoder.prior = visual_embeds
        output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)

        return output_image, x_denoised

    def save_model(self, outf):
        sd = {}
        sd["vae_lora_encoder_modules"] = self.lora_vae_modules_encoder
        sd["vae_lora_decoder_modules"] = self.lora_vae_modules_decoder
        sd["vae_c_lora_encoder_modules"] = self.lora_vae_c_modules_encoder
        sd["unet_lora_encoder_modules"], sd["unet_lora_decoder_modules"], sd["unet_lora_others_modules"] = \
            self.lora_unet_modules_encoder, self.lora_unet_modules_decoder, self.lora_unet_others
        sd["rank_unet"] = self.lora_rank_unet
        sd["rank_vae"] = self.lora_rank_vae
        sd["rank_vae_c"] = self.lora_rank_vae_c
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k}
        sd["state_dict_vae"] = {k: v for k, v in self.vae.state_dict().items() if "lora" in k or "skip" in k}
        sd["state_dict_vae_c"] = {k: v for k, v in self.vae_c.state_dict().items() if "lora" in k or "skip" in k}
        sd["proj"] = self.proj.state_dict()
        torch.save(sd, outf)

    def load_ckpt(self, model):

        unet_state_dict = model["state_dict_unet"]
        for n, p in self.unet.named_parameters():
            if "lora" in n:
                if p.shape == unet_state_dict[n].shape:
                    p.data.copy_(unet_state_dict[n])
                else:
                    print(f"跳过不匹配的lora参数：{n}，模型形状{p.shape}，权重形状{unet_state_dict[n].shape}")
            elif "conv_in" in n:
                if p.shape != unet_state_dict[n].shape:
                    print(f"处理不匹配的conv_in参数：{n}，模型形状{p.shape}，权重形状{unet_state_dict[n].shape}")
                    if "weight" in n:
                        pretrained_weight = unet_state_dict[n]
                        expanded_weight = pretrained_weight.repeat(1, 2, 1, 1)  # 重复输入通道
                        p.data.copy_(expanded_weight)
                        print(f"已扩展 {n} 通道：{pretrained_weight.shape} → {expanded_weight.shape}")
                    else:
                        print(f"跳过不匹配的conv_in参数：{n}，使用模型默认初始化")
                else:
                    p.data.copy_(unet_state_dict[n])

        self.unet.set_adapter(["default_encoder", "default_decoder", "default_others"])

        for n, p in self.vae.named_parameters():
            if n in model["state_dict_vae"]:
                if p.shape == model["state_dict_vae"][n].shape:
                    p.data.copy_(model["state_dict_vae"][n])
                    print(f"成功载入vae lora参数：{n}")
                else:
                    print(f"跳过不匹配的vae lora参数：{n}")
        self.vae.set_adapter(['default_encoder','default_decoder'])

        for n, p in self.vae_c.named_parameters():
            if n in model["state_dict_vae_c"]:
                if p.shape == model["state_dict_vae_c"][n].shape:
                    p.data.copy_(model["state_dict_vae_c"][n])
                    print(f"成功载入vae_c lora参数：{n}")
                else:
                    print(f"跳过不匹配的vae_c lora参数：{n}")
        self.vae_c.set_adapter(['default_encoder'])

        self.set_train()
        print(f"成功载入权重")


class GMODiff_test(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model, subfolder="scheduler")
        self.noise_scheduler.set_timesteps(1, device="cuda")
        self.vae = AutoencoderKL.from_pretrained(self.args.pretrained_model, subfolder="vae")
        self.vae_c = AutoencoderKL.from_pretrained(self.args.pretrained_model, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(self.args.pretrained_model, subfolder="unet")

        self.proj = MaskVisualFusionModule().to("cuda")

        original_conv_in = self.unet.conv_in
        out_channels = original_conv_in.out_channels  # 320（固定）
        kernel_size = original_conv_in.kernel_size
        padding = original_conv_in.padding
        stride = original_conv_in.stride

        new_conv_in = torch.nn.Conv2d(
            in_channels=8,  # 修改为原通道数的2倍
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

        with torch.no_grad():
            new_conv_in.weight[:, :4, :, :] = original_conv_in.weight.clone()
            new_conv_in.weight[:, 4:, :, :] = torch.randn_like(
                new_conv_in.weight[:, 4:, :, :]) * original_conv_in.weight.std()
            if original_conv_in.bias is not None:
                new_conv_in.bias = torch.nn.Parameter(original_conv_in.bias.clone())  # 这里是修改点

        self.unet.conv_in = new_conv_in

        self.vae.decoder.skip_conv_1 = DRB(512, 512).cuda()
        self.vae.decoder.skip_conv_2 = DRB(256, 512).cuda()
        self.vae.decoder.skip_conv_3 = DRB(128, 512).cuda()
        self.vae.decoder.skip_conv_4 = DRB(128, 256).cuda()
        self.vae.decoder.ignore_skip = False
        self.vae.encoder.forward = my_vae_encoder_fwd.__get__(self.vae.encoder, self.vae.encoder.__class__)
        self.vae.decoder.forward = my_vae_decoder_fwd.__get__(self.vae.decoder, self.vae.decoder.__class__)
        self.vae.decoder.gamma = 1


        self.weight_dtype = torch.float32
        if args.mixed_precision == "fp16":
            self.weight_dtype = torch.float16

        if args.gmodiff_path is None:
            print('Do not provide any GMODiff path!')
        else:
            gmodiff = torch.load(args.gmodiff_path, map_location='cuda')
            self.load_ckpt(gmodiff)

        if self.args.merge_and_unload_lora:
            print(f'===> MERGE LORA <===')
            self.vae = self.vae.merge_and_unload()
            self.unet = self.unet.merge_and_unload()

        self.unet.to("cuda", dtype=self.weight_dtype)
        self.vae.to("cuda", dtype=self.weight_dtype)
        self.vae_c.to("cuda", dtype=self.weight_dtype)
        self.timesteps = torch.tensor([499], device="cuda").long()
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.cuda()

    def load_ckpt(self, model):
        self.proj.load_state_dict(model["proj"])
        lora_conf_encoder = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian",
                                       target_modules=model["unet_lora_encoder_modules"])
        lora_conf_decoder = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian",
                                       target_modules=model["unet_lora_decoder_modules"])
        lora_conf_others = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian",
                                      target_modules=model["unet_lora_others_modules"])
        self.unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
        self.unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
        self.unet.add_adapter(lora_conf_others, adapter_name="default_others")
        for n, p in self.unet.named_parameters():
            if "lora" in n or "conv_in" in n:
                p.data.copy_(model["state_dict_unet"][n])
        self.unet.set_adapter(["default_encoder", "default_decoder", "default_others"])


        vae_lora_conf_decoder = LoraConfig(r=model["rank_vae"], init_lora_weights="gaussian",
                                           target_modules=model["vae_lora_decoder_modules"])
        self.vae.add_adapter(vae_lora_conf_decoder, adapter_name="default_decoder")

        vae_lora_conf_encoder = LoraConfig(r=model["rank_vae"], init_lora_weights="gaussian",
                                           target_modules=model["vae_lora_encoder_modules"])
        self.vae.add_adapter(vae_lora_conf_encoder, adapter_name="default_encoder")
        for n, p in self.vae.named_parameters():
            if "lora" in n or "skip" in n:
                p.data.copy_(model["state_dict_vae"][n])
        self.vae.set_adapter(['default_encoder', 'default_decoder'])

        vae_c_lora_conf_encoder = LoraConfig(r=model["rank_vae_c"], init_lora_weights="gaussian",
                                           target_modules=model["vae_c_lora_encoder_modules"])
        self.vae_c.add_adapter(vae_c_lora_conf_encoder, adapter_name="default_encoder")
        for n, p in self.vae_c.named_parameters():
            if "lora" in n :
                p.data.copy_(model["state_dict_vae_c"][n])
        self.vae_c.set_adapter(['default_encoder'])

    @torch.no_grad()
    def forward(self, lq, visual_embedding,condition):
        condition = self.vae_c.encode(condition.to(self.weight_dtype)).latent_dist.sample() * self.vae_c.config.scaling_factor
        lq_latent = self.vae.encode(lq.to(self.weight_dtype)).latent_dist.sample() * self.vae.config.scaling_factor
        visual_embedding = self.proj(visual_embedding[0],visual_embedding[1]).to(torch.float16)

        _, _, h, w = lq_latent.size()
        tile_size, tile_overlap = (self.args.latent_tiled_size, self.args.latent_tiled_overlap)
        if h * w <= tile_size * tile_size:
            model_pred = self.unet(torch.cat((lq_latent, condition), dim=1), self.timesteps, encoder_hidden_states=visual_embedding).sample
        else:
            print(f"[Tiled Latent]: the input size is {lq.shape[-2]}x{lq.shape[-1]}, need to tiled")
            tile_size = min(tile_size, min(h, w))
            tile_weights = self._gaussian_weights(tile_size, tile_size, 1)

            grid_rows = 0
            cur_x = 0
            while cur_x < lq_latent.size(-1):
                cur_x = max(grid_rows * tile_size - tile_overlap * grid_rows, 0) + tile_size
                grid_rows += 1

            grid_cols = 0
            cur_y = 0
            while cur_y < lq_latent.size(-2):
                cur_y = max(grid_cols * tile_size - tile_overlap * grid_cols, 0) + tile_size
                grid_cols += 1

            input_list = []
            noise_preds = []
            for row in range(grid_rows):
                for col in range(grid_cols):
                    if col < grid_cols - 1 or row < grid_rows - 1:
                        ofs_x = max(row * tile_size - tile_overlap * row, 0)
                        ofs_y = max(col * tile_size - tile_overlap * col, 0)
                    if row == grid_rows - 1:
                        ofs_x = w - tile_size
                    if col == grid_cols - 1:
                        ofs_y = h - tile_size

                    input_start_x = ofs_x
                    input_end_x = ofs_x + tile_size
                    input_start_y = ofs_y
                    input_end_y = ofs_y + tile_size

                    input_tile1 = lq_latent[:, :, input_start_y:input_end_y, input_start_x:input_end_x]
                    input_tile2 = condition[:, :, input_start_y:input_end_y, input_start_x:input_end_x]
                    input_list.append(torch.cat((input_tile1, input_tile2), dim=1))

                    if len(input_list) == 1 or col == grid_cols - 1:
                        input_list_t = torch.cat(input_list, dim=0)
                        model_out = self.unet(input_list_t, self.timesteps,
                                              encoder_hidden_states=visual_embedding).sample
                        input_list = []
                    noise_preds.append(model_out)

            noise_pred = torch.zeros(lq_latent.shape, device=lq_latent.device)
            contributors = torch.zeros(lq_latent.shape, device=lq_latent.device)
            for row in range(grid_rows):
                for col in range(grid_cols):
                    if col < grid_cols - 1 or row < grid_rows - 1:
                        ofs_x = max(row * tile_size - tile_overlap * row, 0)
                        ofs_y = max(col * tile_size - tile_overlap * col, 0)
                    if row == grid_rows - 1:
                        ofs_x = w - tile_size
                    if col == grid_cols - 1:
                        ofs_y = h - tile_size

                    input_start_x = ofs_x
                    input_end_x = ofs_x + tile_size
                    input_start_y = ofs_y
                    input_end_y = ofs_y + tile_size

                    noise_pred[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += noise_preds[
                                                                                                  row * grid_cols + col] * tile_weights
                    contributors[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += tile_weights
            noise_pred /= contributors
            model_pred = noise_pred
        self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
        self.vae.decoder.prior = visual_embedding
        x_denoised = self.noise_scheduler.step(model_pred, self.timesteps, lq_latent, return_dict=True).prev_sample
        output_image = (
            self.vae.decode(x_denoised.to(self.weight_dtype) / self.vae.config.scaling_factor).sample).clamp(-1, 1)
        return output_image


    def _gaussian_weights(self, tile_width, tile_height, nbatches):
        """Generates a gaussian mask of weights for tile contributions"""
        from numpy import pi, exp, sqrt
        import numpy as np

        latent_width = tile_width
        latent_height = tile_height

        var = 0.01
        midpoint = (latent_width - 1) / 2  # -1 because index goes from 0 to latent_width - 1
        x_probs = [
            exp(-(x - midpoint) * (x - midpoint) / (latent_width * latent_width) / (2 * var)) / sqrt(2 * pi * var) for x
            in range(latent_width)]
        midpoint = latent_height / 2
        y_probs = [
            exp(-(y - midpoint) * (y - midpoint) / (latent_height * latent_height) / (2 * var)) / sqrt(2 * pi * var) for
            y in range(latent_height)]

        weights = np.outer(y_probs, x_probs)
        return torch.tile(torch.tensor(weights, device=self.device), (nbatches, self.unet.config.in_channels, 1, 1))

