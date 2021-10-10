# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from . import vit_seg_configs as configs
from .vit_seg_modeling_resnet_skip import ResNetV2,ResNetV2_transunet


logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3) #batch size x num_attention_heads x num_patches x attention_head_size-->every attentioin head has part of info from the hidden size for each patch 

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # output last 2 dimension num_patches x num_patches
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3, bn_pretrain=False,pretrain=False, use_tranunet_enc_dec=False, dropout_rate=0.2,eval_uncer=False): #original in_channels = 3
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.pretrain=pretrain
        self.config = config
        img_size = _pair(img_size)
        self.num_tokens = 0
        
        if self.pretrain:
            self.num_tokens = 1
            # self.rot_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))###>
            # self.rot_axis_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))###>
            self.contrastive_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))###>
            self.ext_tok_pos_embeddings = nn.Parameter(torch.zeros(1, self.num_tokens, config.hidden_size))

        if config.patches.get("grid") is not None:   # ResNet
            grid_size = config.patches["grid"]
            #seems most of the time the patch_size=1
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])# (img_size / 16) would be the size of the image after the resnet 
            # print("patch_size")
            # print(patch_size)
            # print("grid_size")
            # print(grid_size)
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])  #should be the grid size[0]x grid size[1]
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            if(use_tranunet_enc_dec):
                self.hybrid_model = ResNetV2_transunet(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor,\
                drp_out_rate=dropout_rate, eval_uncer_f=False)
                in_channels = self.hybrid_model.width * 16
            else:
                self.hybrid_model = ResNetV2(bn_pretrain)
                in_channels = self.hybrid_model.out_width# * 16
            
            
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)#output size =img_size[0] after resnet / patch_size[0] = n_patches^(1/2)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])


    def forward(self, x):
        # print(x.shape)
        if self.hybrid:
            x, features = self.hybrid_model(x)
            # print("x, features = self.hybrid_model(x)")
            # print(x.shape)
        else:
            features = None
        # print("x before batch embedding")
        # print(x.size())
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        # print("x = self.patch_embeddings(x)")
        # print(x.shape)
        retain_size_2 = x.size()[2]
        retain_size_3 = x.size()[3]
        x = x.flatten(2)
        # print("x = x.flatten(2)")
        # print(x.shape)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)
        # print("x = x.transpose(-1, -2)")
        # print(x.shape)
        
        # print(" self.position_embeddings")
        # print( self.position_embeddings.shape)
        
        if self.pretrain:
            B=x.shape[0]
            # rot_token = self.rot_token.expand(B, -1, -1) 
            # rot_axis_token = self.rot_axis_token.expand(B, -1, -1) 
            contrastive_token = self.contrastive_token.expand(B, -1, -1) 
            x = torch.cat((contrastive_token, x), dim=1) ###>
            all_pos_embeddings = torch.cat((self.ext_tok_pos_embeddings, self.position_embeddings), dim=1) ###>
            embeddings = x + all_pos_embeddings
        else:
            embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features,retain_size_2,retain_size_3


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[(ROOT+ '/'+ATTENTION_Q+ "/kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[(ROOT+ '/'+ ATTENTION_K+ "/kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[(ROOT+ '/'+ ATTENTION_V+ "/kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[(ROOT+ '/'+ ATTENTION_OUT+ "/kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[(ROOT+ '/'+ ATTENTION_Q+ "/bias")]).view(-1)
            key_bias = np2th(weights[(ROOT+ '/'+ ATTENTION_K+ "/bias")]).view(-1)
            value_bias = np2th(weights[(ROOT+ '/'+ ATTENTION_V+ "/bias")]).view(-1)
            out_bias = np2th(weights[(ROOT+ '/'+ ATTENTION_OUT+ "/bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[(ROOT+ '/'+ FC_0+ "/kernel")]).t()
            mlp_weight_1 = np2th(weights[(ROOT+ '/'+ FC_1+ "/kernel")]).t()
            mlp_bias_0 = np2th(weights[(ROOT+ '/'+ FC_0+ "/bias")]).t()
            mlp_bias_1 = np2th(weights[(ROOT+ '/'+ FC_1+ "/bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[(ROOT+ '/'+ ATTENTION_NORM+ "/scale")]))
            self.attention_norm.bias.copy_(np2th(weights[(ROOT+ '/'+ ATTENTION_NORM+ "/bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[(ROOT+ '/'+ MLP_NORM+ "/scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[(ROOT+ '/'+ MLP_NORM+ "/bias")]))


class Encoder(nn.Module):
    def __init__(self, config, vis, low_dim=512, pretrain=False, contrastive=False):
        super(Encoder, self).__init__()
        self.vis = vis
        self.pretrain = pretrain
        self.contrastive = contrastive
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))
            
        
        if self.contrastive:
            # self.rot_head = nn.Linear(config.hidden_size, 4) ###>
            # self.rot_axis_head = nn.Linear(config.hidden_size, 1) ###>
            self.contrastive_head = nn.Linear(config.hidden_size, low_dim) ###>
            

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        if self.contrastive:
            # x_rot = self.rot_head(encoded[:, 0])
            # x_rot_axis = self.rot_axis_head(encoded[:, 1])
            x_contrastive = self.contrastive_head(encoded[:, 1])###>
            
            return x_contrastive, encoded[:, 1:], attn_weights
            
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis, low_dim=512, rm_transformer=False, bn_pretrain=False,pretrain=False, contrastive=False, \
    use_tranunet_enc_dec=False, dropout_rate=0.2,eval_uncer=False):
        super(Transformer, self).__init__()
        self.pretrain = pretrain
        self.contrastive=contrastive
        self.rm_transformer = rm_transformer
            
        self.embeddings = Embeddings(config, img_size=img_size, bn_pretrain=bn_pretrain,pretrain=contrastive, use_tranunet_enc_dec=use_tranunet_enc_dec,\
        dropout_rate=dropout_rate, eval_uncer=eval_uncer)
        if rm_transformer==False:
            self.encoder = Encoder(config, vis, low_dim=low_dim,pretrain=pretrain, contrastive=contrastive)

    def forward(self, input_ids):
        embedding_output, features,bfr_flat_size_2,bfr_flat_size_3 = self.embeddings(input_ids)
        # hybrid_output = embedding_output
        if self.contrastive:
            x_contrastive, encoded, attn_weights = self.encoder(embedding_output)
            return x_contrastive, encoded, attn_weights, features,bfr_flat_size_2,bfr_flat_size_3
        
        if self.rm_transformer==False:
            encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
            # encoded =  torch.cat([encoded , hybrid_output], dim=2)
            return encoded, attn_weights, features,bfr_flat_size_2,bfr_flat_size_3
        else:
            return embedding_output, None, features,bfr_flat_size_2,bfr_flat_size_3


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)



class UpBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, drop_out=True, pretrain=False):
        super(UpBlock, self).__init__()
        self.drop_out = drop_out
        self.in_filters = in_filters
        self.out_filters = out_filters
        self.pretrain = pretrain

        self.dropout1 = nn.Dropout2d(p=dropout_rate)

        self.dropout2 = nn.Dropout2d(p=dropout_rate)

        self.conv1 = nn.Conv2d(in_filters//4 + 2*out_filters, out_filters, (3,3), padding=1)
        self.act1 = nn.LeakyReLU()
        if pretrain:
            self.bn1 = nn.BatchNorm2d(out_filters)
        else:
            self.bn1_ = nn.BatchNorm2d(out_filters)

        self.conv2 = nn.Conv2d(out_filters, out_filters, (3,3),dilation=2, padding=2)
        self.act2 = nn.LeakyReLU()
        if pretrain:
            self.bn2 = nn.BatchNorm2d(out_filters)
        else:
            self.bn2_ = nn.BatchNorm2d(out_filters)
        

        self.conv3 = nn.Conv2d(out_filters, out_filters, (2,2), dilation=2,padding=1)
        self.act3 = nn.LeakyReLU()
        if pretrain:
            self.bn3 = nn.BatchNorm2d(out_filters)
        else:
            self.bn3_ = nn.BatchNorm2d(out_filters)
        


        self.conv4 = nn.Conv2d(out_filters*3,out_filters,kernel_size=(1,1))
        self.act4 = nn.LeakyReLU()
        if pretrain:
            self.bn4 = nn.BatchNorm2d(out_filters)
        else:
            self.bn4_ = nn.BatchNorm2d(out_filters)
        

        self.dropout3 = nn.Dropout2d(p=dropout_rate)

    def forward(self, x, skip):
        # print("before shiffle x.shape")
        # print(x.shape)
        upA = nn.PixelShuffle(2)(x) # kind of reshape
        # print("upA.shape")
        # print(upA.shape)
        if self.drop_out:
            upA = self.dropout1(upA)

        # print("upA.shape")
        # print(upA.shape)
        # print("skip.shape")
        # print(skip.shape)
        upB = torch.cat((upA,skip),dim=1)
        # print("x skip concat")
        # print(upB.shape)
        if self.drop_out:
            upB = self.dropout2(upB)

        upE = self.conv1(upB)
        upE = self.act1(upE)
        if self.pretrain:
            upE1 = self.bn1(upE)
        else:
            upE1 = self.bn1_(upE)

        upE = self.conv2(upE1)
        upE = self.act2(upE)
        if self.pretrain:
            upE2 = self.bn2(upE)
        else:
            upE2 = self.bn2_(upE)
        

        upE = self.conv3(upE2)
        upE = self.act3(upE)
        if self.pretrain:
            upE3 = self.bn3(upE)
        else:
            upE3 = self.bn3_(upE)
        

        concat = torch.cat((upE1,upE2,upE3),dim=1)
        upE = self.conv4(concat)
        upE = self.act4(upE)
        if self.pretrain:
            upE = self.bn4(upE)
        else:
            upE = self.bn4_(upE)
        
        if self.drop_out:
            upE = self.dropout3(upE)

        return upE
        

class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class DecoderCup(nn.Module):
    def __init__(self, config, pretrain):
        super().__init__()
        

        self.upBlock1 = UpBlock(config.hidden_size, 4 * 32, 0.2, pretrain=pretrain)
        self.upBlock2 = UpBlock(4 * 32, 4 * 32, 0.2, pretrain=pretrain)
        self.upBlock3 = UpBlock(4 * 32, 2 * 32, 0.2, pretrain=pretrain)
        self.upBlock4 = UpBlock(2 * 32, 32, 0.2, drop_out=False, pretrain=pretrain)
        
        self.last_layer_num_chs = 32

    def forward(self, hidden_states,bfr_flat_size_2,bfr_flat_size_3, features=None):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = bfr_flat_size_2,bfr_flat_size_3
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        # print("size before decoder processing")
        # print(x.shape)
        up4e = self.upBlock1(x,features[0])
        up3e = self.upBlock2(up4e, features[1])
        up2e = self.upBlock3(up3e, features[2])
        up1e = self.upBlock4(up2e, features[3])
        # print("size after decoder processing")
        # print(up1e.shape)        
        return up1e


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
            dropout_rate=0.2, eval_uncer=False,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        
        self.dropout1 = nn.Dropout2d(p=dropout_rate)

        self.dropout2 = nn.Dropout2d(p=dropout_rate)
        
        self.dropout3 = nn.Dropout2d(p=dropout_rate)
        self.eval_uncer = eval_uncer

    def forward(self, x, skip=None):
        x = self.up(x)
        if self.eval_uncer:
            x = self.dropout1(x)
            
        if skip is not None:
            # print("x size")
            # print(x.size())
            # print("skip size")
            # print(skip.size())
            x = torch.cat([x, skip], dim=1)
            if self.eval_uncer:
                x = self.dropout2(x)
            # print("cat size")
            # print(x.size())
        x = self.conv1(x)
        x = self.conv2(x)
        if self.eval_uncer:
            x = self.dropout3(x)
        return x
        
class DecoderCup_TransUnet(nn.Module):
    def __init__(self, config,dropout_rate=0.2, eval_uncer=False):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(
            config.hidden_size,  #*2 was removed as the concatenation after the transformer with the output of the resnet was removed
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels
            for i in range(4-self.config.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0

        else:
            skip_channels=[0,0,0,0]
            
        dropout_flag_list = [eval_uncer,eval_uncer,eval_uncer,False]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch,dropout_rate=dropout_rate, eval_uncer=False) \
            for in_ch, out_ch, sk_ch, drpot_flag in zip(in_channels, out_channels, skip_channels, dropout_flag_list)
        ]
        self.blocks = nn.ModuleList(blocks)
        
    def forward(self, hidden_states,bfr_flat_size_2,bfr_flat_size_3, features=None):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = bfr_flat_size_2,bfr_flat_size_3
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x



class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, low_dim=512, zero_head=False, vis=False, rm_transformer=False, bn_pretrain=False, pretrain=False, \
    contrastive=False, use_tranunet_enc_dec=False, dropout_rate=0.2,eval_uncer=False):
        super(VisionTransformer, self).__init__()
        self.pretrain = pretrain
        self.contrastive = contrastive
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, vis,low_dim=low_dim, rm_transformer=rm_transformer, bn_pretrain=bn_pretrain,\
        pretrain=pretrain, contrastive=contrastive, use_tranunet_enc_dec=use_tranunet_enc_dec, dropout_rate=dropout_rate ,eval_uncer=eval_uncer)
        # if self.pretrain:
        if(use_tranunet_enc_dec):
            self.decoder = DecoderCup_TransUnet(config, dropout_rate=dropout_rate, eval_uncer=eval_uncer)
            in_chs=  config['decoder_channels'][-1]
        else:
            self.decoder = DecoderCup(config, bn_pretrain)
            in_chs= self.decoder.last_layer_num_chs
        # else:
            # self.decoder_finetune = DecoderCup(config)
        if pretrain:
           
            self.recon_head = SegmentationHead(
                in_channels=in_chs,
                out_channels=5, #range,x,y,z,remission
                kernel_size=3,
            )
            
            # create learnable parameters for the MTL task
            if contrastive:
                self.contrastive_w = nn.Parameter(torch.tensor([1.0]))###>
                self.recons_w = nn.Parameter(torch.tensor([1.0]))###>
                self.nce_converge_w = nn.Parameter(torch.tensor([1.0]))###>
        else:
            self.segmentation_head = SegmentationHead(
                in_channels=in_chs,
                out_channels=config['n_classes'],
                kernel_size=3,
                )
        self.config = config

    def forward(self, x):
        if x.size()[1] == 1:
            #if x size was 2x2 then the output of the repeat(1,3,1,1) is 1x3x2x2
            x = x.repeat(1,3,1,1)# 1st size of x is repeated by 1, 2nd size of x is repeated by 3, 3rd and 4th size of x is repeated by 1
            
        if self.contrastive:
            x_contrastive, x, attn_weights, features,bfr_flat_size_2,bfr_flat_size_3 = self.transformer(x)  # (B, n_patch, hidden)
        else:
            x, attn_weights, features,bfr_flat_size_2,bfr_flat_size_3 = self.transformer(x)  # (B, n_patch, hidden)
        # print("x in vision transformer")
        # print(x.size())
        # if self.pretrain:
        x = self.decoder(x,bfr_flat_size_2,bfr_flat_size_3, features)
        # else:
            # x = self.decoder_finetune(x,bfr_flat_size_2,bfr_flat_size_3, features)
        if self.pretrain:
            logits = self.recon_head(x)
            if self.contrastive:
                return x_contrastive, logits, self.contrastive_w, self.recons_w, self.nce_converge_w
            else:
                return logits, None, None, None
            # return logits, self.rot_w, self.contrastive_w, self.recons_w
        else:
            logits = self.segmentation_head(x)
            return logits

    def load_from(self, weights):
        with torch.no_grad():

            res_weight = weights
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1]-1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                if self.classifier == "seg":
                    _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            # Encoder whole
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                # self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(res_weight["conv_root/kernel"], conv=True))
                gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
                gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)

CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'R50-ViT-L_16': configs.get_r50_l16_config(),
    'testing': configs.get_testing(),
}


