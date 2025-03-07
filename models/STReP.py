import torch
import torch.nn as nn
from math import sqrt
import numpy as np

from functools import partial
from einops import rearrange, repeat
from layers.maskgenerator import MaskGenerator


class get_SpatialEmb(nn.Module):
    def __init__(self, d_model, num_nodes, dropout=0.1):
        super().__init__()
        self.spatial_embedding = nn.Embedding(num_nodes, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, spatial_indexs=None):
        if spatial_indexs is None:
            batch, _,  num_nodes, _ = x.shape
            spatial_indexs = torch.LongTensor(
                torch.arange(num_nodes))  # (N,)
        spatial_emb = self.spatial_embedding(
            spatial_indexs.to(x.device)).unsqueeze(0).unsqueeze(1) # (N, d)->(1, 1, N, d)
        return spatial_emb


class get_TemporalEmb(nn.Module):
    def __init__(self, in_dim, d_model, slice_size_per_day, dropout=0.1):
        super().__init__()
        self.time_in_day_embedding = nn.Embedding(slice_size_per_day, d_model)
        self.day_in_week_embedding = nn.Embedding(7, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, t_hour, t_day):
        time_in_day_emb = self.time_in_day_embedding(t_hour) #BT1d
        day_in_week_emb = self.day_in_week_embedding(t_day)         
        return time_in_day_emb, day_in_week_emb


class Projector_1(nn.Module):
    def __init__(self, in_dim, d_model, hasCross=True):
        super().__init__()
        in_units = in_dim*2 if hasCross else in_dim
        self.hasCross = hasCross
        self.linear1 = nn.Linear(in_units, d_model)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_model, d_model)

    def forward(self, x, latestX):
        if self.hasCross:
            t_len = x.shape[1]
            if latestX.shape[1]!= t_len :
                latestX=latestX.repeat([1, t_len, 1, 1])
            data = torch.cat([x, latestX], dim=-1)
        else:
            data = x
        data = self.linear1(data)
        data = self.activation(data)
        data = self.linear2(data)
        return data


class Projector_2(nn.Module):
    def __init__(self, tau, d_model):
        super().__init__()
        assert tau in [1, 2, 3]
        self.tau = tau
        if tau == 2:
            tcn_pad_l, tcn_pad_r = 0, 1
            self.padding = nn.ReplicationPad2d((tcn_pad_l, tcn_pad_r, 0, 0))  # x must be like (B,C,N,T)
        elif tau == 3:
            tcn_pad_l, tcn_pad_r = 1, 1
            self.padding = nn.ReplicationPad2d((tcn_pad_l, tcn_pad_r, 0, 0))  # x must be like (B,C,N,T)
        self.time_conv = nn.Conv2d(d_model, d_model, (1, tau))

    def forward(self, x):
        x = rearrange(x,'b t n c -> b c n t')
        if self.tau > 1:
            x = self.padding(x)
        data = self.time_conv(x)
        data =  rearrange(data,'b c n t -> b t n c')
        return data


class MultiHeadsAttention(nn.Module):
    def __init__(self, scale=None, attention_dropout=0.1, returnA=False):
        super(MultiHeadsAttention, self).__init__()
        self.scale = scale
        self.returnA = returnA
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.returnA:
            return V.contiguous(), A.contiguous()
        else:
            return V.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_values=None, mix=True, dropout=0.1, returnA=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = MultiHeadsAttention(scale=None, attention_dropout=dropout, returnA=returnA)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.returnA = returnA
        self.mix = mix

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        keys = keys.view(B, S, H, -1)
        values = values.view(B, S, H, -1)

        out, A = self.inner_attention(
            queries,
            keys,
            values,
        )
        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)
        out = self.out_projection(out)
        if self.returnA:
            return out, A
        else:
            return out, None


class STEncoderLayer(nn.Module):
    def __init__(self, factor, d_model, n_heads, input_length, time_factor, num_nodes=None,
                 d_ff=None, dropout=0.1, att_dropout=0.1, return_att=False):
        super().__init__()
        d_ff = d_ff or 4*d_model
        self.return_att = return_att

        assert num_nodes is not None

        self.proxy_token = nn.Parameter(torch.zeros(1, factor, d_model)) # proxy is learnable parameters
        self.time_factor = time_factor
        self.time_readout = nn.Sequential(nn.Conv2d(input_length, time_factor, kernel_size=(1,1)), 
                                          nn.GELU(),
                                          nn.Conv2d(time_factor, time_factor, kernel_size=(1,1)))
        self.time_recover = nn.Sequential(nn.Conv2d(time_factor, time_factor, kernel_size=(1,1)), 
                                          nn.GELU(),
                                          nn.Conv2d(time_factor, input_length, kernel_size=(1,1)))
        
        self.node2proxy = AttentionLayer(d_model, n_heads, dropout=att_dropout, returnA=return_att)
        self.proxy2node = AttentionLayer(d_model, n_heads, dropout=att_dropout, returnA=return_att)

        self.dropout = nn.Dropout(dropout)

        self.FFN = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))

    def forward(self, data):  # data:BTNC
        batch = data.shape[0]
        T = data.shape[1]

        z_proxy = repeat(self.proxy_token,'o m d -> (o b) m d', b = batch*self.time_factor)
        
        kv_data = self.time_readout(data)
        kv_data = rearrange(kv_data, 'b t n c-> (b t) n c')

        proxy_feature, A1 = self.node2proxy(z_proxy, kv_data, kv_data)
        node_feature, A2 = self.proxy2node(kv_data, proxy_feature, proxy_feature)
        enc_feature = kv_data + self.dropout(node_feature)
        enc_feature = enc_feature + self.dropout(self.FFN(enc_feature))

        final_out = rearrange(enc_feature, '(b T) N d -> b T N d', b=batch)
        final_out = self.time_recover(final_out)

        if self.return_att:
            A1 = rearrange(A1, '(b t) h l s -> b t h l s', b=batch)
            A2 = rearrange(A2, '(b t) h l s -> b t h l s', b=batch)
            return final_out, [A1, A2]
        else:
            return final_out, None


class Decoder_pred(nn.Module):
    def __init__(self, num_nodes, input_length, predict_length, d_model, pre_dim,
                 activation='gelu'):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_length = input_length
        self.d_model = d_model
        self.pre_dim = pre_dim
        self.predict_length = predict_length
        assert activation in ['gelu', 'relu']
        self.end_conv1 = nn.Conv2d(in_channels=input_length, out_channels=predict_length, kernel_size=1, bias=True)
        self.end_conv2 = nn.Linear(d_model, pre_dim)

    def forward(self, data):
        rep = self.end_conv1(data)
        data = self.end_conv2(rep)
        return data


class Decoder_recon(nn.Module):
    def __init__(self, num_nodes, input_length,  d_model, in_dim, hasRes=True):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_length = input_length
        self.d_model = d_model
        self.hasRes = hasRes
        self.end_conv1 = nn.Sequential(nn.Conv2d(in_channels=input_length, out_channels=input_length, kernel_size=1, bias=True),
                                       nn.GELU())
        self.end_conv2 = nn.Linear(d_model, in_dim)

    def forward(self, data):
        skip = data
        rep = self.end_conv1(data)
        if self.hasRes: rep = rep + skip
        data = self.end_conv2(rep)
        return data

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.input_length = args.input_length
        self.predict_length = args.predict_length
        self.in_dim = args.in_dim
        self.pre_dim = args.in_dim if args.pre_dim is None else args.pre_dim
        self.num_nodes = args.num_nodes
        self.tau = args.tau
        self.d_model = args.d_model
        self.hasCross = bool(args.hasCross)
        self.num_layers = args.num_layers
        self.mask_ratio = args.mask_ratio
        self.mask_only = args.loss_type == 'mask_only'
        self.pred_only = args.loss_type == 'pred_only'

        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, args.d_model))
        self.data_encoding = Projector_1(args.in_dim, args.d_model, self.hasCross)
        self.get_spatial_emb = get_SpatialEmb(args.d_model, args.num_nodes, args.st_emb_dropout)
        self.get_temporal_emb = get_TemporalEmb(args.in_dim, args.d_model, args.slice_size_per_day, args.st_emb_dropout)
        self.tcn_encoding = Projector_2(args.tau, args.d_model)

        self.spatial_agg_list = nn.ModuleList([
                    STEncoderLayer(args.M, args.d_model, n_heads=args.n_heads, num_nodes=args.num_nodes, dropout=args.spatial_dropout,
                                att_dropout=args.spatial_att_dropout,
                                time_factor=args.time_factor, input_length=args.input_length) for _ in range(args.num_layers)])
        self.rep_dropout = nn.Dropout(args.rep_dropout)
        self.predictor = Decoder_pred(args.num_nodes, args.input_length, args.predict_length, args.d_model, pre_dim=self.pre_dim)
        self.reconstructor = Decoder_recon(args.num_nodes, args.input_length, args.d_model, args.in_dim)
        self.note = f'{args.input_length}to{args.predict_length}_{args.note}'
    
    def _getnote(self):
        return self.note
    
    def _input_processing(self, x):
        inputx = x[0]
        if len(x)==2:
            x_data, x_time_mark = x  # x: (B,T,N,C) t_hour:(B,T,1) t_day:(B,T,1)
            B, T, N, _ = x_data.shape
        elif len(x)==3:
            x_data, x_time_mark, y_data = x
            B, T, N, _ = x_data.shape
        
        x_t_hour = x_time_mark[...,0:1]
        x_t_day = x_time_mark[...,1:2] 
        latestX = x_data[:, -1:, :, :]

        if len(x) == 3:
            return inputx, latestX, x_t_hour, x_t_day, y_data
        else:
            return inputx, latestX, x_t_hour, x_t_day
    
    def _mask_raw_data(self, data, unmasked_token_index=None, masked_token_index=None):
        B, T, N, C = data.shape

        MaskEngine = MaskGenerator(T, self.mask_ratio)
        unmasked_token_index, masked_token_index = MaskEngine.uniform_rand()
        data_o = data[:, unmasked_token_index, :, :]   

        return data_o, unmasked_token_index, masked_token_index

    def get_rep(self, x):
        group =self._input_processing(x)
        if len(x) == 3:
            inputx, latestX, x_t_hour, x_t_day, inputy = group
        else:
            inputx, latestX, x_t_hour, x_t_day = group

        spatial_emb = self.get_spatial_emb(inputx)
        x_tod_emb, x_dow_emb = self.get_temporal_emb(x_t_hour, x_t_day)
        data = self.data_encoding(inputx, latestX)
        data = data + x_tod_emb + x_dow_emb+ spatial_emb
        data = self.tcn_encoding(data)
        rep = self._encode(data)
        return rep, latestX

    def _encode(self, data):
        skip = data
        for i in range(self.num_layers):
            data, _ = self.spatial_agg_list[i](data)
        data += skip
        return data

    
    def forward(self, x):
        group = self._input_processing(x)
        assert len(x) == 3
        inputx, latestX, x_t_hour, x_t_day, inputy = group
        

        spatial_emb = self.get_spatial_emb(inputx)
        x_tod_emb, x_dow_emb = self.get_temporal_emb(x_t_hour, x_t_day)
        x_o, x_unmasked_token_index, x_masked_token_index =self._mask_raw_data(inputx)
        
        x_tod_emb_o = x_tod_emb[:, x_unmasked_token_index, :, :]
        x_dow_emb_o = x_dow_emb[:, x_unmasked_token_index, :, :]
        semb_o = spatial_emb

        x_data_o  = self.data_encoding(x_o, latestX)
        x_data_o =  x_data_o + x_tod_emb_o + x_dow_emb_o+ semb_o
        x_data_o = self.tcn_encoding(x_data_o)

        B, T_, N_, d = x_data_o.shape
        x_tod_emb_m = x_tod_emb[:, x_masked_token_index,:,:] # B, T, 1, d
        x_dow_emb_m = x_dow_emb[:, x_masked_token_index,:,:]
        x_mask_token = self.mask_token.expand(B, len(x_masked_token_index), N_, d)
        x_mask_token = x_mask_token + x_tod_emb_m + x_dow_emb_m
        x_data_full = torch.cat([x_mask_token, x_data_o], dim=1)
        x_rep_full = self._encode(x_data_full)

        if self.pred_only:
            reconx = None
        else:
            reconx = self.reconstructor(x_rep_full)

        if self.mask_only:
            predy = None
        else:
            x_rep_drop = self.rep_dropout(x_rep_full)
            predy = self.predictor(x_rep_drop)

        return inputx, reconx, inputy, predy
