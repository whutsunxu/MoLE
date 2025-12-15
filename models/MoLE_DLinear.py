import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import hashlib
from utils.headdropout import HeadDropout

import ttnn

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class Model(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, configs):
        super(Model, self).__init__()


        self.num_predictions = configs.t_dim

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs.individual
        self.channels = configs.enc_in


        # time feature size
        self.expected_time_features = 4 if configs.freq.lower().endswith('h') else 5


        self.Linear_Seasonal = nn.Linear(self.seq_len,self.pred_len * self.num_predictions)
        self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len * self.num_predictions)

        input_dim = self.expected_time_features
        self.Linear_Temporal = nn.Sequential(
            nn.Linear(input_dim, self.num_predictions * self.channels),
            nn.ReLU(),
            nn.Linear(self.num_predictions * self.channels, self.num_predictions * self.channels)
        )

        self.head_dropout = HeadDropout(configs.head_dropout)

        self.kernel_size = configs.kernel_size
        self.stride=1


    def forward(self, x, x_mark, return_gating_weights=False, return_seperate_head=False, device=None, type=ttnn.float32):
        # x: [Batch, Input length, Channel]
        # print("x: {}, x_mark: {}".format(x.shape, x_mark.shape))
        x_mark_initial = None
        seasonal_init, trend_init=None,None
        if device==None:
            x_mark_initial = x_mark[:,0]
            # print("x_mark_initial: {}".format(x_mark_initial.shape))
            seasonal_init, trend_init = self.decompsition(x)
            # print("seasonal_init(res): {}, trend_init(avg): {}".format(seasonal_init.shape, trend_init.shape))
            seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
            # print("after permutation, seasonal_init(res): {}, trend_init(avg): {}".format(seasonal_init.shape, trend_init.shape))
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            # print("seasonal_output: {}".format(seasonal_output.shape))
            trend_output = self.Linear_Trend(trend_init)

            # print("trend_output: {}".format(trend_output.shape))
            x = seasonal_output + trend_output

            temporal_out = self.Linear_Temporal(x_mark_initial)
            temporal_out = temporal_out.reshape(-1, self.num_predictions)

            # print("temporal_out: {}".format(temporal_out.shape))
            temporal_out = self.head_dropout(temporal_out)
            temporal_out = nn.Softmax(dim=1)(temporal_out)

            x_raw = x.reshape(-1, self.pred_len, self.num_predictions)

        else:
            "---------------------------- slice --------------------------"
            x_mark_initial = ttnn.slice(
                x_mark, slice_start=(0, 0, 0), slice_end=(x_mark.shape[0], 1, x_mark.shape[2]), slice_step=(1, 1, 1)
            )
            # x_mark_initial=ttnn.to_torch(tt_x_mark_initial)

            "---------------------------- mov avg --------------------------"

            x_low_h = ttnn.slice(
                x, slice_start=(0, 0, 0), slice_end=(x.shape[0], 1, x.shape[2]), slice_step=(1, 1, 1)
            )
            x_high_h = ttnn.slice(
                x, slice_start=(0, x.shape[1]-1, 0), slice_end=(x.shape[0], x.shape[1], x.shape[2]), slice_step=(1, 1, 1)
            )

            x_low_h_repeat=ttnn.repeat(x_low_h, (1, (self.kernel_size - 1) // 2, 1))
            x_high_h_repeat=ttnn.repeat(x_high_h, (1, (self.kernel_size - 1) // 2, 1))

            x_pad = ttnn.concat([x_low_h_repeat, x, x_high_h_repeat], dim=1)  ## N H C

            x_nch=ttnn.permute(x_pad, (0, 2, 1))  ## N C H

            host_x_nch=ttnn.to_torch(x_nch)
            avgpool1d = torch.nn.AvgPool1d(kernel_size=self.kernel_size, stride=self.stride, padding=0)
            host_avg_x_nch=avgpool1d(host_x_nch)
            # print("avgpool1d: ", avgpool1d)
            # print("host_x_nch: ", host_x_nch.shape)
            # print("host_avg_x_nch: ", host_avg_x_nch.shape)

            avg_x_nch=ttnn.from_torch(host_avg_x_nch, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=type, device=device)

            trend_init=ttnn.permute(avg_x_nch, (0, 2, 1))  ## N H C
            # print("trend_init: ",trend_init.shape)
            seasonal_init=x-trend_init

            trend_init_nch=ttnn.permute(trend_init, (0, 2, 1))  ## N C H
            seasonal_init_nch=ttnn.permute(seasonal_init, (0, 2, 1))  ## N C H

            "---------------------------- linear season --------------------------"
            # print("after permutation, seasonal_init(res): {}, trend_init(avg): {}".format(seasonal_init.shape, trend_init.shape))
            weight_season=self.Linear_Seasonal.weight
            weight_season=ttnn.from_torch(weight_season, layout=ttnn.TILE_LAYOUT, device=device, dtype=type)

            bias_season=self.Linear_Seasonal.bias
            bias_season=ttnn.from_torch(bias_season, layout=ttnn.TILE_LAYOUT, device=device, dtype=type)

            seasonal_init = ttnn.to_layout(seasonal_init_nch, layout=ttnn.TILE_LAYOUT)

            # print("input:{}, weight: {}, bias: {}".format(seasonal_init.shape, weight_season.shape, bias_season.shape))

            seasonal_output=ttnn.linear(seasonal_init, weight_season, bias=bias_season, transpose_b=True)

            "---------------------------- linear tread --------------------------"
            weight_tread=self.Linear_Trend.weight
            weight_tread=ttnn.from_torch(weight_tread, layout=ttnn.TILE_LAYOUT, device=device, dtype=type)

            bias_tread=self.Linear_Trend.bias
            bias_tread=ttnn.from_torch(bias_tread, layout=ttnn.TILE_LAYOUT, device=device, dtype=type)
            trend_init_nch = ttnn.to_layout(trend_init_nch, layout=ttnn.TILE_LAYOUT)

            trend_output=ttnn.linear(trend_init_nch, weight_tread, bias=bias_tread, transpose_b=True)

            x = seasonal_output + trend_output
            # x=ttnn.to_torch(x)


            "---------------------------- linear temp --------------------------"
            # temporal_out = self.Linear_Temporal(x_mark_initial).reshape(-1, self.num_predictions)

            weight_temp0=self.Linear_Temporal[0].weight
            weight_temp0=ttnn.from_torch(weight_temp0, layout=ttnn.TILE_LAYOUT, device=device, dtype=type)

            bias_temp0=self.Linear_Temporal[0].bias
            bias_temp0=ttnn.from_torch(bias_temp0, layout=ttnn.TILE_LAYOUT, device=device, dtype=type)
            x_mark_initial = ttnn.to_layout(x_mark_initial, layout=ttnn.TILE_LAYOUT)
            output_temp0=ttnn.linear(x_mark_initial, weight_temp0, bias=bias_temp0, transpose_b=True)

            output_temp1=ttnn.relu(output_temp0)

            weight_temp2=self.Linear_Temporal[2].weight
            weight_temp2=ttnn.from_torch(weight_temp2, layout=ttnn.TILE_LAYOUT, device=device, dtype=type)

            bias_temp2=self.Linear_Temporal[2].bias
            bias_temp2=ttnn.from_torch(bias_temp2, layout=ttnn.TILE_LAYOUT, device=device, dtype=type)
            output_temp2=ttnn.linear(output_temp1, weight_temp2, bias=bias_temp2, transpose_b=True)

            temporal_out = ttnn.reshape(output_temp2, (-1, self.num_predictions))

            "---------------------------- softmax --------------------------"
            # print("temporal_out: {}".format(temporal_out.shape))
            temporal_out = ttnn.softmax(temporal_out, dim=1)  ## (TODO softmax brings a lot of precision loss from 1e-4 upto 1e-3)

            temporal_out=ttnn.to_torch(temporal_out)

            # x_raw = x.reshape(-1, self.pred_len, self.num_predictions)

            x_raw=ttnn.reshape(x, (-1, self.pred_len, self.num_predictions))

            x_raw=ttnn.to_torch(x_raw)

        # print("temporal_out: {}".format(temporal_out.shape))

        x = torch.matmul(x_raw, temporal_out.unsqueeze(2)).squeeze(2).reshape(-1, self.channels, self.pred_len).permute(0,2,1)
        # print("x_raw: {}, temporal_out.unsqueeze(2): {}, x: {}".format(x_raw.shape, temporal_out.unsqueeze(2).shape, x.shape))

        return x
