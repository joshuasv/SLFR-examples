import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from src.models.encoders.ASLR import TransformerEncoder

from IPython import embed; from sys import exit

class Encoder(nn.Module):

    def __init__(
            self,
            num_hid,
            msg3d_num_keypoints,
            msg3d_channels_per_keypoint,
            msg3d_num_gcn_scales,
            msg3d_dropout,
            tformer_num_head,
            tformer_num_feed_forward,
            tformer_dropout,
            tformer_n_layers,
        ):
        super().__init__()

        self.seq_emb = MSG3D(
            msg3d_num_keypoints,
            msg3d_channels_per_keypoint,
            msg3d_num_gcn_scales,
            num_hid,
            msg3d_dropout,
        )
        self.encoder = nn.ModuleList(
            [TransformerEncoder(num_hid, tformer_num_head, tformer_num_feed_forward, tformer_dropout) for _ in range(tformer_n_layers)]
        )

    def forward(self, seq, seq_mask):
        # seq = (batch, seq_len, features)
        seq = self.seq_emb(seq)
        
        for l in self.encoder:
            seq = l(seq, seq_mask)

        return seq
    

class MSG3D(nn.Module):

    def __init__(
            self,
            num_keypoints,
            channels_per_keypoint,
            num_gcn_scales,
            num_hid,
            dropout,
        ):
        super().__init__()

        self.num_keypoints = num_keypoints
        self.channels = channels_per_keypoint

        A_binary = AdjMatrixGraph(num_keypoints).A_binary
        self.data_bn = nn.BatchNorm1d(channels_per_keypoint * num_keypoints)

        c1 = 30
        c2 = c1 * 2     # 60
        c3 = c2 * 2     # 120

        self.sgcn1 = nn.Sequential(
            MS_GCN(num_gcn_scales, channels_per_keypoint, c1, A_binary, dropout=dropout, disentangled_agg=True),
            # (batch, c1, seq_len, num_kps)
            MS_TCN(c1, c1, dilations=[1,3,4]),
            MS_TCN(c1, c1, dilations=[1,3,4]))
        self.sgcn1[-1].act = nn.Identity()
        self.tcn1 = MS_TCN(c1, c1, dilations=[1,3,4])

        self.sgcn2 = nn.Sequential(
            MS_GCN(num_gcn_scales, c1, c1, A_binary, dropout=dropout, disentangled_agg=True),
            MS_TCN(c1, c2, stride=1, dilations=[1,3,4]),
            MS_TCN(c2, c2, dilations=[1,3,4]))
        self.sgcn2[-1].act = nn.Identity()
        self.tcn2 = MS_TCN(c2, c2, dilations=[1,3,4])

        self.sgcn3 = nn.Sequential(
            MS_GCN(num_gcn_scales, c2, c2, A_binary, dropout=dropout, disentangled_agg=True),
            MS_TCN(c2, c3, stride=1, dilations=[1,3,4]),
            MS_TCN(c3, c3, dilations=[1,3,4]))
        self.sgcn3[-1].act = nn.Identity()
        self.tcn3 = MS_TCN(c3, c3, dilations=[1,3,4])

        self.fc = nn.Linear(c3*num_keypoints, num_hid)


    def forward(self, x):
        N, V, _ = x.size()
        x = x.view(N, V, self.num_keypoints, self.channels).permute((0, 3, 1, 2))
        N, C, T, V = x.size()
        
        # N C T V M => N C T V
        # 0 1 2 3 4 => 0 1 2 3
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T).permute(0,2,3,1).contiguous()

        x = F.relu(self.sgcn1(x))
        x = self.tcn1(x)

        x = F.relu(self.sgcn2(x))
        x = self.tcn2(x)

        x = F.relu(self.sgcn3(x))
        x = self.tcn3(x)

        out = x
        B, C, T, V = out.size()
        # (batch, c3, seq_len, num_kps)

        # out = out.max(3)[0].permute(0,2,1)

        # next option
        out = out.permute(0,2,1,3).contiguous().view(B, T, C*V)

        # (batch, seq_len, c3)
        # out = out.view(N, out_channels, -1)
        # out = out.mean(2)   # Global Average Pooling (Spatial+Temporal)
        out = self.fc(out)
        return out
    

class MS_GCN(nn.Module):
    """"Multi-sacle graph convolution"""
    def __init__(
            self,
            num_scales,
            in_channels,
            out_channels,
            A_binary,
            disentangled_agg=True,
            use_mask=True,
            dropout=0,
            activation='relu'
        ):
        super().__init__()
        self.num_scales = num_scales

        if disentangled_agg:
            A_powers = [k_adjacency(A_binary, k, with_self=True) for k in range(num_scales)]
            A_powers = np.concatenate([normalize_adjacency_matrix(g) for g in A_powers])
        else:
            A_powers = [A_binary + np.eye(len(A_binary)) for k in range(num_scales)]
            A_powers = [normalize_adjacency_matrix(g) for g in A_powers]
            A_powers = [np.linalg.matrix_power(g, k) for k, g in enumerate(A_powers)]
            A_powers = np.concatenate(A_powers)

        self.register_buffer('A_powers', torch.Tensor(A_powers).to(torch.float32))
        self.use_mask = use_mask
        if use_mask:
            # NOTE: the inclusion of residual mask appears to slow down training noticeably
            self.A_res = nn.init.uniform_(nn.Parameter(torch.Tensor(self.A_powers.shape)), -1e-6, 1e-6)

        self.mlp = MLP(in_channels * num_scales, [out_channels], dropout=dropout, activation=activation)

    def forward(self, x):
        N, C, T, V = x.shape
        A = self.A_powers.to(x.dtype)
        if self.use_mask:
            A = A + self.A_res.to(x.dtype)
        
        support = torch.einsum('vu,nctu->nctv', A, x)
        support = support.view(N, C, T, self.num_scales, V)
        support = support.permute(0,3,1,2,4).contiguous().view(N, self.num_scales*C, T, V)
        out = self.mlp(support)
        return out
    

class MS_TCN(nn.Module):
    """Multi-scale temporal convolution"""
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            dilations=[1,2,3,4],
            residual_attr=True,
            residual_kernel_size=1,
            activation='relu'
        ):
        super().__init__()

        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches

        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0,bias=False),
                nn.BatchNorm2d(branch_channels),
                activation_factory(activation),
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation),
            )
            for dilation in dilations
        ])

        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0,bias=False),
            nn.BatchNorm2d(branch_channels),
            activation_factory(activation),
            nn.AvgPool2d(kernel_size=(3,1), stride=(stride,1), padding=(1,0)),
            nn.BatchNorm2d(branch_channels)
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride,1),bias=False),
            nn.BatchNorm2d(branch_channels)
        ))

        # Residual connection
        temp_conv_module = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)
        identity_module = torch.nn.Identity()
        # residual_fn = self.get_residual(residual_attr, in_channels, out_channels, stride, identity_module, temp_conv_module)
        # self.residual = residual_fn
        if (in_channels == out_channels) and (stride == 1):
            self.residual = identity_module
        else:
            self.residual = temp_conv_module

        self.act = activation_factory(activation)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res
        out = self.act(out)
        return out


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1),bias=False)

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu', dropout=0):
        super().__init__()
        channels = [in_channels] + out_channels
        self.layers = nn.ModuleList()
        for i in range(1, len(channels)):
            if dropout > 0.0:
                self.layers.append(nn.Dropout(p=dropout))
            self.layers.append(nn.Conv2d(channels[i-1], channels[i], kernel_size=1,bias=False))
            self.layers.append(nn.BatchNorm2d(channels[i]))
            self.layers.append(activation_factory(activation))

    def forward(self, x):
        # Input shape: (N,C,T,V)
        for layer in self.layers:
            x = layer(x)
        return x


class AdjMatrixGraph:

    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        # self.A_binary = np.eye(num_nodes, dtype=np.float32)
        self.A_binary = np.ones((num_nodes, num_nodes), dtype=np.float32)


def k_adjacency(A, k, with_self=False, self_factor=1):
    assert isinstance(A, np.ndarray)
    I = np.eye(len(A), dtype=A.dtype)
    if k == 0:
        return I
    Ak = np.minimum(np.linalg.matrix_power(A + I, k), 1) \
       - np.minimum(np.linalg.matrix_power(A + I, k - 1), 1)
    if with_self:
        Ak += (self_factor * I)
    return Ak

def normalize_adjacency_matrix(A):
    node_degrees = A.sum(-1)
    degs_inv_sqrt = np.power(node_degrees, -0.5)
    norm_degs_matrix = np.eye(len(node_degrees)) * degs_inv_sqrt
    return (norm_degs_matrix @ A @ norm_degs_matrix).astype(np.float32)

def activation_factory(name, inplace=True):
    if name == 'relu':
        return nn.ReLU(inplace=inplace)
    elif name == 'leakyrelu':
        return nn.LeakyReLU(0.2, inplace=inplace)
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'linear' or name is None:
        return nn.Identity()
    else:
        raise ValueError('Not supported activation:', name)