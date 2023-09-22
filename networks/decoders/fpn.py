from numpy import identity
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.layers.basic import ConvGN


class FPNSegmentationHead(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 decode_intermediate_input=True,
                 hidden_dim=256,
                 shortcut_dims=[24, 32, 96, 1280],
                 gn_groups=8,
                 align_corners=True,
                 use_adapters=True,
                 use_shortcuts=True):
        super().__init__()
        self.align_corners = align_corners

        self.decode_intermediate_input = decode_intermediate_input

        if isinstance(hidden_dim,(list,tuple)):
            self.conv_in = ConvGN(in_dim, hidden_dim[0], 1, gn_groups=gn_groups)
            self.conv_16x = ConvGN(hidden_dim[0], hidden_dim[1], 3, gn_groups=gn_groups)
            self.conv_8x = ConvGN(hidden_dim[1], hidden_dim[2], 3, gn_groups=gn_groups)
            self.conv_4x = ConvGN(hidden_dim[2], hidden_dim[3], 3, gn_groups=gn_groups)
        else:
            self.conv_in = ConvGN(in_dim, hidden_dim, 1, gn_groups=gn_groups)
            self.conv_16x = ConvGN(hidden_dim, hidden_dim, 3, gn_groups=gn_groups)
            self.conv_8x = ConvGN(hidden_dim, hidden_dim // 2, 3, gn_groups=gn_groups)
            self.conv_4x = ConvGN(hidden_dim // 2, hidden_dim // 2, 3, gn_groups=gn_groups)
        
        self.use_shortcuts = use_shortcuts
        if use_shortcuts == False:
            use_adapters = False
        
        if use_adapters:
            if isinstance(hidden_dim,(list,tuple)):
                self.adapter_16x = nn.Conv2d(shortcut_dims[-2], hidden_dim[0], 1)
                self.adapter_8x = nn.Conv2d(shortcut_dims[-3], hidden_dim[1], 1)
                self.adapter_4x = nn.Conv2d(shortcut_dims[-4], hidden_dim[2], 1)
            else:
                self.adapter_16x = nn.Conv2d(shortcut_dims[-2], hidden_dim, 1)
                self.adapter_8x = nn.Conv2d(shortcut_dims[-3], hidden_dim, 1)
                self.adapter_4x = nn.Conv2d(shortcut_dims[-4], hidden_dim // 2, 1)
        else:
            self.adapter_16x = torch.nn.Identity()
            self.adapter_8x = torch.nn.Identity()
            self.adapter_4x = torch.nn.Identity()
        
        if isinstance(hidden_dim,(list,tuple)):
            self.conv_out = nn.Conv2d(hidden_dim[3], out_dim, 1)
        else:
            self.conv_out = nn.Conv2d(hidden_dim // 2, out_dim, 1)

        self._init_weight()

    def forward(self, inputs, shortcuts, return_xs=False, return_act=True):

        xs = []
        if self.decode_intermediate_input:
            x = torch.cat(inputs, dim=1)
        else:
            if isinstance(inputs,(list,tuple)):
                x = inputs[-1]
            else:
                x = inputs
        x = self.conv_in(x)
        if not return_act:
            xs.append(x)
        x = F.relu_(x)
        if return_act:
            xs.append(x)
        
        if self.use_shortcuts:
            x = self.conv_16x(self.adapter_16x(shortcuts[-2]) + x)
        else:
            x = self.conv_16x(x)
        if not return_act:
            xs.append(x)
        x = F.relu_(x)
        if return_act:
            xs.append(x)

        x = F.interpolate(x,
                          size=shortcuts[-3].size()[-2:],
                          mode="bilinear",
                          align_corners=self.align_corners)
        if self.use_shortcuts:
            x = self.conv_8x(self.adapter_8x(shortcuts[-3]) + x)
        else:
            x = self.conv_8x(x)
        if not return_act:
            xs.append(x)
        x = F.relu_(x)
        if return_act:
            xs.append(x)

        x = F.interpolate(x,
                          size=shortcuts[-4].size()[-2:],
                          mode="bilinear",
                          align_corners=self.align_corners)
        if self.use_shortcuts:
            x = self.conv_4x(self.adapter_4x(shortcuts[-4]) + x)
        else:
            x = self.conv_4x(x)
        if not return_act:
            xs.append(x)
        x = F.relu_(x)
        if return_act:
            xs.append(x)
        
        x = self.conv_out(x)
        xs.append(x)

        if return_xs:
            return xs
        else:
            return x

    def _init_weight(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class ScalableFPNSegmentationHead(nn.Module):
    def __init__(self,
                 in_dims,
                 out_dim,
                 decode_intermediate_input=True,
                 hidden_dim=256,
                 shortcut_dims=[24, 32, 96, 1280],
                 align_corners=True):
        super().__init__()
        self.align_corners = align_corners

        self.decode_intermediate_input = decode_intermediate_input

        self.conv_in = []
        for in_dim in in_dims:
            self.conv_in.append(ConvGN(in_dim, hidden_dim, 1))
        self.conv_in = nn.ModuleList(self.conv_in)

        self.conv_16x = ConvGN(hidden_dim, hidden_dim, 3)
        self.conv_8x = ConvGN(hidden_dim, hidden_dim // 2, 3)
        self.conv_4x = ConvGN(hidden_dim // 2, hidden_dim // 2, 3)

        self.adapter_16x = nn.Conv2d(shortcut_dims[-2], hidden_dim, 1)
        self.adapter_8x = nn.Conv2d(shortcut_dims[-3], hidden_dim, 1)
        self.adapter_4x = nn.Conv2d(shortcut_dims[-4], hidden_dim // 2, 1)

        self.conv_out = nn.Conv2d(hidden_dim // 2, out_dim, 1)

        self._init_weight()

    def forward(self, inputs, shortcuts, conv_in_idx=0):

        if self.decode_intermediate_input:
            x = torch.cat(inputs, dim=1)
        else:
            x = inputs[-1]

        conv_in_idx = min(conv_in_idx, len(self.conv_in))

        x = F.relu_(self.conv_in[conv_in_idx](x))
        x = F.relu_(self.conv_16x(self.adapter_16x(shortcuts[-2]) + x))

        x = F.interpolate(x,
                          size=shortcuts[-3].size()[-2:],
                          mode="bilinear",
                          align_corners=self.align_corners)
        x = F.relu_(self.conv_8x(self.adapter_8x(shortcuts[-3]) + x))

        x = F.interpolate(x,
                          size=shortcuts[-4].size()[-2:],
                          mode="bilinear",
                          align_corners=self.align_corners)
        x = F.relu_(self.conv_4x(self.adapter_4x(shortcuts[-4]) + x))

        x = self.conv_out(x)

        return x

    def _init_weight(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
