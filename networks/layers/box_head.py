import torch.nn as nn
import torch
import torch.nn.functional as F
from networks.layers.normalization import FrozenBatchNorm2d
from networks.layers.basic import ConvGN
from networks.layers.transformer import TransformerHeadBlock
from networks.layers.position import PositionEmbeddingSine
# import time


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1,
         freeze_bn=False):
    if freeze_bn:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            FrozenBatchNorm2d(out_planes),
            nn.ReLU(inplace=True))
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))


class Corner_Predictor(nn.Module):
    """ Corner Predictor module"""

    def __init__(self, inplanes=64, outplanes=1, channel=256, downsample=1, align_corners=False, freeze_bn=False):
        super(Corner_Predictor, self).__init__()
        '''top-left corner'''
        if downsample>1:
            if align_corners:
                self.conv1_tl = conv(inplanes, channel, kernel_size=downsample+1, 
                                    stride=downsample, padding=downsample//2,
                                    freeze_bn=freeze_bn)
            else:
                self.conv1_tl = conv(inplanes, channel, kernel_size=downsample, 
                                    stride=downsample, padding=0,
                                    freeze_bn=freeze_bn)
        else:
            self.conv1_tl = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_tl = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_tl = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_tl = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_tl = nn.Conv2d(channel // 8, outplanes, kernel_size=1)

        '''bottom-right corner'''
        if downsample>1:
            if align_corners:
                self.conv1_br = conv(inplanes, channel, kernel_size=downsample+1, 
                                    stride=downsample, padding=downsample//2,
                                    freeze_bn=freeze_bn)
            else:
                self.conv1_br = conv(inplanes, channel, kernel_size=downsample, 
                                    stride=downsample, padding=0,
                                    freeze_bn=freeze_bn)
        else:
            self.conv1_br = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_br = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_br = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_br = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_br = nn.Conv2d(channel // 8, outplanes, kernel_size=1)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, return_dist=False, softmax=True):
        """ Forward pass with input x (b,c_in,h,w). 
        return (b,c_out,4)"""
        
        score_map_tl, score_map_br = self.get_score_map(x) # (b,c_out,h,w)
        input_h,input_w = score_map_br.shape[2:]
        if return_dist:
            coorx_tl, coory_tl, prob_vec_tl = self.soft_argmax(score_map_tl, return_dist=True, softmax=softmax)
            coorx_br, coory_br, prob_vec_br = self.soft_argmax(score_map_br, return_dist=True, softmax=softmax)
            return torch.stack((coorx_tl/input_w, coory_tl/input_h, coorx_br/input_w, coory_br/input_h), dim=2), prob_vec_tl, prob_vec_br
        else:
            coorx_tl, coory_tl = self.soft_argmax(score_map_tl)
            coorx_br, coory_br = self.soft_argmax(score_map_br)
            return torch.stack((coorx_tl/input_w, coory_tl/input_h, coorx_br/input_w, coory_br/input_h), dim=2)

    def get_score_map(self, x):
        # top-left branch
        x_tl1 = self.conv1_tl(x)
        x_tl2 = self.conv2_tl(x_tl1)
        x_tl3 = self.conv3_tl(x_tl2)
        x_tl4 = self.conv4_tl(x_tl3)
        score_map_tl = self.conv5_tl(x_tl4)

        # bottom-right branch
        x_br1 = self.conv1_br(x)
        x_br2 = self.conv2_br(x_br1)
        x_br3 = self.conv3_br(x_br2)
        x_br4 = self.conv4_br(x_br3)
        score_map_br = self.conv5_br(x_br4)
        return score_map_tl, score_map_br

    def soft_argmax(self, score_map, return_dist=False, softmax=True):
        '''about coordinates and indexs
        score_map: b,c,h,w
        '''
        (b,c,h,w) = score_map.shape
        with torch.no_grad():
            indice_x = torch.arange(0, w).view(-1, 1)
            indice_y = torch.arange(0,h).view(-1,1)
            # generate mesh-grid
            coord_x = indice_x.repeat((h, 1)) \
                .view((h*w,)).float().to(score_map.device)
            coord_y = indice_y.repeat((1, w)) \
                .view((h*w,)).float().to(score_map.device)
        """ get soft-argmax coordinate for a given heatmap """
        score_vec = score_map.view((b,c, h*w))
        prob_vec = F.softmax(score_vec, dim=2)
        exp_x = torch.sum((coord_x * prob_vec), dim=2)
        exp_y = torch.sum((coord_y * prob_vec), dim=2)
        if return_dist:
            if softmax:
                return exp_x, exp_y, prob_vec
            else:
                return exp_x, exp_y, score_vec
        else:
            return exp_x, exp_y

class Center_Predictor(nn.Module):
    def __init__(self, inplanes=64, channel=256, outplanes=1, freeze_bn=False):
        super().__init__()
        self.outplanes = outplanes
        # center predict
        self.conv1_ctr = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_ctr = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_ctr = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_ctr = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_ctr = nn.Conv2d(channel // 8, outplanes, kernel_size=1)

        # offset regress
        self.conv1_offset = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_offset = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_offset = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_offset = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_offset = nn.Conv2d(channel // 8, outplanes*2, kernel_size=1)

        # size regress
        self.conv1_size = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_size = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_size = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_size = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_size = nn.Conv2d(channel // 8, outplanes*2, kernel_size=1)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, gt_score_map=None, return_corners=True):
        """ Forward pass with input x. """
        score_map_ctr, size_map, offset_map = self.get_score_map(x)
        self.input_h,self.input_w = size_map.shape[2:]

        # assert gt_score_map is None
        if gt_score_map is None:
            bbox = self.cal_bbox(score_map_ctr, size_map, offset_map, return_corners=return_corners)
        else:
            bbox = self.cal_bbox(gt_score_map.unsqueeze(1), size_map, offset_map, return_corners=return_corners)

        return [score_map_ctr, bbox, size_map, offset_map]

    def cal_bbox(self, score_map_ctr, size_map, offset_map, return_score=False, return_corners=True):
        max_score, idx = torch.max(score_map_ctr.flatten(2), dim=2, keepdim=True)
        idx_y = idx // self.input_w
        idx_x = idx % self.input_w

        idx = idx.repeat(1, 2, 1) #(B,2*11,1)
        size = size_map.flatten(2).gather(dim=2, index=idx)
        offset = offset_map.flatten(2).gather(dim=2, index=idx)

        # bbox = torch.cat([idx_x - size[:, 0] / 2, idx_y - size[:, 1] / 2,
        #                   idx_x + size[:, 0] / 2, idx_y + size[:, 1] / 2], dim=1) / self.feat_sz
        cx = (idx_x.to(torch.float) + offset[:, :self.outplanes]) / self.input_w
        cy = (idx_y.to(torch.float) + offset[:, self.outplanes:]) / self.input_h
        w = size[:,:self.outplanes]
        h = size[:,self.outplanes:]
        if return_corners:
            # x1, y1, x2, y2
            x1 = cx - w/2
            x2 = cx + w/2
            y1 = cy - h/2
            y2 = cy + h/2
            bbox = torch.cat([x1,y1,x2,y2], dim=-1)
        else:
            # cx, cy, w, h
            bbox = torch.cat([cx,cy,w,h], dim=-1)

        if return_score:
            return bbox, max_score
        return bbox

    # def get_pred(self, score_map_ctr, size_map, offset_map):
    #     max_score, idx = torch.max(score_map_ctr.flatten(1), dim=1, keepdim=True)
    #     idx_y = idx // self.feat_sz
    #     idx_x = idx % self.feat_sz

    #     idx = idx.unsqueeze(1).expand(idx.shape[0], 2, 1)
    #     size = size_map.flatten(2).gather(dim=2, index=idx)
    #     offset = offset_map.flatten(2).gather(dim=2, index=idx).squeeze(-1)

    #     # bbox = torch.cat([idx_x - size[:, 0] / 2, idx_y - size[:, 1] / 2,
    #     #                   idx_x + size[:, 0] / 2, idx_y + size[:, 1] / 2], dim=1) / self.feat_sz
    #     return size * self.feat_sz, offset

    def get_score_map(self, x):

        def _sigmoid(x):
            y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
            return y

        # ctr branch
        x_ctr1 = self.conv1_ctr(x)
        x_ctr2 = self.conv2_ctr(x_ctr1)
        x_ctr3 = self.conv3_ctr(x_ctr2)
        x_ctr4 = self.conv4_ctr(x_ctr3)
        score_map_ctr = self.conv5_ctr(x_ctr4)

        # offset branch
        x_offset1 = self.conv1_offset(x)
        x_offset2 = self.conv2_offset(x_offset1)
        x_offset3 = self.conv3_offset(x_offset2)
        x_offset4 = self.conv4_offset(x_offset3)
        score_map_offset = self.conv5_offset(x_offset4)

        # size branch
        x_size1 = self.conv1_size(x)
        x_size2 = self.conv2_size(x_size1)
        x_size3 = self.conv3_size(x_size2)
        x_size4 = self.conv4_size(x_size3)
        score_map_size = self.conv5_size(x_size4)
        return _sigmoid(score_map_ctr), _sigmoid(score_map_size), score_map_offset



class PinpointHead(nn.Module):
    def __init__(self,
                inplanes=256,
                outplanes=11,
                num_layers=3,
                d_model=256,
                att_nhead=8,
                dim_feedforward=1024,
                droppath=0.1,
                softmax_ctr=False,
                sep_token=False,
                xy_pooling=False,
                xy_pooling_type='avg',
                preconv=False,
                reverse_pooling=False,
                pos_type='sin',
                norm_align=False) -> None:
        super().__init__()
        self.softmax_ctr = softmax_ctr
        self.num_layers = num_layers
        self.outplanes = outplanes
        self.pos_type = pos_type
        self.xy_pooling = xy_pooling
        self.xy_pooling_type = xy_pooling_type
        self.use_preconv = preconv
        self.reverse_pooling=reverse_pooling
        if pos_type == 'sin':
            self.pos_generator = PositionEmbeddingSine(
                                d_model // 2, normalize=True)
        self.conv_in = ConvGN(inplanes,d_model,3)
        self.sep_token = sep_token
        self.norm_align = norm_align
        if self.sep_token:
            self.token1 = nn.Parameter(torch.zeros(1, 1, d_model))
            self.token2 = nn.Parameter(torch.zeros(1, 1, d_model))
            self.token3 = nn.Parameter(torch.zeros(1, 1, d_model))
            self.token4 = nn.Parameter(torch.zeros(1, 1, d_model))
        elif self.xy_pooling:
            pass
        else:
            self.token1 = nn.Parameter(torch.zeros(1, 1, d_model))
            self.token2 = nn.Parameter(torch.zeros(1, 1, d_model))
        block = TransformerHeadBlock
        layers = []
        for i in range(num_layers):
            layers.append(
                block(d_model, att_nhead, dim_feedforward,droppath))
        self.layers = nn.ModuleList(layers)
        if self.sep_token:
            self.linear11 = nn.Linear(d_model,outplanes)
            self.linear12 = nn.Linear(d_model,outplanes)
            self.linear21 = nn.Linear(d_model,outplanes)
            self.linear22 = nn.Linear(d_model,outplanes)
        elif self.xy_pooling:
            if self.use_preconv:
                def pre_conv(d_model,outplanes):
                    return nn.Sequential(ConvGN(d_model,d_model//4,3),nn.ReLU(),
                                    nn.Conv2d(d_model//4,outplanes,1))
                self.preconv_x1 = pre_conv(d_model,outplanes)
                self.preconv_y1 = pre_conv(d_model,outplanes)
                self.preconv_x2 = pre_conv(d_model,outplanes)
                self.preconv_y2 = pre_conv(d_model,outplanes)
            else:
                def conv1d(d_model,outplanes):
                    return nn.Sequential(nn.Conv1d(d_model,d_model,1),
                                        nn.GroupNorm(8,d_model),
                                        nn.ReLU(),
                                        nn.Conv1d(d_model,outplanes,1))
                self.conv_x1 = conv1d(d_model,outplanes)
                self.conv_y1 = conv1d(d_model,outplanes)
                self.conv_x2 = conv1d(d_model,outplanes)
                self.conv_y2 = conv1d(d_model,outplanes)
        else:
            self.linear1 = nn.Linear(d_model,2*outplanes)
            self.linear2 = nn.Linear(d_model,2*outplanes)
        self.linear3 = nn.Linear(d_model,outplanes)
        
        self.init_weights()


    def forward(self,x, return_med=None):
        'x:(B,C,H,W)'
        x = self.conv_in(x)
        if self.pos_type == 'sin':
            pos_emb = self.pos_generator(x)
            x = x + pos_emb
        (b,c,h,w) = x.shape
        x = x.view(b,c,h*w).permute(2,0,1)
        if self.sep_token:
            x = torch.cat((self.token1.expand(-1, x.shape[1], -1), x), dim=0)
            x = torch.cat((self.token2.expand(-1, x.shape[1], -1), x), dim=0)
            x = torch.cat((self.token3.expand(-1, x.shape[1], -1), x), dim=0)
            x = torch.cat((self.token4.expand(-1, x.shape[1], -1), x), dim=0)

            for idx, layer in enumerate(self.layers):
                x = layer(x)

            tl_x = self._sigmoid(self.linear11(x[0,:,:])).unsqueeze(-1) #(B,C) -> (B,11,1)
            tl_y = self._sigmoid(self.linear12(x[1,:,:])).unsqueeze(-1)
            br_x = self._sigmoid(self.linear21(x[2,:,:])).unsqueeze(-1)
            br_y = self._sigmoid(self.linear22(x[3,:,:])).unsqueeze(-1)
            bbox = torch.cat([tl_x,tl_y,br_x,br_y],dim=-1) #(B,11,4)
            
            score_map_ctr = self.linear3(x[4:,:,:])
        elif self.xy_pooling:
            for idx, layer in enumerate(self.layers):
                x = layer(x)
            x = x.permute(1,2,0).view(b,c,h,w) #(HW,B,C) -> (B,C,H,W)
            if self.use_preconv:
                def preconv_softmax(fun,x):
                    (b,c,h,w) = x.shape
                    _x = torch.softmax(fun(x).view(b,-1,h*w),dim=-1).view(b,-1,h,w)
                    # print(_x.shape)
                    return _x
                feat_x1 = torch.sum(preconv_softmax(self.preconv_x1,x),dim=-2)
                feat_y1 = torch.sum(preconv_softmax(self.preconv_y1,x),dim=-1)
                feat_x2 = torch.sum(preconv_softmax(self.preconv_x2,x),dim=-2)
                feat_y2 = torch.sum(preconv_softmax(self.preconv_y2,x),dim=-1)
                
                tl_x = self.soft_argmax1d(feat_x1,input_soft=True) # (B,11)
                tl_y = self.soft_argmax1d(feat_y1,input_soft=True)
                br_x = self.soft_argmax1d(feat_x2,input_soft=True)
                br_y = self.soft_argmax1d(feat_y2,input_soft=True)
                # print(feat_x1,feat_y1,feat_x2,feat_y2)
                # print(tl_x,tl_y,br_x,br_y)
                # exit()
            else:
                if self.xy_pooling_type == 'avg':
                    feat_x = torch.mean(x,dim=-2)
                    feat_y = torch.mean(x,dim=-1)
                elif self.xy_pooling_type == 'max':
                    feat_x = torch.max(x,dim=-2)[0]
                    feat_y = torch.max(x,dim=-1)[0]
                if self.reverse_pooling:
                    heat_x1 = self.conv_x1(feat_y)
                    heat_y1 = self.conv_y1(feat_x)
                    heat_x2 = self.conv_x2(feat_y)
                    heat_y2 = self.conv_y2(feat_x)
                else:
                    heat_x1 = self.conv_x1(feat_x)
                    heat_y1 = self.conv_y1(feat_y)
                    heat_x2 = self.conv_x2(feat_x)
                    heat_y2 = self.conv_y2(feat_y)
                tl_x = self.soft_argmax1d(heat_x1) # (B,11)
                tl_y = self.soft_argmax1d(heat_y1)
                br_x = self.soft_argmax1d(heat_x2)
                br_y = self.soft_argmax1d(heat_y2)
            bbox = torch.stack([tl_x,tl_y,br_x,br_y],dim=-1) #(B,11,4)
            if return_med=='feature':
                return [None,bbox,None,feat_x,feat_y]
            elif return_med=='heatmap':
                return [None,bbox,None,torch.cat([heat_x1,heat_x2],dim=1),torch.cat([heat_y1,heat_y2],dim=1)]
            else:
                return [None,bbox,None]
        else:
            x = torch.cat((self.token1.expand(-1, x.shape[1], -1), x), dim=0)
            x = torch.cat((self.token2.expand(-1, x.shape[1], -1), x), dim=0)

            for idx, layer in enumerate(self.layers):
                x = layer(x)

            tl = self._sigmoid(self.linear1(x[0,:,:])) #(B,C) -> (B,22)
            tl_x = tl[:,:self.outplanes].unsqueeze(-1) #(B,11,1)
            tl_y = tl[:,self.outplanes:].unsqueeze(-1)
            br = self._sigmoid(self.linear2(x[1,:,:])) #(B,C) -> (B,22)
            br_x = br[:,:self.outplanes].unsqueeze(-1)
            br_y = br[:,self.outplanes:].unsqueeze(-1)
            bbox = torch.cat([tl_x,tl_y,br_x,br_y],dim=-1) #(B,11,4)
            
            score_map_ctr = self.linear3(x[2:,:,:])
        
        score_map_ctr_2d = score_map_ctr.permute(1,2,0).view(b,self.outplanes,h,w)
        ctr_x,ctr_y = self.soft_argmax(score_map_ctr_2d)
        coord_ctr = torch.stack([ctr_x,ctr_y],dim=-1)
        if self.softmax_ctr:
            score_map_ctr = F.softmax(score_map_ctr,dim=0)
        else:
            score_map_ctr = self._sigmoid(score_map_ctr)
        score_map_ctr = score_map_ctr.permute(1,2,0).view(b,self.outplanes,h,w) #(HW,B,11)->(B,11,H,W)
        return [score_map_ctr,bbox,coord_ctr]
    
    @staticmethod
    def _sigmoid(x):
            y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
            return y
    def soft_argmax(self, score_map, return_dist=False, softmax=True, normalize=True, input_soft=False):
        '''about coordinates and indexs
        score_map: b,c,h,w
        '''
        (b,c,h,w) = score_map.shape
        with torch.no_grad():
            indice_x = torch.arange(0, w).view(-1, 1)
            indice_y = torch.arange(0,h).view(-1,1)
            # generate mesh-grid
            coord_x = indice_x.repeat((h, 1)) \
                .view((h*w,)).float().to(score_map.device)
            coord_y = indice_y.repeat((1, w)) \
                .view((h*w,)).float().to(score_map.device)
        """ get soft-argmax coordinate for a given heatmap """
        score_vec = score_map.view((b,c, h*w))
        if input_soft:
            prob_vec = score_vec
        else:
            prob_vec = F.softmax(score_vec, dim=2)
        exp_x = torch.sum((coord_x * prob_vec), dim=2)
        exp_y = torch.sum((coord_y * prob_vec), dim=2)
        if normalize:
            if self.norm_align:
                exp_x = exp_x/(w-1)
                exp_y = exp_y/(h-1)
            else:
                exp_x = exp_x/w
                exp_y = exp_y/h
        if return_dist:
            if softmax:
                return exp_x, exp_y, prob_vec
            else:
                return exp_x, exp_y, score_vec
        else:
            return exp_x, exp_y
    def soft_argmax1d(self, score_vec, normalize=True, input_soft=False):
        (b,c,l) = score_vec.shape
        with torch.no_grad():
            coords = torch.arange(0,l).float().to(score_vec.device)
        """ get soft-argmax coordinate for a given heatmap """
        if input_soft:
            prob_vec  = score_vec
        else:
            prob_vec = F.softmax(score_vec, dim=2)
        exp_l = torch.sum((coords * prob_vec), dim=2)
        if normalize:
            if self.norm_align:
                exp_l = exp_l / (l-1)
            else:
                exp_l = exp_l / l
        return exp_l

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        if self.sep_token:
            nn.init.normal_(self.token1, std=1e-6)
            nn.init.normal_(self.token2, std=1e-6)
            nn.init.normal_(self.token3, std=1e-6)
            nn.init.normal_(self.token4, std=1e-6)
        elif self.xy_pooling:
            pass
        else:
            nn.init.normal_(self.token1, std=1e-6)
            nn.init.normal_(self.token2, std=1e-6)

class FPNBoxHead(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 decode_intermediate_input=True,
                 hidden_dim=256,
                 shortcut_dims=[24, 32, 96, 1280],
                 gn_groups=8,
                 align_corners=True,
                 use_adapters=True,
                 use_shortcuts=True,
                 out_scale=8,
                 xy_pooling_type='avg',
                 norm_align=False,
                 preconv=False):
        super().__init__()
        self.align_corners = align_corners
        self.out_scale=out_scale
        if out_scale not in [16,8,4]:
            raise ValueError
        self.xy_pooling_type = xy_pooling_type

        self.decode_intermediate_input = decode_intermediate_input
        self.norm_align = norm_align
        self.use_preconv = preconv

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
        
        d_model = hidden_dim if self.out_scale == 16 else hidden_dim // 2
        if self.use_preconv:
            self.preconv_x1 = nn.Conv2d(d_model, out_dim,1)
            self.preconv_x2 = nn.Conv2d(d_model, out_dim,1)
            self.preconv_y1 = nn.Conv2d(d_model, out_dim,1)
            self.preconv_y2 = nn.Conv2d(d_model, out_dim,1)
        else:
            def conv1d(d_model,outplanes):
                return nn.Sequential(nn.Conv1d(d_model,d_model,1),
                                    nn.GroupNorm(8,d_model),
                                    nn.ReLU(),
                                    nn.Conv1d(d_model,outplanes,1))
            self.conv_x1 = conv1d(d_model,out_dim)
            self.conv_y1 = conv1d(d_model,out_dim)
            self.conv_x2 = conv1d(d_model,out_dim)
            self.conv_y2 = conv1d(d_model,out_dim)
        
        if isinstance(hidden_dim,(list,tuple)):
            self.conv_out = nn.Conv2d(hidden_dim[3], out_dim, 1)
        else:
            self.conv_out = nn.Conv2d(hidden_dim // 2, out_dim, 1)

        self._init_weight()

    def forward(self, inputs, shortcuts, return_xs=False):

        xs = []
        x = inputs
        x = self.conv_in(x)
        x = F.relu_(x)
        xs.append(x)
        
        if self.use_shortcuts:
            x = self.conv_16x(self.adapter_16x(shortcuts[-2]) + x)
        else:
            x = self.conv_16x(x)
        x = F.relu_(x)
        if self.out_scale == 16:
            return self.feat2box(x)
        xs.append(x)

        x = F.interpolate(x,
                          size=shortcuts[-3].size()[-2:],
                          mode="bilinear",
                          align_corners=self.align_corners)
        if self.use_shortcuts:
            x = self.conv_8x(self.adapter_8x(shortcuts[-3]) + x)
        else:
            x = self.conv_8x(x)
        x = F.relu_(x)
        if self.out_scale == 8:
            return self.feat2box(x)
        xs.append(x)

        x = F.interpolate(x,
                          size=shortcuts[-4].size()[-2:],
                          mode="bilinear",
                          align_corners=self.align_corners)
        if self.use_shortcuts:
            x = self.conv_4x(self.adapter_4x(shortcuts[-4]) + x)
        else:
            x = self.conv_4x(x)
        x = F.relu_(x)
        if self.out_scale == 4:
            return self.feat2box(x)
        xs.append(x)
        
        x = self.conv_out(x)
        xs.append(x)

        if return_xs:
            return xs
        else:
            return x
    
    def feat2box(self,x):
        (b,c,h,w) = x.shape
        if self.use_preconv:
            def preconv_softmax(fun,x):
                _x = torch.softmax(fun(x).view(b,-1,h*w),dim=-1).view(b,-1,h,w)
                # print(_x.shape)
                return _x
            feat_x1 = torch.sum(preconv_softmax(self.preconv_x1,x),dim=-2)
            feat_y1 = torch.sum(preconv_softmax(self.preconv_y1,x),dim=-1)
            feat_x2 = torch.sum(preconv_softmax(self.preconv_x2,x),dim=-2)
            feat_y2 = torch.sum(preconv_softmax(self.preconv_y2,x),dim=-1)
            
            tl_x = self.soft_argmax1d(feat_x1,input_soft=True) # (B,11)
            tl_y = self.soft_argmax1d(feat_y1,input_soft=True)
            br_x = self.soft_argmax1d(feat_x2,input_soft=True)
            br_y = self.soft_argmax1d(feat_y2,input_soft=True)
        else:
            if self.xy_pooling_type == 'avg':
                feat_x = torch.mean(x,dim=-2)
                feat_y = torch.mean(x,dim=-1)
            elif self.xy_pooling_type == 'max':
                feat_x = torch.max(x,dim=-2)[0]
                feat_y = torch.max(x,dim=-1)[0]
            heat_x1 = self.conv_x1(feat_x)
            heat_y1 = self.conv_y1(feat_y)
            heat_x2 = self.conv_x2(feat_x)
            heat_y2 = self.conv_y2(feat_y)
            tl_x = self.soft_argmax1d(heat_x1) # (B,11)
            tl_y = self.soft_argmax1d(heat_y1)
            br_x = self.soft_argmax1d(heat_x2)
            br_y = self.soft_argmax1d(heat_y2)
        bbox = torch.stack([tl_x,tl_y,br_x,br_y],dim=-1) #(B,11,4)
        return bbox
    
    def soft_argmax1d(self, score_vec, normalize=True, input_soft=False):
        (b,c,l) = score_vec.shape
        with torch.no_grad():
            coords = torch.arange(0,l).float().to(score_vec.device)
        """ get soft-argmax coordinate for a given heatmap """
        if input_soft:
            prob_vec  = score_vec
        else:
            prob_vec = F.softmax(score_vec, dim=2)
        exp_l = torch.sum((coords * prob_vec), dim=2)
        if normalize:
            if self.norm_align:
                exp_l = exp_l / (l-1)
            else:
                exp_l = exp_l / l
        return exp_l
    def _init_weight(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, BN=False):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        if BN:
            self.layers = nn.ModuleList(nn.Sequential(nn.Linear(n, k), nn.BatchNorm1d(k))
                                        for n, k in zip([input_dim] + h, h + [output_dim]))
        else:
            self.layers = nn.ModuleList(nn.Linear(n, k)
                                        for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def build_box_head(cfg):
    if cfg.MODEL.HEAD_TYPE == "MLP":
        hidden_dim = cfg.MODEL.HIDDEN_DIM
        mlp_head = MLP(hidden_dim, hidden_dim, 4, 3)  # dim_in, dim_hidden, dim_out, 3 layers
        return mlp_head
    elif "CORNER" in cfg.MODEL.HEAD_TYPE:
        # if cfg.MODEL.BACKBONE.DILATION is False:
        #     stride = 16
        # else:
        #     stride = 8
        stride = 16
        feat_sz = int(cfg.DATA.SEARCH.SIZE / stride)
        # channel = getattr(cfg.MODEL, "HEAD_DIM", 256)
        channel = getattr(cfg.MODEL, "HEAD_DIM", 384)
        freeze_bn = getattr(cfg.MODEL, "HEAD_FREEZE_BN", False)
        print("head channel: %d" % channel)
        if cfg.MODEL.HEAD_TYPE == "CORNER":
            corner_head = Corner_Predictor(inplanes=cfg.MODEL.HIDDEN_DIM, channel=channel,
                                           feat_sz=feat_sz, stride=stride, freeze_bn=freeze_bn)
        else:
            raise ValueError()
        return corner_head
    else:
        raise ValueError("HEAD TYPE %s is not supported." % cfg.MODEL.HEAD_TYPE)