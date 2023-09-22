import torch
import torch.nn as nn
from networks.encoders import build_encoder
from networks.layers.transformer import LongShortTermTransformer
from networks.decoders import build_decoder
from networks.layers.position import PositionEmbeddingSine
from networks.layers.transformer import BoxTransformer, PatchEmbed
from networks.layers.vision_transformer import VisionTransformer
from networks.layers.box_head import Corner_Predictor,Center_Predictor,PinpointHead, FPNBoxHead

class AOT(nn.Module):
    def __init__(self, cfg, encoder='mobilenetv2', decoder='fpn'):
        super().__init__()
        self.cfg = cfg
        self.max_obj_num = cfg.MODEL_MAX_OBJ_NUM
        self.epsilon = cfg.MODEL_EPSILON

        self.encoder = build_encoder(encoder,
                                     frozen_bn=cfg.MODEL_FREEZE_BN,
                                     freeze_at=cfg.TRAIN_ENCODER_FREEZE_AT)
        self.encoder_projector = nn.Conv2d(cfg.MODEL_ENCODER_DIM[-1],
                                           cfg.MODEL_ENCODER_EMBEDDING_DIM,
                                           kernel_size=1)
        self.LSTT = LongShortTermTransformer(
            cfg.MODEL_LSTT_NUM,
            cfg.MODEL_ENCODER_EMBEDDING_DIM,
            cfg.MODEL_SELF_HEADS,
            cfg.MODEL_ATT_HEADS,
            emb_dropout=cfg.TRAIN_LSTT_EMB_DROPOUT,
            droppath=cfg.TRAIN_LSTT_DROPPATH,
            lt_dropout=cfg.TRAIN_LSTT_LT_DROPOUT,
            st_dropout=cfg.TRAIN_LSTT_ST_DROPOUT,
            droppath_lst=cfg.TRAIN_LSTT_DROPPATH_LST,
            droppath_scaling=cfg.TRAIN_LSTT_DROPPATH_SCALING,
            intermediate_norm=cfg.MODEL_DECODER_INTERMEDIATE_LSTT,
            return_intermediate=True,
            block_version="v1")

        decoder_indim = cfg.MODEL_ENCODER_EMBEDDING_DIM * \
            (cfg.MODEL_LSTT_NUM +
             1) if cfg.MODEL_DECODER_INTERMEDIATE_LSTT else cfg.MODEL_ENCODER_EMBEDDING_DIM

        self.decoder = build_decoder(
            decoder,
            in_dim=decoder_indim,
            out_dim=cfg.MODEL_MAX_OBJ_NUM + 1,
            decode_intermediate_input=cfg.MODEL_DECODER_INTERMEDIATE_LSTT,
            hidden_dim=cfg.MODEL_ENCODER_EMBEDDING_DIM,
            shortcut_dims=cfg.MODEL_ENCODER_DIM,
            align_corners=cfg.MODEL_ALIGN_CORNERS)

        if cfg.MODEL_ALIGN_CORNERS:
            self.patch_wise_id_bank = nn.Conv2d(
                cfg.MODEL_MAX_OBJ_NUM + 1,
                cfg.MODEL_ENCODER_EMBEDDING_DIM,
                kernel_size=17,
                stride=16,
                padding=8)
        else:
            self.patch_wise_id_bank = nn.Conv2d(
                cfg.MODEL_MAX_OBJ_NUM + 1,
                cfg.MODEL_ENCODER_EMBEDDING_DIM,
                kernel_size=16,
                stride=16,
                padding=0)

        self.id_dropout = nn.Dropout(cfg.TRAIN_LSTT_ID_DROPOUT, True)

        self.pos_generator = PositionEmbeddingSine(
            cfg.MODEL_ENCODER_EMBEDDING_DIM // 2, normalize=True)

        # box transformer
        if self.cfg.MODEL_USE_BOXT or self.cfg.MODEL_USE_BOXT_COND:
            if self.cfg.MODEL_BOXT_PATCH_EMB == 'conv':
                self.BoxT_patch = PatchEmbed(patch_size=cfg.TRAIN_BOXT_PATCH_SIZE,
                                                in_chans=3 + cfg.MODEL_MAX_OBJ_NUM + 1,embed_dim=cfg.MODEL_BOXT_ENCODER_DIM)
            elif 'box_id_encoder' in self.cfg.MODEL_BOXT_PATCH_EMB:
                self.BoxT_patch = build_encoder(cfg.MODEL_BOX_ID_ENCODER,
                                        frozen_bn=cfg.MODEL_BOX_ID_ENCODER_FROZEN_BN,
                                        freeze_at=cfg.MODEL_BOX_ID_ENCODER_FREEZE_AT,
                                        in_channel=3 + cfg.MODEL_MAX_OBJ_NUM + 1 if self.cfg.MODEL_USE_BOXT else 3,
                                        use_ln=cfg.MODEL_BOX_ID_ENCODER_USE_LN)
                self.BoxT_patch_adaptor = nn.Conv2d(cfg.MODEL_ENCODER_EMBEDDING_DIM,cfg.MODEL_BOXT_ENCODER_DIM,kernel_size=1)
            else:
                raise NotImplementedError
            self.BoxT_pos = PositionEmbeddingSine(cfg.MODEL_BOXT_ENCODER_DIM//2,normalize=True)
            if self.cfg.MODEL_BOXT_ENCODER == 'default':
                self.BoxT = BoxTransformer(
                    cfg.MODEL_BOXT_NUM,
                    cfg.MODEL_BOXT_ENCODER_DIM,
                    cfg.MODEL_SELF_HEADS,
                    cfg.MODEL_ATT_HEADS,
                    dual_cross=cfg.MODEL_BOXT_DUAL_CROSS,
                    droppath=cfg.TRAIN_BOXT_DROPPATH,
                    droppath_cross=cfg.TRAIN_BOXT_DROPPATH_CROSS,
                )
            elif self.cfg.MODEL_BOXT_ENCODER == 'ViT-Tiny':
                self.BoxT = VisionTransformer(
                                    # global_pool='',
                                    embed_dim=cfg.MODEL_BOXT_ENCODER_DIM,
                                    depth=12,
                                    num_heads=3,
                                    # class_token=False,
                                    # no_embed_class=True,
                                    drop_path_rate=cfg.TRAIN_BOXT_DROPPATH,
                            )
            elif self.cfg.MODEL_BOXT_ENCODER == 'ViT-Small':
                self.BoxT = VisionTransformer(
                                    # global_pool='',
                                    embed_dim=cfg.MODEL_BOXT_ENCODER_DIM,
                                    depth=12,
                                    num_heads=6,
                                    # class_token=False,
                                    # no_embed_class=True,
                                    drop_path_rate=cfg.TRAIN_BOXT_DROPPATH,
                            )
            if cfg.MODEL_BOXT_DECODER == 'deconv':
                if cfg.MODEL_ALIGN_CORNERS:
                    self.BoxT_decoder = nn.ConvTranspose2d(
                        cfg.MODEL_BOXT_ENCODER_DIM,
                        cfg.MODEL_MAX_OBJ_NUM + 1,
                        kernel_size=17,
                        stride=16,
                        padding=8)
                else:
                    self.BoxT_decoder = nn.ConvTranspose2d(
                        cfg.MODEL_BOXT_ENCODER_DIM,
                        cfg.MODEL_MAX_OBJ_NUM + 1,
                        kernel_size=16,
                        stride=16,
                        padding=0)
            elif cfg.MODEL_BOXT_DECODER == 'fpn':
                self.BoxT_decoder = build_decoder(
                    'fpn',
                    in_dim=cfg.MODEL_BOXT_ENCODER_DIM,
                    out_dim=cfg.MODEL_MAX_OBJ_NUM + 1,
                    decode_intermediate_input=False,
                    hidden_dim=cfg.MODEL_MS_ENCODER_EMBEDDING_DIMS,
                    shortcut_dims=cfg.MODEL_ENCODER_DIM,
                    use_shortcuts=cfg.MODEL_BOXT_SHORTCUTS,
                    align_corners=cfg.MODEL_ALIGN_CORNERS)
            else:
                raise NotImplementedError
        if cfg.MODEL_BOXT_EMB_ID:
            self.id_conv = nn.Conv2d(cfg.MODEL_ENCODER_EMBEDDING_DIM,cfg.MODEL_ENCODER_EMBEDDING_DIM,1)
        self.id_norm = nn.LayerNorm(cfg.MODEL_ENCODER_EMBEDDING_DIM)
        
        if cfg.MODEL_BOX_HEAD_IN == 'feature':
            inplanes = cfg.MODEL_ENCODER_EMBEDDING_DIM
            outplanes = cfg.MODEL_MAX_OBJ_NUM + 1
        elif cfg.MODEL_BOX_HEAD_IN == 'imgmask':
            inplanes = 4
            outplanes = 1
        elif cfg.MODEL_BOX_HEAD_IN == 'intermediate_features':
            inplanes=cfg.MODEL_ENCODER_EMBEDDING_DIM * cfg.MODEL_LSTT_NUM
            outplanes=cfg.MODEL_MAX_OBJ_NUM + 1
        elif cfg.MODEL_BOX_HEAD_IN == 'enc_lstt_features':
            inplanes=cfg.MODEL_ENCODER_EMBEDDING_DIM*2
            outplanes=cfg.MODEL_MAX_OBJ_NUM + 1
        elif cfg.MODEL_BOX_HEAD_IN == 'enc_lstt_mask_features':
            inplanes=cfg.MODEL_ENCODER_EMBEDDING_DIM*3
            outplanes=cfg.MODEL_MAX_OBJ_NUM + 1
            if cfg.MODEL_ALIGN_CORNERS:
                self.box_head_mask_conv = nn.Conv2d(
                    cfg.MODEL_MAX_OBJ_NUM + 1,
                    cfg.MODEL_ENCODER_EMBEDDING_DIM,
                    kernel_size=17,
                    stride=16,
                    padding=8)
            else:
                self.box_head_mask_conv = nn.Conv2d(
                    cfg.MODEL_MAX_OBJ_NUM + 1,
                    cfg.MODEL_ENCODER_EMBEDDING_DIM,
                    kernel_size=16,
                    stride=16,
                    padding=0)
        
        if cfg.MODEL_BOX_HEAD == None:
            pass
        elif cfg.MODEL_BOX_HEAD == 'corner':
            self.box_head = Corner_Predictor(inplanes=inplanes, channel=cfg.MODEL_BOX_HEAD_DIM,
                                            outplanes=outplanes,downsample=cfg.MODEL_BOX_HEAD_DOWNSAMPLE,
                                            align_corners=cfg.MODEL_ALIGN_CORNERS)
        elif cfg.MODEL_BOX_HEAD == 'center':
            self.box_head = Center_Predictor(inplanes=inplanes, 
                                            channel=cfg.MODEL_BOX_HEAD_DIM,
                                            outplanes=outplanes)
        elif cfg.MODEL_BOX_HEAD == 'pinpoint':
            self.box_head = PinpointHead(inplanes=inplanes,outplanes=outplanes,
                                            d_model=cfg.MODEL_BOX_HEAD_DIM,
                                            num_layers=cfg.MODEL_BOX_HEAD_LAYER,
                                            softmax_ctr=cfg.MODEL_BOX_HEAD_SOFTMAX,
                                            droppath=cfg.MODEL_BOX_HEAD_DROPPATH,
                                            sep_token=cfg.MODEL_BOX_HEAD_SEP_TOKEN,
                                            pos_type=cfg.MODEL_BOX_HEAD_POS_TYPE,
                                            xy_pooling=cfg.MODEL_BOX_HEAD_XY_POOLING,
                                            xy_pooling_type=cfg.MODEL_BOX_HEAD_XY_POOLING_TYPE,
                                            preconv=cfg.MODEL_BOX_HEAD_POOLING_PRECONV,
                                            reverse_pooling=cfg.MODEL_BOX_HEAD_REVERSE_POOLING,
                                            norm_align=cfg.MODEL_BOX_HEAD_NORM_ALIGN)
        elif cfg.MODEL_BOX_HEAD == 'fpn':
            self.box_head = FPNBoxHead(in_dim=inplanes,out_dim=outplanes,
                                        hidden_dim=cfg.MODEL_ENCODER_EMBEDDING_DIM,
                                        shortcut_dims=cfg.MODEL_ENCODER_DIM,
                                        align_corners=cfg.MODEL_ALIGN_CORNERS,
                                        out_scale=cfg.MODEL_BOX_HEAD_SCALE,
                                        xy_pooling_type=cfg.MODEL_BOX_HEAD_XY_POOLING_TYPE)
        else:
            raise NotImplementedError
        
        
        self._init_weight()

    def get_pos_emb(self, x):
        pos_emb = self.pos_generator(x)
        return pos_emb

    def get_id_emb(self, x):
        id_emb = self.patch_wise_id_bank(x)
        id_emb = self.id_dropout(id_emb)
        return id_emb

    def encode_image(self, img):
        xs = self.encoder(img)
        xs[-1] = self.encoder_projector(xs[-1])
        return xs

    def decode_id_logits(self, lstt_emb, shortcuts):
        n, c, h, w = shortcuts[-1].size()
        decoder_inputs = [shortcuts[-1]]
        for emb in lstt_emb:
            decoder_inputs.append(emb.view(h, w, n, c).permute(2, 3, 0, 1))
        pred_logit = self.decoder(decoder_inputs, shortcuts)
        return pred_logit

    def LSTT_forward(self,
                     curr_embs,
                     long_term_memories,
                     short_term_memories,
                     curr_id_emb=None,
                     pos_emb=None,
                     size_2d=(30, 30)):
        n, c, h, w = curr_embs[-1].size()
        curr_emb = curr_embs[-1].view(n, c, h * w).permute(2, 0, 1)
        lstt_embs, lstt_memories = self.LSTT(curr_emb, long_term_memories,
                                             short_term_memories, curr_id_emb,
                                             pos_emb, size_2d)
        lstt_curr_memories, lstt_long_memories, lstt_short_memories = zip(
            *lstt_memories)
        return lstt_embs, lstt_curr_memories, lstt_long_memories, lstt_short_memories

    def _init_weight(self):
        nn.init.xavier_uniform_(self.encoder_projector.weight)
        nn.init.orthogonal_(
            self.patch_wise_id_bank.weight.view(
                self.cfg.MODEL_ENCODER_EMBEDDING_DIM, -1).permute(0, 1),
            gain=17**-2 if self.cfg.MODEL_ALIGN_CORNERS else 16**-2)
        
    def get_box_pos_emb(self,x):
        return self.BoxT_pos(x)

    # def get_id_emb(self, x):
    #     id_emb = self.patch_wise_id_bank(x)
    #     id_emb = self.id_norm(id_emb.permute(2, 3, 0, 1)).permute(2, 3, 0, 1)
    #     id_emb = self.id_dropout(id_emb)
        return id_emb
    def get_id_from_emb(self,emb):
        id_emb = self.id_conv(emb)
        id_emb = self.id_norm(id_emb.permute(2, 3, 0, 1)).permute(2, 3, 0, 1)
        id_emb = self.id_dropout(id_emb)
        return id_emb
    
    def get_patch_emb(self,img):
        if self.cfg.MODEL_BOXT_PATCH_EMB == 'conv':
            return self.BoxT_patch(img)
        elif 'box_id_encoder' in self.cfg.MODEL_BOXT_PATCH_EMB:
            xs = self.BoxT_patch(img)
            return self.BoxT_patch_adaptor(xs[-1])
        else:
            raise NotImplementedError
    
    def box_transformer(self,img_emb,boxes_emb):
        if self.cfg.MODEL_BOXT_ENCODER == 'default':
            res = self.BoxT(img_emb,boxes_emb)
        elif 'ViT' in self.cfg.MODEL_BOXT_ENCODER:
            if self.cfg.MODEL_BOXT_BOXIN:
                input_emb = torch.cat([img_emb,boxes_emb],dim=0)
                res = self.BoxT.forward_features(input_emb,patch_input=True)
                res = res[:img_emb.shape[0]]
            else:
                res = self.BoxT.forward_features(img_emb,patch_input=True)
        return res

    def box_transformer_decoder(self,id_emb_2d,shortcuts=None):
        
        if self.cfg.MODEL_BOXT_DECODER == 'fpn':
            outputs = self.BoxT_decoder(id_emb_2d,shortcuts,return_xs=True)
            id_embs = outputs[:-1]
            id_logits = outputs[-1]
        else:
            raise NotImplementedError
        return id_embs,id_logits
    
    def decode_boxes(self,feature_in,shortcuts=None):
        if self.cfg.MODEL_BOX_HEAD == 'fpn':
            output = self.box_head(feature_in,shortcuts)
        else:
            output = self.box_head(feature_in, return_med=self.cfg.MODEL_DECODE_MASK_WITH_BOX)
        return output
