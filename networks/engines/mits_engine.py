import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from utils.math import generate_permute_matrix
from utils.image import one_hot_mask,boxize_mask,corner_grid_sample
from utils.heatmap import generate_heatmap

from networks.layers.basic import seq_to_2d
from networks.layers.loss import dice_loss,GeneralizedBoxIoULoss,FocalLoss

class MITSEngine(nn.Module):
    def __init__(self,
                 aot_model,
                 gpu_id=0,
                 long_term_mem_gap=9999,
                 short_term_mem_skip=1):
        super().__init__()

        self.cfg = aot_model.cfg
        self.align_corners = aot_model.cfg.MODEL_ALIGN_CORNERS
        self.AOT = aot_model

        self.max_obj_num = aot_model.max_obj_num
        self.gpu_id = gpu_id
        self.long_term_mem_gap = long_term_mem_gap
        self.short_term_mem_skip = short_term_mem_skip
        self.losses = None
        self.is_box = []
        self.restart_engine()

    def forward(self,
                all_frames,
                all_masks,
                batch_size,
                obj_nums,
                step=0,
                tf_board=False,
                use_prev_pred=False,
                enable_prev_frame=False,
                use_prev_prob=False,
                meta=None):  # only used for training
        self.boards = {'image': {}, 'scalar': {}}
        self.use_prev_pred = use_prev_pred
        if self.losses is None:
            self._init_losses()
        if meta is not None and 'is_box' in meta.keys():
            self.is_box = meta['is_box']
        self.freeze_id = True if use_prev_pred else False
        aux_weight = self.aux_weight * max(self.aux_step - step,
                                           0.) / self.aux_step

        self.offline_encoder(all_frames, all_masks)

        self.add_reference_frame(frame_step=0, obj_nums=obj_nums)

        grad_state = torch.no_grad if aux_weight == 0 else torch.enable_grad
        with grad_state():
            ref_aux_loss, ref_aux_mask = self.generate_loss_mask(
                self.offline_masks[self.frame_step], step)

        aux_losses = [ref_aux_loss]
        aux_masks = [ref_aux_mask]

        curr_losses, curr_masks = [], []
        if enable_prev_frame:
            self.set_prev_frame(frame_step=1)
            with grad_state():
                prev_aux_loss, prev_aux_mask = self.generate_loss_mask(
                    self.offline_masks[self.frame_step], step)
            aux_losses.append(prev_aux_loss)
            aux_masks.append(prev_aux_mask)
        else:
            self.match_propogate_one_frame()
            curr_loss, curr_mask, curr_prob = self.generate_loss_mask(
                self.offline_masks[self.frame_step], step, return_prob=True)
            self.update_short_term_memory(
                curr_mask if not use_prev_prob else curr_prob,
                None if use_prev_pred else self.assign_identity(
                    self.offline_one_hot_masks[self.frame_step]))
            curr_losses.append(curr_loss)
            curr_masks.append(curr_mask)

        self.match_propogate_one_frame()
        curr_loss, curr_mask, curr_prob = self.generate_loss_mask(
            self.offline_masks[self.frame_step], step, return_prob=True)
        curr_losses.append(curr_loss)
        curr_masks.append(curr_mask)
        for _ in range(self.total_offline_frame_num - 3):
            self.update_short_term_memory(
                curr_mask if not use_prev_prob else curr_prob,
                None if use_prev_pred else self.assign_identity(
                    self.offline_one_hot_masks[self.frame_step]))
            self.match_propogate_one_frame()
            curr_loss, curr_mask, curr_prob = self.generate_loss_mask(
                self.offline_masks[self.frame_step], step, return_prob=True)
            curr_losses.append(curr_loss)
            curr_masks.append(curr_mask)

        aux_loss = torch.cat(aux_losses, dim=0).mean(dim=0)
        pred_loss = torch.cat(curr_losses, dim=0).mean(dim=0)

        loss = aux_weight * aux_loss + pred_loss
        if self.cfg.MODEL_USE_BOXT:
            box_loss = self.calculate_box_loss()
            loss = loss + box_loss

        all_pred_mask = aux_masks + curr_masks

        all_frame_loss = aux_losses + curr_losses

        return loss, all_pred_mask, all_frame_loss, self.boards

    def _init_losses(self):
        cfg = self.cfg

        from networks.layers.loss import CrossEntropyLoss, SoftJaccordLoss
        bce_loss = CrossEntropyLoss(
            cfg.TRAIN_TOP_K_PERCENT_PIXELS,
            cfg.TRAIN_HARD_MINING_RATIO * cfg.TRAIN_TOTAL_STEPS)
        iou_loss = SoftJaccordLoss()

        losses = [bce_loss, iou_loss]
        loss_weights = [0.5, 0.5]
        loss_weights = [cfg.TRAIN_BCE_LOSS_WEIGHT,cfg.TRAIN_IOU_LOSS_WEIGHT]
        self.losses = nn.ModuleList(losses)
        self.loss_weights = loss_weights
        self.aux_weight = cfg.TRAIN_AUX_LOSS_WEIGHT
        self.aux_step = cfg.TRAIN_TOTAL_STEPS * cfg.TRAIN_AUX_LOSS_RATIO + 1e-5

    def encode_one_img_mask(self, img=None, mask=None, frame_step=-1):
        if frame_step == -1:
            frame_step = self.frame_step

        if self.enable_offline_enc:
            curr_enc_embs = self.offline_enc_embs[frame_step]
        elif img is None:
            curr_enc_embs = None
        else:
            curr_enc_embs = self.AOT.encode_image(img)

        if mask is not None:
            curr_one_hot_mask = one_hot_mask(mask, self.max_obj_num)
        elif self.enable_offline_enc:
            curr_one_hot_mask = self.offline_one_hot_masks[frame_step]
        else:
            curr_one_hot_mask = None

        return curr_enc_embs, curr_one_hot_mask

    def offline_encoder(self, all_frames, all_masks=None):
        self.enable_offline_enc = True
        self.offline_frames = all_frames.size(0) // self.batch_size
        self.offline_imgs = all_frames

        # extract backbone features
        self.offline_enc_embs = self.split_frames(
            self.AOT.encode_image(all_frames), self.batch_size)
        self.total_offline_frame_num = len(self.offline_enc_embs)

        if all_masks is not None:
            # extract mask embeddings
            offline_one_hot_masks = one_hot_mask(all_masks, self.max_obj_num)
            self.offline_masks = list(
                torch.split(all_masks, self.batch_size, dim=0))
            self.offline_one_hot_masks = list(
                torch.split(offline_one_hot_masks, self.batch_size, dim=0))

        if self.input_size_2d is None:
            self.update_size(all_frames.size()[2:],
                             self.offline_enc_embs[0][-1].size()[2:])

    def assign_identity(self, one_hot_mask,reshape=True):
        if self.enable_id_shuffle:
            one_hot_mask = torch.einsum('bohw,bot->bthw', one_hot_mask,
                                        self.id_shuffle_matrix)
        if reshape:
            id_emb = self.AOT.get_id_emb(one_hot_mask).view(
                self.batch_size, -1, self.enc_hw).permute(2, 0, 1)
        else:
            id_emb = self.AOT.get_id_emb(one_hot_mask)

        if self.training and self.freeze_id:
            id_emb = id_emb.detach()

        return id_emb
    
    def id_box2mask(self, img, one_hot_mask, return_logits=False):
        # Note object number could be 0!
        # 1, get image and boxed image feature
        h_box,w_box = self.cfg.TRAIN_BOX_SIZE
        img_box_one_hot_mask,corners_list = boxize_mask(one_hot_mask,return_corners=True)

        # get gt id embs (shuffled in assign_identity)
        gt_mask_id_emb = self.assign_identity(one_hot_mask,reshape=False).detach()
        if self.cfg.MODEL_BOXT_ID_DETACH:
            img_box_id_emb = self.assign_identity(img_box_one_hot_mask,reshape=False).detach()
        else:
            img_box_id_emb = self.assign_identity(img_box_one_hot_mask,reshape=False)

        # shuffle one hot mask!
        if self.enable_id_shuffle:
            one_hot_mask = torch.einsum('bohw,bot->bthw', one_hot_mask,
                                        self.id_shuffle_matrix)
            img_box_one_hot_mask = torch.einsum('bohw,bot->bthw', img_box_one_hot_mask,
                                        self.id_shuffle_matrix)

        img_x = self.AOT.get_patch_emb(torch.cat([img,img_box_one_hot_mask],dim=1))
        img_boxes_list = [] # [B,(Nb,3,ho,wo)] with standardized box size: ho,wo
        for b in range(self.batch_size):
            if self.obj_nums[b] == 0:
                img_boxes_list.append([])
                continue
            img_boxes = []
            if self.obj_nums[b] != len(corners_list[b]):
                print(self.obj_nums,b,corners_list)
            # for i in range(self.obj_nums[b]):
            for i in range(len(corners_list[b])):
                # print(b,i,corners_list)
                [x1,y1,x2,y2] = list(map(int,corners_list[b][i]))
                img_box = img[b,:,y1:y2+1,x1:x2+1]
                img_box = F.interpolate(img_box.unsqueeze(0),(h_box,w_box),mode='bilinear',align_corners=self.align_corners)
                label_box = img_box_one_hot_mask[b,:,y1:y2+1,x1:x2+1]
                label_box = F.interpolate(label_box.unsqueeze(0),(h_box,w_box),mode='nearest')
                img_boxes.append(torch.cat([img_box,label_box],dim=1)) # (1,3+11,ho,wo)
            img_boxes_list.append(torch.cat(img_boxes,dim=0)) # (Nb,3,ho,wo)
        boxes_x_list = [] # [B,(Nb,C,hoo,woo)]
        for b in range(self.batch_size):
            if self.obj_nums[b]>0:
                boxes_x_list.append(self.AOT.get_patch_emb(img_boxes_list[b]))
                self.box_enc_size_2d = boxes_x_list[b].shape[2],boxes_x_list[b].shape[3]
                self.box_enc_hw = boxes_x_list[b].shape[2] * boxes_x_list[b].shape[3]
            else:
                boxes_x_list.append(None)
        
        # 3, get pos embed
        img_pos_emb = self.AOT.get_box_pos_emb(img_x) #(1,C,hmm,wmm)
        boxes_pos_emb_list = [] # [B,(Nb,C,hoo,woo)]
        for b in range(self.batch_size):
            if self.obj_nums[b] == 0:
                boxes_pos_emb_list.append([])
                continue
            boxes_pos_emb = []
            # for i in range(self.obj_nums[b]):
            for i in range(len(corners_list[b])):
                corner = corners_list[b][i]
                # normalize box corner to [0,1]
                corner[0] = corner[0]/img.shape[2]
                corner[2] = corner[2]/img.shape[2]
                corner[1] = corner[1]/img.shape[3]
                corner[3] = corner[3]/img.shape[3]
                boxes_pos_emb.append(corner_grid_sample(img_pos_emb,corner,self.box_enc_size_2d))
            boxes_pos_emb_list.append(torch.cat(boxes_pos_emb,0))
        
        # 4, cross attention and mask loss
        box_mask_loss_fun = torch.nn.CrossEntropyLoss()
        self.box_id_losses = []
        self.box_mask_losses = []
        pred_id_emb_list = []
        self.box_masks = []

        # feature: boxes_x_list
        # id: boxes_id_emb_list
        # pos: boxes_pos_emb_list
        for b in range(self.batch_size):
            if self.obj_nums[b] == 0:
                pred_id_emb_list.append(gt_mask_id_emb[b].unsqueeze(0))
                self.box_masks.append(torch.argmax(one_hot_mask[b].unsqueeze(0),dim=1))
                continue
            # prepare input
            img_emb = img_x[b]+img_pos_emb[0] # (C,hmm,wmm)
            C_dim = img_emb.shape[0]
            img_emb = img_emb.permute(1,2,0).reshape(-1,C_dim).unsqueeze(1) # (hmm*wmm,1,C)
            
            boxes_emb = boxes_x_list[b]+boxes_pos_emb_list[b] # (Nb,C,hoo,woo)
            boxes_emb = boxes_emb.permute(0,2,3,1).reshape(-1,C_dim).unsqueeze(1) # (Nb*hoo*woo,1,C)
            
            # predict mask id embedding
            img_emb_t = self.AOT.box_transformer(img_emb,boxes_emb)
            (_,bs,c) = img_emb_t.shape
            
            img_emb_t_2d = img_emb_t.permute(1,2,0).reshape(bs,c,*self.enc_size_2d)
            shortcuts = [x[b].unsqueeze(0) for x in self.curr_enc_embs]
            if self.cfg.MODEL_BOXT_PRE_ID_SKIP:
                img_box_id_emb_b = img_box_id_emb[b].unsqueeze(0)
                img_emb_t_2d = img_emb_t_2d+img_box_id_emb_b
            if self.cfg.MODEL_BOXT_EMB_ID:
                pred_mask_id_emb_b = self.AOT.get_id_from_emb(img_emb_t_2d)
                _,pred_box_logits = self.AOT.box_transformer_decoder(img_emb_t_2d, shortcuts)
            else:
                pred_mask_id_embs_b,pred_box_logits = self.AOT.box_transformer_decoder(img_emb_t_2d, shortcuts)
                pred_mask_id_emb_b = pred_mask_id_embs_b[0]
            if self.cfg.MODEL_BOXT_ID_SKIP and not self.cfg.MODEL_BOXT_PRE_ID_SKIP:
                img_box_id_emb_b = img_box_id_emb[b].unsqueeze(0)
                pred_mask_id_emb_b = pred_mask_id_emb_b+img_box_id_emb_b
            pred_id_emb_list.append(pred_mask_id_emb_b)
            if self.training and not self.is_box[b]:
                # predict mask
                self.box_masks.append(torch.argmax(pred_box_logits,dim=1))
                pred_box_logits = F.interpolate(pred_box_logits,
                                           size=(one_hot_mask.shape[-2],one_hot_mask.shape[-1]),
                                           mode="bilinear",
                                           align_corners=self.align_corners)
                mask_loss = box_mask_loss_fun(pred_box_logits,one_hot_mask[b].unsqueeze(0))
                if self.cfg.TRAIN_BOXT_MASK_PRJ_LOSS_WEIGHT > 0.:
                    prj_loss = self.cfg.TRAIN_BOXT_MASK_PRJ_LOSS_WEIGHT * \
                                (dice_loss(torch.max(F.softmax(pred_box_logits,dim=1),dim=2,keepdim=True)[0],
                                            torch.max(one_hot_mask[b].unsqueeze(0),dim=2,keepdim=True)[0]) \
                                + dice_loss(torch.max(F.softmax(pred_box_logits,dim=1),dim=3,keepdim=True)[0],
                                            torch.max(one_hot_mask[b].unsqueeze(0),dim=3,keepdim=True)[0]))
                    mask_loss += prj_loss
                self.box_mask_losses.append(mask_loss)
        
        # id reshape and freeze
        pred_mask_id_emb = torch.cat(pred_id_emb_list,dim=0)
        bs,c,_,_ = pred_mask_id_emb.shape
        pred_mask_id_emb = pred_mask_id_emb.view(bs,c,-1).permute(2,0,1)
        if self.training and self.freeze_id:
            pred_mask_id_emb = pred_mask_id_emb.detach()
        if return_logits:
            return pred_mask_id_emb,pred_box_logits
        else:
            return pred_mask_id_emb

    def split_frames(self, xs, chunk_size):
        new_xs = []
        for x in xs:
            all_x = list(torch.split(x, chunk_size, dim=0))
            new_xs.append(all_x)
        return list(zip(*new_xs))

    def add_reference_frame(self,
                            img=None,
                            mask=None,
                            frame_step=-1,
                            obj_nums=None,
                            img_embs=None):
        if self.obj_nums is None and obj_nums is None:
            print('No objects for reference frame!')
            exit()
        elif obj_nums is not None:
            self.obj_nums = obj_nums

        if frame_step == -1:
            frame_step = self.frame_step

        if img_embs is None:
            curr_enc_embs, curr_one_hot_mask = self.encode_one_img_mask(
                img, mask, frame_step)
        else:
            _, curr_one_hot_mask = self.encode_one_img_mask(
                None, mask, frame_step)
            curr_enc_embs = img_embs

        if curr_enc_embs is None:
            print('No image for reference frame!')
            exit()

        if curr_one_hot_mask is None:
            print('No mask for reference frame!')
            exit()

        if self.input_size_2d is None:
            self.update_size(img.size()[2:], curr_enc_embs[-1].size()[2:])
        
        if img is None:
            img = self.offline_imgs[0::self.offline_frames]

        self.curr_enc_embs = curr_enc_embs
        # curr_one_hot_mask,corners_list = boxize_mask(curr_one_hot_mask,return_corners=True)
        self.curr_one_hot_mask = curr_one_hot_mask

        if self.pos_emb is None:
            self.pos_emb = self.AOT.get_pos_emb(curr_enc_embs[-1]).expand(
                self.batch_size, -1, -1,
                -1).view(self.batch_size, -1, self.enc_hw).permute(2, 0, 1)

        # Determine whether box or mask
        if self.training:
            if self.cfg.MODEL_USE_BOXT or self.cfg.MODEL_USE_BOXT_COND:
                prob = 0.0 if self.cfg.TRAIN_BOXT_ONLY else torch.rand(1)[0]
                if prob < self.cfg.TRAIN_BOXT_PROB:
                    curr_id_embs = self.id_box2mask(img,curr_one_hot_mask)
                    if self.cfg.TRAIN_BOXT_MASK_ONLY:
                        curr_id_embs = self.assign_identity(curr_one_hot_mask)
                else:
                    curr_id_embs = self.assign_identity(curr_one_hot_mask)
            else:
                curr_id_embs = self.assign_identity(curr_one_hot_mask)
        else:
            if self.cfg.TEST_BOXT:
                curr_id_embs = self.id_box2mask(img,curr_one_hot_mask)
            else:
                curr_id_embs = self.assign_identity(curr_one_hot_mask)
        self.curr_id_embs = curr_id_embs

        # self matching and propagation
        self.curr_lstt_output = self.AOT.LSTT_forward(curr_enc_embs,
                                                      None,
                                                      None,
                                                      curr_id_embs,
                                                      pos_emb=self.pos_emb,
                                                      size_2d=self.enc_size_2d)

        lstt_embs, lstt_curr_memories, lstt_long_memories, lstt_short_memories = self.curr_lstt_output

        if self.long_term_memories is None:
            self.long_term_memories = lstt_long_memories
        else:
            self.update_long_term_memory(lstt_long_memories,is_ref=True)
        self.ref_frame_num += 1
        self.last_long_step = frame_step

        self.short_term_memories_list = [lstt_short_memories]
        self.short_term_memories = lstt_short_memories
        self.last_short_step = frame_step

    def set_prev_frame(self, img=None, mask=None, frame_step=1):
        self.frame_step = frame_step
        curr_enc_embs, curr_one_hot_mask = self.encode_one_img_mask(
            img, mask, frame_step)

        if curr_enc_embs is None:
            print('No image for previous frame!')
            exit()

        if curr_one_hot_mask is None:
            print('No mask for previous frame!')
            exit()

        self.curr_enc_embs = curr_enc_embs
        self.curr_one_hot_mask = curr_one_hot_mask

        curr_id_emb = self.assign_identity(curr_one_hot_mask)
        self.curr_id_embs = curr_id_emb

        # self matching and propagation
        self.curr_lstt_output = self.AOT.LSTT_forward(curr_enc_embs,
                                                      None,
                                                      None,
                                                      curr_id_emb,
                                                      pos_emb=self.pos_emb,
                                                      size_2d=self.enc_size_2d)

        lstt_embs, lstt_curr_memories, lstt_long_memories, lstt_short_memories = self.curr_lstt_output

        if self.long_term_memories is None:
            self.long_term_memories = lstt_long_memories
        else:
            self.update_long_term_memory(lstt_long_memories)
        self.last_long_step = frame_step

        self.short_term_memories_list = [lstt_short_memories]
        self.short_term_memories = lstt_short_memories

    def update_long_term_memory(self, new_long_term_memories, is_ref=False):
        if self.long_term_memories is None:
            self.long_term_memories = new_long_term_memories
        updated_long_term_memories = []
        for new_long_term_memory, last_long_term_memory in zip(
                new_long_term_memories, self.long_term_memories):
            updated_e = []
            for new_e, last_e in zip(new_long_term_memory,
                                     last_long_term_memory):
                if not self.training:
                    if new_e is None or last_e is None:
                        updated_e.append(None)
                    else:
                        e_len = new_e.shape[0]
                        e_num = last_e.shape[0] // e_len
                        max_num = self.cfg.TEST_LONG_TERM_MEM_MAX
                        if max_num <= e_num:
                            last_e = torch.cat([last_e[:e_len*(max_num-(self.ref_frame_num+1))],
                                                last_e[-self.ref_frame_num*e_len:]],dim=0)
                        if is_ref:
                            updated_e.append(torch.cat([last_e,new_e], dim=0))
                        else:
                            updated_e.append(torch.cat([new_e, last_e], dim=0))
                else:
                    if new_e is None or last_e is None:
                        updated_e.append(None)
                    else:
                        updated_e.append(torch.cat([new_e, last_e], dim=0))
            updated_long_term_memories.append(updated_e)
        self.long_term_memories = updated_long_term_memories

    def update_short_term_memory(self, curr_mask, curr_id_emb=None, _update_long=True,_update_short=True):
        update_long = False
        update_short = False
        if self.frame_step - self.last_long_step >= self.long_term_mem_gap and _update_long:
            update_long  = True
        if self.cfg.MODEL_FIXED_SHORT_MEM:
            if self.frame_step - self.last_short_step >= self.short_term_mem_skip and _update_short:
                update_short = True
        else:
            update_short = _update_short
        if update_long or update_short:
            if curr_id_emb is None:
                if len(curr_mask.size()) == 3 or curr_mask.size()[0] == 1:
                    curr_one_hot_mask = one_hot_mask(curr_mask, self.max_obj_num)
                else:
                    curr_one_hot_mask = curr_mask
                curr_id_emb = self.assign_identity(curr_one_hot_mask)

            lstt_curr_memories = self.curr_lstt_output[1]
            lstt_curr_memories_2d = []
            for layer_idx in range(len(lstt_curr_memories)):
                curr_k, curr_v, curr_id_k, curr_id_v = lstt_curr_memories[
                    layer_idx]
                curr_id_k, curr_id_v = self.AOT.LSTT.layers[
                    layer_idx].fuse_key_value_id(curr_id_k, curr_id_v, curr_id_emb)
                lstt_curr_memories[layer_idx][2], lstt_curr_memories[layer_idx][
                    3] = curr_id_k, curr_id_v
                local_curr_id_k = seq_to_2d(
                    curr_id_k, self.enc_size_2d) if curr_id_k is not None else None
                local_curr_id_v = seq_to_2d(curr_id_v, self.enc_size_2d)
                lstt_curr_memories_2d.append([
                    seq_to_2d(curr_k, self.enc_size_2d),
                    seq_to_2d(curr_v, self.enc_size_2d), local_curr_id_k,
                    local_curr_id_v
                ])

        if update_short:
            if self.cfg.MODEL_FIXED_SHORT_MEM:
                self.last_short_step = self.frame_step
                self.short_term_memories = lstt_curr_memories_2d
            else:
                self.short_term_memories_list.append(lstt_curr_memories_2d)
                self.short_term_memories_list = self.short_term_memories_list[
                    -self.short_term_mem_skip:]
                self.short_term_memories = self.short_term_memories_list[0]

        if update_long:
            self.update_long_term_memory(lstt_curr_memories)
            self.last_long_step = self.frame_step

    def match_propogate_one_frame(self, img=None, img_embs=None):
        self.frame_step += 1
        if img_embs is None:
            curr_enc_embs, _ = self.encode_one_img_mask(
                img, None, self.frame_step)
        else:
            curr_enc_embs = img_embs
        self.curr_enc_embs = curr_enc_embs

        self.curr_lstt_output = self.AOT.LSTT_forward(curr_enc_embs,
                                                      self.long_term_memories,
                                                      self.short_term_memories,
                                                      None,
                                                      pos_emb=self.pos_emb,
                                                      size_2d=self.enc_size_2d)

    def decode_current_logits(self, output_size=None):
        curr_enc_embs = self.curr_enc_embs
        curr_lstt_embs = self.curr_lstt_output[0]
        if self.cfg.MODEL_DECODE_MASK_WITH_BOX!=None:
            b,c,h,w = curr_enc_embs[-1].shape
            x_map = self.x_feat.unsqueeze(-2).repeat(1,1,h,1)
            y_map = self.y_feat.unsqueeze(-1).repeat(1,1,1,w)
            xy_map = torch.cat([x_map,y_map],dim=1)
            pred_id_logits = self.AOT.decode_id_logits(curr_lstt_embs,curr_enc_embs,
                                                        ext_feat=xy_map)
        else:
            pred_id_logits = self.AOT.decode_id_logits(curr_lstt_embs,
                                                    curr_enc_embs)

        if self.enable_id_shuffle:  # reverse shuffle
            pred_id_logits = torch.einsum('bohw,bto->bthw', pred_id_logits,
                                          self.id_shuffle_matrix)

        # remove unused identities
        for batch_idx, obj_num in enumerate(self.obj_nums):
            pred_id_logits[batch_idx, (obj_num+1):] = - \
                1e+10 if pred_id_logits.dtype == torch.float32 else -1e+4

        self.pred_id_logits = pred_id_logits

        if output_size is not None:
            pred_id_logits = F.interpolate(pred_id_logits,
                                           size=output_size,
                                           mode="bilinear",
                                           align_corners=self.align_corners)

        return pred_id_logits

    def decode_current_boxes(self, output_size=None, img=None, mask=None):
        if output_size is None:
            output_size = self.input_size_2d
        if 'feature' in self.cfg.MODEL_BOX_HEAD_IN:
            if self.cfg.MODEL_BOX_HEAD_IN == 'feature':
                feature_in = self.curr_lstt_output[0][-1].view(*self.enc_size_2d,self.batch_size,-1).permute(2,3,0,1)
            elif self.cfg.MODEL_BOX_HEAD_IN == 'intermediate_features':
                feature_in = torch.cat(self.curr_lstt_output[0],-1)
                feature_in = feature_in.view(*self.enc_size_2d,self.batch_size,-1).permute(2,3,0,1)
            elif self.cfg.MODEL_BOX_HEAD_IN == 'enc_lstt_features':
                lstt_feature = self.curr_lstt_output[0][-1].view(*self.enc_size_2d,self.batch_size,-1).permute(2,3,0,1)
                enc_feature = self.curr_enc_embs[-1]
                feature_in = torch.cat([enc_feature,lstt_feature],dim=1)
            elif self.cfg.MODEL_BOX_HEAD_IN == 'enc_lstt_mask_features':
                lstt_feature = self.curr_lstt_output[0][-1].view(*self.enc_size_2d,self.batch_size,-1).permute(2,3,0,1)
                enc_feature = self.curr_enc_embs[-1]
                _one_hot_mask = one_hot_mask(mask, self.max_obj_num)
                if self.enable_id_shuffle:
                    _one_hot_mask = torch.einsum('bohw,bot->bthw', _one_hot_mask,
                                                self.id_shuffle_matrix)
                mask_feature = self.AOT.box_head_mask_conv(_one_hot_mask)
                feature_in = torch.cat([enc_feature,lstt_feature,mask_feature],dim=1)
            
            
            if self.cfg.MODEL_BOX_HEAD == 'corner':
                pred_boxes = self.AOT.decode_boxes(feature_in)
                if self.enable_id_shuffle:
                    pred_boxes = torch.einsum('bos,bto->bts',pred_boxes,self.id_shuffle_matrix)
            elif self.cfg.MODEL_BOX_HEAD == 'center':
                output = self.AOT.decode_boxes(feature_in)
                score_map_ctr,pred_boxes = output[0],output[1] # box (x1,y1,x2,y2)
                if self.enable_id_shuffle:  # reverse shuffle
                    score_map_ctr = torch.einsum('bohw,bto->bthw', score_map_ctr,
                                                self.id_shuffle_matrix)
                    pred_boxes = torch.einsum('bos,bto->bts',pred_boxes,self.id_shuffle_matrix)
                self.score_map_ctr = score_map_ctr
            elif self.cfg.MODEL_BOX_HEAD == 'pinpoint':
                output = self.AOT.decode_boxes(feature_in)
                score_map_ctr,pred_boxes,coord_ctr = output[0],output[1],output[2] # box (x1,y1,x2,y2)
                if self.cfg.MODEL_DECODE_MASK_WITH_BOX!=None:
                    self.x_feat,self.y_feat = output[3],output[4]
                if self.enable_id_shuffle:  # reverse shuffle
                    pred_boxes = torch.einsum('bos,bto->bts',pred_boxes,self.id_shuffle_matrix)
                    if not self.cfg.MODEL_BOX_HEAD_XY_POOLING:
                        score_map_ctr = torch.einsum('bohw,bto->bthw', score_map_ctr,
                                                    self.id_shuffle_matrix)
                        coord_ctr = torch.einsum('bos,bto->bts',coord_ctr,self.id_shuffle_matrix)
                self.score_map_ctr = score_map_ctr
                self.coord_ctr = coord_ctr
            elif self.cfg.MODEL_BOX_HEAD == 'fpn':
                pred_boxes = self.AOT.decode_boxes(feature_in, self.curr_enc_embs)
                if self.enable_id_shuffle:
                    pred_boxes = torch.einsum('bos,bto->bts',pred_boxes,self.id_shuffle_matrix)
            pred_boxes_list = []
            for batch_idx, obj_num in enumerate(self.obj_nums):
                boxes_b = []
                for i in range(obj_num):
                    boxes_b.append(pred_boxes[batch_idx,i+1])
                pred_boxes_list.append(boxes_b)
            self.pred_boxes_list = pred_boxes_list # b,n,4
        elif self.cfg.MODEL_BOX_HEAD_IN == 'imgmask':
            pred_boxes_list = []
            
            pred_size = self.pred_id_logits_list[0].shape[2:]
            if len(mask.shape)==3:
                mask = mask.unsqueeze(1) # b,h,w -> h,1,h,w
            mask = F.interpolate(mask.to(torch.float),size=pred_size,mode='nearest')
            if img is None:
                img = self.offline_imgs[self.frame_step::self.offline_frames]
            img = F.interpolate(img,size=pred_size,mode='bilinear',align_corners=self.align_corners)
            
            for batch_idx, obj_num in enumerate(self.obj_nums):
                boxes_b = []
                _img = img[batch_idx].unsqueeze(0)
                for i in range(1,obj_num+1):
                    _mask = (mask[batch_idx]==i).unsqueeze(0)
                    imgmask = torch.cat([_img,_mask],dim=1) # (1,4,h,w)
                    box = self.AOT.decode_box_from_imgmask(imgmask)
                    boxes_b.append(box.squeeze(0).squeeze(0))
                pred_boxes_list.append(boxes_b)
            self.pred_boxes_list = pred_boxes_list
        return pred_boxes_list

    def predict_current_mask(self, output_size=None, return_prob=False):
        if output_size is None:
            output_size = self.input_size_2d

        pred_id_logits = F.interpolate(self.pred_id_logits,
                                       size=output_size,
                                       mode="bilinear",
                                       align_corners=self.align_corners)
        pred_mask = torch.argmax(pred_id_logits, dim=1)

        if not return_prob:
            return pred_mask
        else:
            pred_prob = torch.softmax(pred_id_logits, dim=1)
            return pred_mask, pred_prob

    def calculate_current_loss(self, gt_mask, step):
        pred_id_logits = self.pred_id_logits

        pred_id_logits = F.interpolate(pred_id_logits,
                                       size=gt_mask.size()[-2:],
                                       mode="bilinear",
                                       align_corners=self.align_corners)

        label_list = []
        logit_list = []
        for batch_idx, obj_num in enumerate(self.obj_nums):
            if self.is_box[batch_idx]:
                continue
            now_label = gt_mask[batch_idx].long()
            now_logit = pred_id_logits[batch_idx, :(obj_num + 1)].unsqueeze(0)
            label_list.append(now_label.long())
            logit_list.append(now_logit)
        
        total_loss = 0
        if self.cfg.TRAIN_WITH_BOX:
            if label_list != []:
                for loss, loss_weight in zip(self.losses, self.loss_weights):
                    total_loss = total_loss + loss_weight * \
                        loss(logit_list, label_list, step)
                total_loss = torch.mean(total_loss,dim=0,keepdim=True)
        else:
            for loss, loss_weight in zip(self.losses, self.loss_weights):
                total_loss = total_loss + loss_weight * \
                    loss(logit_list, label_list, step)

        if self.cfg.TRAIN_BOX_LOSS:
            _,boxes = boxize_mask(one_hot_mask(gt_mask, self.max_obj_num),return_corners=True,norm_coord=True)
            gt_boxes = []
            pred_boxes = []
            for b in range(len(boxes)):
                boxes_b = boxes[b]
                for n in range(len(boxes_b)):
                    gt_box = torch.stack(boxes_b[n])
                    gt_boxes.append(gt_box.unsqueeze(0))
                    pred_boxes.append(self.pred_boxes_list[b][n].unsqueeze(0))
            if len(gt_boxes) > 0:
                gt_boxes_cat = torch.cat(gt_boxes,dim=0)
                pred_boxes_cat = torch.cat(pred_boxes,dim=0)

                w1 = self.cfg.TRAIN_BOX_L1_LOSS_WEIGHT
                w2 = self.cfg.TRAIN_BOX_GIOU_LOSS_WEIGHT
                box_loss_fun1 = nn.L1Loss()
                box_loss_fun2 = GeneralizedBoxIoULoss()
                box_l1_loss = box_loss_fun1(gt_boxes_cat,pred_boxes_cat)
                box_giou_loss,box_iou = box_loss_fun2(gt_boxes_cat,pred_boxes_cat)
                self.boards['scalar']['box_l1_loss'] = box_l1_loss
                self.boards['scalar']['box_giou_loss'] = box_giou_loss
                self.boards['scalar']['box_iou'] = box_iou.mean() if box_iou!=None else 0.
                total_loss = total_loss + w1*box_l1_loss + w2*box_giou_loss
                # print(box_l1_loss,box_giou_loss)

                if self.cfg.MODEL_BOX_HEAD == 'center' or self.cfg.MODEL_BOX_HEAD == 'pinpoint':
                    if self.cfg.TRAIN_BOX_FOCAL_LOSS_WEIGHT > 0 and self.score_map_ctr != None:
                        w3 = self.cfg.TRAIN_BOX_FOCAL_LOSS_WEIGHT
                        score_map_size = self.score_map_ctr.shape[-1]
                        pred_score_map_ctr_list = []
                        for b in range(len(boxes)):
                            boxes_b = boxes[b]
                            for n in range(len(boxes_b)):
                                pred_score_map_ctr_list.append(self.score_map_ctr[b,n+1].unsqueeze(0))
                        pred_score_map_ctr = torch.cat(pred_score_map_ctr_list,dim=0) # (N,4)
                        gt_score_map_ctr = generate_heatmap(gt_boxes_cat.unsqueeze(1),score_map_size)

                        score_map_loss_fun = FocalLoss(self.cfg.TRAIN_BOX_FOCAL_ALPHA,self.cfg.TRAIN_BOX_FOCAL_BETA)
                        score_map_loss = score_map_loss_fun(pred_score_map_ctr,gt_score_map_ctr)
                        self.boards['scalar']['box_focal_loss'] = score_map_loss
                        total_loss = total_loss + w3*score_map_loss
                        # print(score_map_loss)
                if self.cfg.MODEL_BOX_HEAD == 'pinpoint' and self.coord_ctr != None:
                    w4 = self.cfg.TRAIN_BOX_CTR_LOSS_WEIGHT
                    gt_coord_ctr = torch.stack([(gt_boxes_cat[:,0]+gt_boxes_cat[:,2])/2,
                                            (gt_boxes_cat[:,1]+gt_boxes_cat[:,3])/2],dim=-1)
                    pred_coord_ctr_list = []
                    for b in range(len(boxes)):
                        boxes_b = boxes[b]
                        for n in range(len(boxes_b)):
                            pred_coord_ctr_list.append(self.coord_ctr[b,n+1].unsqueeze(0))
                    pred_coord_ctr = torch.cat(pred_coord_ctr_list,dim=0)
                    coord_ctr_loss = box_loss_fun1(pred_coord_ctr,gt_coord_ctr)
                    self.boards['scalar']['box_ctr_loss'] = coord_ctr_loss
                    total_loss = total_loss + w4*coord_ctr_loss
                    # print(coord_ctr_loss)
        if self.cfg.TRAIN_WITH_BOX and len(total_loss.shape) == 0:
            total_loss = total_loss.view(1)
        return total_loss

    def generate_loss_mask(self, gt_mask, step, return_prob=False):
        if self.cfg.MODEL_DECODE_MASK_WITH_BOX:
            self.decode_current_boxes()
            self.decode_current_logits()
            if return_prob:
                mask, prob = self.predict_current_mask(return_prob=True)
                
            else:
                mask = self.predict_current_mask()
        else:
            self.decode_current_logits()
            if return_prob:
                mask, prob = self.predict_current_mask(return_prob=True)
                
            else:
                mask = self.predict_current_mask()
            if self.cfg.TRAIN_BOX_LOSS:
                if self.use_prev_pred:
                    self.decode_current_boxes(mask=mask)
                else:
                    self.decode_current_boxes(mask=gt_mask)
        
            
        loss = self.calculate_current_loss(gt_mask, step)
        if return_prob:
            return loss, mask, prob
        else:
            return loss, mask
    
    def calculate_box_loss(self):
        mask_loss_w = self.cfg.TRAIN_BOX_ID_ENCODER_MASK_LOSS_WEIGHT if self.cfg.MODEL_USE_BOX_ID_ENCODER\
                        else self.cfg.TRAIN_BOXT_MASK_LOSS_WEIGHT
        id_loss_w = self.cfg.TRAIN_BOX_ID_ENCODER_ID_LOSS_WEIGHT if self.cfg.MODEL_USE_BOX_ID_ENCODER\
                        else self.cfg.TRAIN_BOXT_ID_LOSS_WEIGHT
        loss = 0
        if self.box_mask_losses != []:
            box_mask_loss = mask_loss_w * torch.stack(self.box_mask_losses,dim=0).mean(dim=0)
            loss = loss + box_mask_loss
            self.boards['scalar']['box_mask_loss'] = box_mask_loss
            for i in range(len(self.box_masks)):
                self.boards['image']['box2mask_'+str(i)] = self.box_masks[i]
        if self.box_id_losses != []:
            box_id_loss = id_loss_w * torch.stack(self.box_id_losses,dim=0).mean(dim=0)
            loss = loss + box_id_loss
            self.boards['scalar']['box_id_loss'] = box_id_loss
        self.box_mask_losses = []
        self.box_id_losses = []
        self.box_masks = []
        return loss

    def keep_gt_mask(self, pred_mask, keep_prob=0.2):
        pred_mask = pred_mask.float()
        gt_mask = self.offline_masks[self.frame_step].float().squeeze(1)

        shape = [1 for _ in range(pred_mask.ndim)]
        shape[0] = self.batch_size
        random_tensor = keep_prob + torch.rand(
            shape, dtype=pred_mask.dtype, device=pred_mask.device)
        random_tensor.floor_()  # binarize

        pred_mask = pred_mask * (1 - random_tensor) + gt_mask * random_tensor

        return pred_mask

    def restart_engine(self, batch_size=1, enable_id_shuffle=False):

        self.batch_size = batch_size
        self.frame_step = 0
        self.last_long_step = -1
        self.last_short_step= -1
        self.enable_id_shuffle = enable_id_shuffle
        self.freeze_id = False

        self.obj_nums = None
        self.pos_emb = None
        self.enc_size_2d = None
        self.enc_hw = None
        self.input_size_2d = None

        self.long_term_memories = None
        self.short_term_memories_list = []
        self.short_term_memories = None

        self.enable_offline_enc = False
        self.offline_enc_embs = None
        self.offline_one_hot_masks = None
        self.offline_frames = -1
        self.total_offline_frame_num = 0
        self.ref_frame_num = 0

        self.curr_enc_embs = None
        self.curr_memories = None
        self.curr_id_embs = None

        self.box_mask_losses = []
        self.box_id_losses = []
        self.box_masks = []
        self.is_box = []

        if enable_id_shuffle:
            self.id_shuffle_matrix = generate_permute_matrix(
                self.max_obj_num + 1, batch_size, gpu_id=self.gpu_id)
        else:
            self.id_shuffle_matrix = None

    def update_size(self, input_size, enc_size):
        self.input_size_2d = input_size
        self.enc_size_2d = enc_size
        self.enc_hw = self.enc_size_2d[0] * self.enc_size_2d[1]


class MITSInferEngine(nn.Module):
    def __init__(self,
                 aot_model,
                 gpu_id=0,
                 long_term_mem_gap=9999,
                 short_term_mem_skip=1,
                 max_aot_obj_num=None):
        super().__init__()

        self.cfg = aot_model.cfg
        self.AOT = aot_model

        if max_aot_obj_num is None or max_aot_obj_num > aot_model.max_obj_num:
            self.max_aot_obj_num = aot_model.max_obj_num
        else:
            self.max_aot_obj_num = max_aot_obj_num

        self.gpu_id = gpu_id
        self.long_term_mem_gap = long_term_mem_gap
        self.short_term_mem_skip = short_term_mem_skip
        
        from typing import List
        self.aot_engines = [] # type: List[MITSEngine]

        self.restart_engine()

    def restart_engine(self):
        del (self.aot_engines)
        self.aot_engines = []
        self.obj_nums = None

    def separate_mask(self, mask):
        if mask is None:
            return [None] * len(self.aot_engines)
        if len(self.aot_engines) == 1:
            return [mask]

        if len(mask.size()) == 3 or mask.size()[0] == 1:
            separated_masks = []
            for idx in range(len(self.aot_engines)):
                start_id = idx * self.max_aot_obj_num + 1
                end_id = (idx + 1) * self.max_aot_obj_num
                fg_mask = ((mask >= start_id) & (mask <= end_id)).float()
                separated_mask = (fg_mask * mask - start_id + 1) * fg_mask
                separated_masks.append(separated_mask)
            return separated_masks
        else:
            prob = mask
            separated_probs = []
            for idx in range(len(self.aot_engines)):
                start_id = idx * self.max_aot_obj_num + 1
                end_id = (idx + 1) * self.max_aot_obj_num
                fg_prob = prob[start_id:(end_id + 1)]
                bg_prob = 1. - torch.sum(fg_prob, dim=1, keepdim=True)
                separated_probs.append(torch.cat([bg_prob, fg_prob], dim=1))
            return separated_probs

    def min_logit_aggregation(self, all_logits):
        if len(all_logits) == 1:
            return all_logits[0]

        fg_logits = []
        bg_logits = []

        for logit in all_logits:
            bg_logits.append(logit[:, 0:1])
            fg_logits.append(logit[:, 1:1 + self.max_aot_obj_num])

        bg_logit, _ = torch.min(torch.cat(bg_logits, dim=1),
                                dim=1,
                                keepdim=True)
        merged_logit = torch.cat([bg_logit] + fg_logits, dim=1)

        return merged_logit

    def soft_logit_aggregation(self, all_logits):
        if len(all_logits) == 1:
            return all_logits[0]

        fg_probs = []
        bg_probs = []

        for logit in all_logits:
            prob = torch.softmax(logit, dim=1)
            bg_probs.append(prob[:, 0:1])
            fg_probs.append(prob[:, 1:1 + self.max_aot_obj_num])

        bg_prob = torch.prod(torch.cat(bg_probs, dim=1), dim=1, keepdim=True)
        merged_prob = torch.cat([bg_prob] + fg_probs,
                                dim=1).clamp(1e-5, 1 - 1e-5)
        merged_logit = torch.logit(merged_prob)

        return merged_logit

    def add_reference_frame(self, img, mask, obj_nums, frame_step=-1):
        if isinstance(obj_nums, list):
            obj_nums = obj_nums[0]
        self.obj_nums = obj_nums
        aot_num = max(np.ceil(obj_nums / self.max_aot_obj_num), 1)
        while (aot_num > len(self.aot_engines)):
            new_engine = MITSEngine(self.AOT, self.gpu_id,
                                     self.long_term_mem_gap,
                                     self.short_term_mem_skip)
            new_engine.eval()
            self.aot_engines.append(new_engine)

        separated_masks = self.separate_mask(mask)
        
        separated_obj_nums = [
            self.max_aot_obj_num for _ in range(len(self.aot_engines)-1)
        ]
        separated_obj_nums.append(obj_nums - self.max_aot_obj_num * (len(self.aot_engines)-1))
        
        img_embs = None
        for aot_engine, separated_mask, separated_obj_num in zip(self.aot_engines,
                                              separated_masks, separated_obj_nums):
            aot_engine.add_reference_frame(img,
                                           separated_mask,
                                           obj_nums=[separated_obj_num],
                                           frame_step=frame_step,
                                           img_embs=img_embs)
            if img_embs is None:  # reuse image embeddings
                img_embs = aot_engine.curr_enc_embs

        self.update_size()

    def match_propogate_one_frame(self, img=None):
        img_embs = None
        for aot_engine in self.aot_engines:
            aot_engine.match_propogate_one_frame(img, img_embs=img_embs)
            if img_embs is None:  # reuse image embeddings
                img_embs = aot_engine.curr_enc_embs

    def decode_current_logits(self, output_size=None):
        all_logits = []
        for aot_engine in self.aot_engines:
            all_logits.append(aot_engine.decode_current_logits(output_size))
        pred_id_logits = self.soft_logit_aggregation(all_logits)
        return pred_id_logits
    
    def decode_current_boxes(self,output_size=None,img=None,mask=None):
        # box number > max_num not considered
        boxes = self.aot_engines[0].decode_current_boxes(output_size,img,mask)
        return boxes

    def update_memory(self, curr_mask, _update_long=True, _update_short=True):
        separated_masks = self.separate_mask(curr_mask)
        for aot_engine, separated_mask in zip(self.aot_engines,
                                              separated_masks):
            aot_engine.update_short_term_memory(separated_mask,_update_long=_update_long,_update_short=_update_short)
    def id_box2mask(self,img,one_hot,return_logits=False):
        if return_logits:
            id_emb,logits = self.aot_engines[0].id_box2mask(img,one_hot,return_logits=return_logits)
            return id_emb,logits
        else:
            id_emb = self.aot_engines[0].id_box2mask(img,one_hot,return_logits=return_logits)
            return id_emb
    def update_memory_with_id(self,curr_id_emb):
        self.aot_engines[0].update_short_term_memory(None,curr_id_emb=curr_id_emb)

    def update_size(self):
        self.input_size_2d = self.aot_engines[0].input_size_2d
        self.enc_size_2d = self.aot_engines[0].enc_size_2d
        self.enc_hw = self.aot_engines[0].enc_hw
