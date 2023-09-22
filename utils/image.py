from tkinter import N
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import threading
# from torchvision.ops import masks_to_boxes

_palette = [
    0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 
    128, 128, 64, 0, 0, 191, 0, 0, 64, 128, 0, 191, 128, 0, 64, 0, 128, 191,
    0, 128, 64, 128, 128, 191, 128, 128, 0, 64, 0, 128, 64, 0, 0, 191, 0, 128,
    191, 0, 0, 64, 128, 128, 64, 128, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0,
    128, 128, 0, 128, 0, 128, 128, 64, 0, 0, 191, 0, 0, 64, 128, 0, 191, 128, 0,
    64, 0, 128, 191, 0, 128, 64, 128, 128, 191, 128, 128, 0, 64, 0, 128, 64, 0, 0,
    191, 0, 128, 191, 0, 0, 64, 128, 128, 64, 128, 64, 192, 128, 64, 64, 128, 64,
    128, 128, 128, 192, 128, 128, 64, 128, 192, 192, 128, 192, 64, 128, 192, 128,
    128, 64, 192, 192, 64, 64, 192, 64, 128, 192, 128, 192, 192, 128, 64, 192, 128,
    128, 192, 192, 64, 192, 192, 128, 192, 64, 192, 64, 64, 128, 64, 128, 192, 64,
    128, 64, 64, 128, 128, 64, 192, 192, 64, 192, 64, 64, 192, 128, 64, 0, 192, 128,
    0, 0, 128, 0, 128, 128, 128, 192, 128, 128, 0, 128, 192, 192, 128, 192, 0, 128,
    192, 128, 128, 0, 192, 192, 0, 0, 192, 0, 128, 192, 128, 192, 192, 128, 0, 192,
    128, 128, 192, 192, 0, 192, 192, 128, 192, 0, 192, 0, 0, 128, 0, 128, 192, 0, 128,
    0, 0, 128, 128, 0, 192, 192, 0, 192, 0, 0, 192, 128, 0, 0, 192, 64, 0, 0, 64, 0, 64,
    64, 64, 192, 64, 64, 0, 64, 192, 192, 64, 192, 0, 64, 192, 64, 64, 0, 192, 192, 0, 0,
    192, 0, 64, 192, 64, 192, 192, 64, 0, 192, 64, 64, 192, 192, 0, 192, 192, 64, 192, 0,
    192, 0, 0, 64, 0, 64, 192, 0, 64, 0, 0, 64, 64, 0, 192, 192, 0, 192, 0, 0, 192, 64, 0,
    0, 128, 64, 0, 0, 64, 0, 64, 64, 64, 128, 64, 64, 0, 64, 128, 128, 64, 128, 0, 64, 128,
    64, 64, 0, 128, 128, 0, 0, 128, 0, 64, 128, 64, 128, 128, 64, 0, 128, 64, 64, 128, 128,
    0, 128, 128, 64, 128, 0, 128, 0, 0, 64, 0, 64, 128, 0, 64, 0, 0, 64, 64, 0, 128, 128, 0,
    128, 0, 0, 128, 64, 0
]


def label2colormap(label):

    m = label.astype(np.uint8)
    r, c = m.shape
    cmap = np.zeros((r, c, 3), dtype=np.uint8)
    cmap[:, :, 0] = (m & 1) << 7 | (m & 8) << 3 | (m & 64) >> 1
    cmap[:, :, 1] = (m & 2) << 6 | (m & 16) << 2 | (m & 128) >> 2
    cmap[:, :, 2] = (m & 4) << 5 | (m & 32) << 1
    return cmap


def one_hot_mask(mask, cls_num):
    if len(mask.size()) == 3:
        mask = mask.unsqueeze(1)
    indices = torch.arange(0, cls_num + 1,
                           device=mask.device).view(1, -1, 1, 1)
    return (mask == indices).float()


def masked_image(image, colored_mask, mask, alpha=0.7):
    mask = np.expand_dims(mask > 0, axis=0)
    mask = np.repeat(mask, 3, axis=0)
    show_img = (image * alpha + colored_mask *
                (1 - alpha)) * mask + image * (1 - mask)
    return show_img


def save_image(image, path):
    im = Image.fromarray(np.uint8(image * 255.).transpose((1, 2, 0)))
    im.save(path)


def _save_mask(mask, path, squeeze_idx=None):
    if squeeze_idx is not None:
        unsqueezed_mask = mask * 0
        for idx in range(1, len(squeeze_idx)):
            obj_id = squeeze_idx[idx]
            mask_i = mask == idx
            unsqueezed_mask += (mask_i * obj_id).astype(np.uint8)
        mask = unsqueezed_mask
    mask = Image.fromarray(mask).convert('P')
    mask.putpalette(_palette)
    mask.save(path)


def save_mask(mask_tensor, path, squeeze_idx=None):
    mask = mask_tensor.cpu().numpy().astype('uint8')
    threading.Thread(target=_save_mask, args=[mask, path, squeeze_idx]).start()

def save_prob(prob,path,squeeze_idx=None,scale=1):
    if scale != 1:
        # prob = F.interpolate(prob,scale_factor=scale,mode='nearest')
        prob = F.interpolate(prob,scale_factor=scale,mode='bilinear')
    prob = prob.squeeze(0)
    
    if squeeze_idx is None:
        idx_prob = prob
    else:
        c,h,w = prob.shape
        idx_prob = torch.zeros((max(squeeze_idx)+1,h,w),device=prob.device,dtype=prob.dtype)
        for i in range(len(squeeze_idx)):
            idx_prob[squeeze_idx[i]] = prob[i]
        # idx_prob = idx_prob[:max(squeeze_idx)+1]
    max_val = np.iinfo(np.uint16).max
    idx_prob = (idx_prob.cpu().numpy()*max_val).astype(np.uint16)

    np.save(path,idx_prob)
    pass

def save_logit(logit,path,obj_num=None):
    if obj_num is not None:
        logit = logit[:obj_num+1]
    torch.save(logit,path)

def flip_tensor(tensor, dim=0):
    inv_idx = torch.arange(tensor.size(dim) - 1, -1, -1,
                           device=tensor.device).long()
    tensor = tensor.index_select(dim, inv_idx)
    return tensor


def shuffle_obj_mask(mask):

    bs, obj_num, _, _ = mask.size()
    new_masks = []
    for idx in range(bs):
        now_mask = mask[idx]
        random_matrix = torch.eye(obj_num, device=mask.device)
        fg = random_matrix[1:][torch.randperm(obj_num - 1)]
        random_matrix = torch.cat([random_matrix[0:1], fg], dim=0)
        now_mask = torch.einsum('nm,nhw->mhw', random_matrix, now_mask)
        new_masks.append(now_mask)

    return torch.stack(new_masks, dim=0)

def label2box(label):
    '''
    label: tensor[int], shape(1,h,w)
    '''
    box_dict = {}
    
    for i in torch.unique(label):
        if i==0:
            continue
        box_dict[int(i)] = masks_to_boxes(label==i)[0]
    return box_dict
def box2label(box,size_2d):
    '''
    box: x1,y1,wb,hb
    size_2d: (h,w)
    '''
    label = np.zeros(size_2d,dtype=np.uint8)
    [x1,y1,x2,y2] = list(map(int,[box[0],box[1],box[0]+box[2],box[1]+box[3]]))
    label[y1:y2,x1:x2] = 1
    return label

def check_box(box_dict,h,w):
    new_dict = {}
    #check if big obj
    for k in box_dict.keys():
        if (box_dict[k][2]- box_dict[k][0]) > (w * 0.5) or \
            (box_dict[k][3] - box_dict[k][1]) > (h * 0.6):
                continue
        else:
            new_dict[k] = box_dict[k]
    return new_dict

def box_filter(box_dict,label,prob=None,h_ratio=0.5,w_ratio=0.5):
    
    
    h,w = label.shape[2],label.shape[3]
    for k in box_dict.keys():
        obj_filter = torch.zeros_like(label,dtype=torch.uint8)
        center_x = (box_dict[k][0] + box_dict[k][2]) /2
        center_y = (box_dict[k][1] + box_dict[k][3]) /2
        box = [center_x - w*w_ratio/2,center_y - h*h_ratio/2,
                center_x + w*w_ratio/2,center_y + h*h_ratio/2]
        box = list(map(lambda x: int(x), box))
        box[0] = max(box[0],0)
        box[1] = max(box[1],0)
        box[2] = min(box[2],w)
        box[3] = min(box[3],h)
        obj_filter[:,:,box[1]:box[3],box[0]:box[2]] = 1
    
        filter_mask = (label.to(torch.uint8)==k) & (obj_filter==0)
        label[filter_mask] = 0.
        
        if prob != None:
            c = prob.shape[1]
            bk = torch.zeros_like(prob)
            bk[:,0] = 1
            filter_mask_c = filter_mask.repeat(1,c,1,1)
            prob[filter_mask_c] = bk[filter_mask_c]
    
    return [label,prob]


def masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
    """
    Compute the bounding boxes around the provided masks.

    Returns a [N, 4] tensor containing bounding boxes. The boxes are in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Args:
        masks (Tensor[N, H, W]): masks to transform where N is the number of masks
            and (H, W) are the spatial dimensions.

    Returns:
        Tensor[N, 4]: bounding boxes
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device, dtype=torch.float)

    n = masks.shape[0]

    bounding_boxes = torch.zeros((n, 4), device=masks.device, dtype=torch.float)

    for index, mask in enumerate(masks):
        y, x = torch.where(mask != 0)
        if len(x) == 0 or len(y) == 0:
            bounding_boxes[index, 0] = 0
            bounding_boxes[index, 1] = 0
            bounding_boxes[index, 2] = 0
            bounding_boxes[index, 3] = 0
        else:
            bounding_boxes[index, 0] = torch.min(x)
            bounding_boxes[index, 1] = torch.min(y)
            bounding_boxes[index, 2] = torch.max(x)
            bounding_boxes[index, 3] = torch.max(y)

    return bounding_boxes

def mask_to_box(mask: torch.Tensor, normalize=False) -> torch.Tensor:
    """
    Compute the bounding box around the provided mask.
    if no box, return None
    Args:
        mask (Tensor (H, W))

    Returns:
        Tensor (4): bounding boxes or None
    """
    if mask.numel() == 0:
        return None

    bounding_box = torch.zeros((4), device=mask.device, dtype=torch.float)

    y, x = torch.where(mask != 0)
    if len(x) == 0 or len(y) == 0:
        return None
    else:
        bounding_box[0] = torch.min(x)
        bounding_box[1] = torch.min(y)
        bounding_box[2] = torch.max(x)
        bounding_box[3] = torch.max(y)
        if normalize:
            bounding_box[0] = bounding_box[0] / mask.shape[-1]
            bounding_box[1] = bounding_box[1] / mask.shape[-2]
            bounding_box[2] = bounding_box[2] / mask.shape[-1]
            bounding_box[3] = bounding_box[3] / mask.shape[-2]
        return bounding_box

def boxize_mask(one_hot_mask,return_corners=False,return_centers=False,norm_coord=False):
    '''
    convert mask to box mask
    Note the channel 0 for background need to be substracted but not filled
    one_hot_mask: (B,N,H,W)
    return (B,N,H,W)
    '''
    corners_list = []
    centers_list = []
    box_mask = torch.zeros_like(one_hot_mask)
    box_mask[:,0,:,:] = 1 # for background
    for b in range(box_mask.shape[0]):
        corners = []
        centers = []
        for n in range(1,box_mask.shape[1]):
            box = mask_to_box(one_hot_mask[b,n])
            if box!=None:
                [x1,y1,x2,y2] = box
                box_mask[b,n,y1.int():y2.int()+1,x1.int():x2.int()+1] = 1
                box_mask[b,0,y1.int():y2.int()+1,x1.int():x2.int()+1] = 0
                if norm_coord:
                    h,w = one_hot_mask.shape[2:]
                    corners.append([x1/w,y1/h,x2/w,y2/h])
                    centers.append([(x1/w+x2/w)/2,(y1/h+y2/h)/2,(x2-x1)/w,(y2-y1)/h])
                else:
                    corners.append([x1,y1,x2,y2])
                    centers.append([(x1+x2)//2,(y1+y2)//2,x2-x1,y2-y1])
        corners_list.append(corners)
        centers_list.append(centers)
    if return_corners:
        return box_mask,corners_list
    elif return_centers:
        return box_mask,centers_list
    else:
        return box_mask

def corner_grid_sample(img,corner,output_shape,align_corners=True):
    '''
    img: N,C,H,W
    corner (normalized): x1,y1,x2,y2 in [0,1]
    output_shape: out_h,out_w
    return img_box: N,C,h,w
    '''
    out_h,out_w = output_shape[0],output_shape[1]
    (x1,y1,x2,y2) = (each * 2 - 1 for each in corner) # [-1,1]
    d_x = torch.linspace(x1,x2,out_w,device=img.device)
    d_y = torch.linspace(y1,y2,out_h,device=img.device)
    meshy,meshx = torch.meshgrid((d_y,d_x))
    grid = torch.stack((meshx,meshy),2).unsqueeze(0)
    output = F.grid_sample(img,grid,align_corners=align_corners)
    return output

def resize_box(box,ratio,x_max=float('inf'),y_max=float('inf'),keep_size=True,min_size=0):
    x1 = float(box[0])
    y1 = float(box[1])
    x2 = float(box[2])
    y2 = float(box[3])
    x_c = (x2+x1)/2
    y_c = (y2+y1)/2
    w = x2-x1
    h = y2-y1
    a = (w*h)**0.5 * ratio # dilated box size
    if keep_size:
        if a < min_size:
            a = min_size
        if a > min([x_max,y_max]) * 0.95: # for large box, restrict its size
            a = min([x_max,y_max]) * 0.95
        # restrict new box center
        if x_c < a/2:
            x_c = a/2
        if x_c > x_max - a/2:
            x_c = x_max - a/2
        if y_c < a/2:
            y_c = a/2
        if y_c > y_max - a/2:
            y_c = y_max - a/2
        # cal corner points
        x1_ = x_c - a/2
        x2_ = x_c + a/2
        y1_ = y_c - a/2
        y2_ = y_c + a/2
    else:
        x1_ = max(x_c - a/2,0)
        x2_ = min(x_c + a/2,x_max)
        y1_ = max(y_c - a/2,0)
        y2_ = min(y_c + a/2,y_max)
    return [x1_,y1_,x2_,y2_]

def update_box(box, x_c, y_c, x_max=float('inf'),y_max=float('inf')):
    a = max(box[2]-box[0],box[3]-box[1])
    if a > min([x_max,y_max]) * 0.95: # for large box, restrict its size
            a = min([x_max,y_max]) * 0.95
    # restrict new box center
    if x_c < a/2:
        x_c = a/2
    if x_c > x_max - a/2:
        x_c = x_max - a/2
    if y_c < a/2:
        y_c = a/2
    if y_c > y_max - a/2:
        y_c = y_max - a/2
    # cal corner points
    x1_ = x_c - a/2
    x2_ = x_c + a/2
    y1_ = y_c - a/2
    y2_ = y_c + a/2
    return [x1_,y1_,x2_,y2_]

def box_iou(boxes1, boxes2):
        """
        :param boxes1: (N, 4) (x1,y1,x2,y2)
        :param boxes2: (N, 4) (x1,y1,x2,y2)
        :return:
        """
        def box_area(boxes):
            return (boxes[:,2]-boxes[:,0]) * (boxes[:,3]-boxes[:,1])
        area1 = box_area(boxes1) # (N,)
        area2 = box_area(boxes2) # (N,)

        lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # (N,2)
        rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # (N,2)

        wh = (rb - lt).clamp(min=0)  # (N,2)
        inter = wh[:, 0] * wh[:, 1]  # (N,)

        union = area1 + area2 - inter
        if union == 0:
            return 0
        else:
            iou = inter / union
            return iou

def boxes2onehot(boxes,size_2d,num_channel):
    '''
    boxes: (N,4)
    size_2d: (h,w)
    '''
    (h,w) = size_2d
    if len(boxes) == 0:
        raise ValueError
    one_hot = torch.zeros((1,num_channel,h,w),device=boxes[0].device)
    one_hot[:,0,:,:] = 1
    for n in range(len(boxes)):
        lt_x = int(boxes[n][0] * w)
        lt_y = int(boxes[n][1] * h)
        br_x = int(boxes[n][2] * w)
        br_y = int(boxes[n][3] * h)
        # print(lt_x,lt_y,br_x,br_y)
        one_hot[:,n+1,lt_y:br_y,lt_x:br_x] = 1
        one_hot[:,0,lt_y:br_y,lt_x:br_x] = 0
    return one_hot

def box_valid(box,size_2d):
    "size_2d: (h,w)"
    if (box[2]-box[0])*size_2d[1] > 1 and (box[3]-box[1]) * size_2d[0] > 1:
        return True
    else:
        return False