import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from itertools import ifilterfalse
except ImportError:  # py3k
    from itertools import filterfalse as ifilterfalse


def dice_loss(probas, labels, smooth=1):

    C = probas.size(1)
    losses = []
    for c in list(range(C)):
        fg = (labels == c).float()
        if fg.sum() == 0:
            continue
        class_pred = probas[:, c]
        p0 = class_pred
        g0 = fg
        numerator = 2 * torch.sum(p0 * g0) + smooth
        denominator = torch.sum(p0) + torch.sum(g0) + smooth
        losses.append(1 - ((numerator) / (denominator)))
    return mean(losses)


def tversky_loss(probas, labels, alpha=0.5, beta=0.5, epsilon=1e-6):
    '''
    Tversky loss function.
        probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
        labels: [P] Tensor, ground truth labels (between 0 and C - 1)

    Same as soft dice loss when alpha=beta=0.5.
    Same as Jaccord loss when alpha=beta=1.0.
    See `Tversky loss function for image segmentation using 3D fully convolutional deep networks`
    https://arxiv.org/pdf/1706.05721.pdf
    '''
    C = probas.size(1)
    losses = []
    for c in list(range(C)):
        fg = (labels == c).float()
        if fg.sum() == 0:
            continue
        class_pred = probas[:, c]
        p0 = class_pred
        p1 = 1 - class_pred
        g0 = fg
        g1 = 1 - fg
        numerator = torch.sum(p0 * g0)
        denominator = numerator + alpha * \
            torch.sum(p0*g1) + beta*torch.sum(p1*g0)
        losses.append(1 - ((numerator) / (denominator + epsilon)))
    return mean(losses)


def flatten_probas(probas, labels, ignore=255):
    """
    Flattens predictions in the batch
    """
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3,
                            1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.view(-1, 1).expand(-1, C)].reshape(-1, C)
    # vprobas = probas[torch.nonzero(valid).squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels


def isnan(x):
    return x != x


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


class DiceLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, tmp_dic, label_dic, step=None):
        total_loss = []
        for idx in range(len(tmp_dic)):
            pred = tmp_dic[idx]
            label = label_dic[idx]
            pred = F.softmax(pred, dim=1)
            label = label.view(1, 1, pred.size()[2], pred.size()[3])
            loss = dice_loss(
                *flatten_probas(pred, label, ignore=self.ignore_index))
            total_loss.append(loss.unsqueeze(0))
        total_loss = torch.cat(total_loss, dim=0)
        return total_loss


class SoftJaccordLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super(SoftJaccordLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, tmp_dic, label_dic, step=None):
        total_loss = []
        for idx in range(len(tmp_dic)):
            pred = tmp_dic[idx]
            label = label_dic[idx]
            pred = F.softmax(pred, dim=1)
            label = label.view(1, 1, pred.size()[2], pred.size()[3])
            loss = tversky_loss(*flatten_probas(pred,
                                                label,
                                                ignore=self.ignore_index),
                                alpha=1.0,
                                beta=1.0)
            total_loss.append(loss.unsqueeze(0))
        total_loss = torch.cat(total_loss, dim=0)
        return total_loss


class CrossEntropyLoss(nn.Module):
    def __init__(self,
                 top_k_percent_pixels=None,
                 hard_example_mining_step=100000):
        super(CrossEntropyLoss, self).__init__()
        self.top_k_percent_pixels = top_k_percent_pixels
        if top_k_percent_pixels is not None:
            assert (top_k_percent_pixels > 0 and top_k_percent_pixels < 1)
        self.hard_example_mining_step = hard_example_mining_step + 1e-5
        if self.top_k_percent_pixels is None:
            self.celoss = nn.CrossEntropyLoss(ignore_index=255,
                                              reduction='mean')
        else:
            self.celoss = nn.CrossEntropyLoss(ignore_index=255,
                                              reduction='none')

    def forward(self, dic_tmp, y, step):
        total_loss = []
        for i in range(len(dic_tmp)):
            pred_logits = dic_tmp[i]
            gts = y[i]
            if self.top_k_percent_pixels is None:
                final_loss = self.celoss(pred_logits, gts)
            else:
                # Only compute the loss for top k percent pixels.
                # First, compute the loss for all pixels. Note we do not put the loss
                # to loss_collection and set reduction = None to keep the shape.
                num_pixels = float(pred_logits.size(2) * pred_logits.size(3))
                pred_logits = pred_logits.view(
                    -1, pred_logits.size(1),
                    pred_logits.size(2) * pred_logits.size(3))
                gts = gts.view(-1, gts.size(1) * gts.size(2))
                pixel_losses = self.celoss(pred_logits, gts)
                if self.hard_example_mining_step == 0:
                    top_k_pixels = int(self.top_k_percent_pixels * num_pixels)
                else:
                    ratio = min(1.0,
                                step / float(self.hard_example_mining_step))
                    top_k_pixels = int((ratio * self.top_k_percent_pixels +
                                        (1.0 - ratio)) * num_pixels)
                top_k_loss, top_k_indices = torch.topk(pixel_losses,
                                                       k=top_k_pixels,
                                                       dim=1)

                final_loss = torch.mean(top_k_loss)
            final_loss = final_loss.unsqueeze(0)
            total_loss.append(final_loss)
        total_loss = torch.cat(total_loss, dim=0)
        return total_loss

class GeneralizedBoxIoULoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, boxes1, boxes2):
        """
        :param boxes1: (N, 4) (x1,y1,x2,y2)
        :param boxes2: (N, 4) (x1,y1,x2,y2)
        :return:
        """
        giou, iou = self.generalized_box_iou(boxes1, boxes2)
        if giou != None and iou != None:
            return (1 - giou).mean(), iou
        else:
            return 0,None
    
    def box_area(self,boxes):
        return (boxes[:,2]-boxes[:,0]) * (boxes[:,3]-boxes[:,1])
    
    def box_iou(self,boxes1, boxes2):
        """
        :param boxes1: (N, 4) (x1,y1,x2,y2)
        :param boxes2: (N, 4) (x1,y1,x2,y2)
        :return:
        """
        area1 = self.box_area(boxes1) # (N,)
        area2 = self.box_area(boxes2) # (N,)

        lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # (N,2)
        rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # (N,2)

        wh = (rb - lt).clamp(min=0)  # (N,2)
        inter = wh[:, 0] * wh[:, 1]  # (N,)

        union = area1 + area2 - inter

        iou = inter / union
        return iou, union

    def generalized_box_iou(self,boxes1, boxes2):
        """
        Generalized IoU from https://giou.stanford.edu/
        The boxes should be in [x0, y0, x1, y1] format
        boxes1: (N, 4)
        boxes2: (N, 4)
        """
        # degenerate boxes gives inf / nan results
        # so do an early check
        # try:
        if (boxes1[:, 2:] > boxes1[:, :2]).all() and (boxes2[:, 2:] > boxes2[:, :2]).all():
            iou, union = self.box_iou(boxes1, boxes2) # (N,)

            lt = torch.min(boxes1[:, :2], boxes2[:, :2])
            rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])

            wh = (rb - lt).clamp(min=0)  # (N,2)
            area = wh[:, 0] * wh[:, 1] # (N,)

            return iou - (area - union) / area, iou
        else:
            return None,None

class FocalLoss(nn.Module):
    def __init__(self, alpha=2, beta=4):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, prediction, target):
        positive_index = target.eq(1).float()
        negative_index = target.lt(1).float()

        negative_weights = torch.pow(1 - target, self.beta)
        # clamp min value is set to 1e-12 to maintain the numerical stability
        prediction = torch.clamp(prediction, 1e-6, 1 - 1e-3)

        positive_loss = torch.log(prediction) * torch.pow(1 - prediction, self.alpha) * positive_index
        negative_loss = torch.log(1 - prediction) * torch.pow(prediction,
                                                              self.alpha) * negative_weights * negative_index
        
        num_positive = positive_index.float().sum()
        num_negative = negative_index.float().sum()
        positive_loss = positive_loss.sum()
        negative_loss = negative_loss.sum()
        # print(positive_loss,negative_loss)

        if num_positive == 0:
            loss = -negative_loss / num_negative
        else:
            loss = -(positive_loss + negative_loss) / (num_positive+num_negative)

        return loss