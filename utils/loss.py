import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:

    # Compute losses
    def __init__(self, model, autobalance=False):
        self.sort_obj_iou = False
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)
        #quoccuong
        det = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.nc = det.nc  # number of classes
        self.nl = det.nl  # number of layers
        self.anchors = det.anchors
        self.device = device

    def __call__(self, p, targets):  # predictions, targets
        bs = p[0].shape[0]  # batch size
        loss = torch.zeros(3, device=self.device)  # [box, obj, cls] losses
        tcls, tbox, indices = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros((pi.shape[0], pi.shape[2], pi.shape[3]), dtype=pi.dtype, device=self.device)  # tgt obj

            n_labels = b.shape[0]  # number of labels
            if n_labels:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh, _, pcls = pi[b, :, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                pxy = pxy.sigmoid() * 1.6 - 0.3
                pwh = (0.2 + pwh.sigmoid() * 4.8) * self.anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                loss[0] += (1.0 - iou).mean()  # box loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, gj, gi, iou = b[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n_labels), tcls[i]] = self.cp
                    loss[2] += self.BCEcls(pcls, t)  # cls loss

            obji = self.BCEobj(pi[:, 4], tobj)
            loss[1] += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        loss[0] *= self.hyp['box']
        loss[1] *= self.hyp['obj']
        loss[2] *= self.hyp['cls']
        return loss.sum() * bs, loss.detach()  # [box, obj, cls] losses

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        nt = targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices = [], [], []
        gain = torch.ones(6, device=self.device)  # normalized to gridspace gain

        g = 0.3  # bias
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float() * g  # offsets

        for i in range(self.nl):
            shape = p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)
            if nt:
                # Matches
                r = t[..., 4:6] / self.anchors[i]  # wh ratio
                j = torch.max(r, 1 / r).max(1)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh = t.chunk(3, 1)  # (image, class), grid xy, grid wh
            b, c = bc.long().T  # image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, grid_y, grid_x indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            tcls.append(c)  # class

        return tcls, tbox, indices

#quoccuong
import math
from utils.RepulsionLoss import repulsion_loss_torch
from utils.general import Wasserstein

class reSwanLoss(nn.Module):
    def __init__(self, loss_fcn):
        super(reSwanLoss, self).__init__()
        self.loss_fcn = loss_fcn
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply SL to each element

    def forward(self, pred, true, auto_iou=0.5):
        loss = self.loss_fcn(pred, true)
        if auto_iou < 0.2:
            auto_iou = 0.2
        b1 = true <= auto_iou - 0.1
        a1 = 1.0
        b2 = (true > (auto_iou - 0.1)) & (true < auto_iou)
        a2 = math.exp(auto_iou)
        b3 = true >= auto_iou
        a3 = torch.exp(-(true - 1.0))
        modulating_weight = a1 * b1 + a2 * b2 + a3 * b3
        loss *= modulating_weight
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

class ComputeNWDLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        self.sort_obj_iou = False
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # update c: 3.0, u: 1.0
        self.C = 3.0
        # reswan loss
        self.u = 1.0
        if self.u > 0:
            BCEcls, BCEobj = reSwanLoss(BCEcls), reSwanLoss(BCEobj)
        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets):  # predictions, targets, model
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        lrepBox, lrepGT = torch.zeros(1, device=device), torch.zeros(1, device=device) #update
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets
        auto_iou=0.5
        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                # nwd update
                nwd = torch.exp(-torch.pow(Wasserstein(pbox.T, tbox[i], x1y1x2y2=False), 1 / 2) / self.C)
                auto_iou = iou.mean()
                lbox += 0.8 * (1.0 - iou).mean() + 0.2 * (1.0 - nwd).mean()
                # lbox += (1.0 - iou).mean()  # iou loss origin

                # Objectness
                score_iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    sort_id = torch.argsort(score_iou)
                    b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    if self.u > 0:
                        lcls += self.BCEcls(ps[:, 5:], t, auto_iou)  # BCE
                    else:
                        lcls += self.BCEcls(ps[:, 5:], t)  # BCE
                    # lcls += self.BCEcls(ps[:, 5:], t)  # BCEåŽŸå§‹
                
                # Replusion Loss update
                dic = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: [], 12: [],
                       13: [], 14: [], 15: [], 16: [], 17: [], 18: [], 19: [], 20: [], 21: [], 22: [], 23: [], 24: [],
                       25: [], 26: [], 27: [], 28: [], 29: [], 30: [], 31: []}
                for indexs, value in enumerate(b):
                    # print(indexs, value)
                    dic[int(value)].append(indexs)
                # print('dic', dic)
                bts = 0 # update
                deta = 0.5 # smooth_ln parameter
                Rp_nms = 0.1 # RepGT loss nms
                _lrepGT = 0.0
                _lrepBox = 0.0
                for id, indexs in dic.items():  # id = batch_name  indexs = target_id
                    if indexs:
                        lrepgt, lrepbox = repulsion_loss_torch(pbox[indexs], tbox[i][indexs], deta=deta, pnms=Rp_nms, gtnms=Rp_nms)
                        _lrepGT += lrepgt
                        _lrepBox += lrepbox
                        bts += 1
                if bts > 0:
                    _lrepGT /= bts
                    _lrepBox /= bts
                lrepGT += _lrepGT
                lrepBox += _lrepBox

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]
            
            # update
            if self.u > 0:
                obji = self.BCEobj(pi[..., 4], tobj, auto_iou)
            else:
                obji = self.BCEobj(pi[..., 4], tobj)
            # obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        # lrep = self.hyp['alpha'] * lrepGT / 3.0 + self.hyp['beta'] * lrepBox / 3.0
        # alpha-0.01: RepGT loss gain 0.04, init: 0.233 * 2
        # beta-0.1: RepBox loss gain 0.6, init: 0.0222 * 2
        lrep = 0.01 * lrepGT / 3.0 + 0.1 * lrepBox / 3.0
        bs = tobj.shape[0]  # batch size, gpu
        
        # no lrep, loss
        loss = lbox + lobj + lcls + lrep
        return loss * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
class ComputeLoss_NEW:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device
        self.BCE_base = nn.BCEWithLogitsLoss(reduction='none')

    def __call__(self, p, targets):  # predictions, targets
        tcls, tbox, indices = self.build_targets(p, targets)  # targets
        bs = p[0].shape[0]  # batch size
        n_labels = targets.shape[0]  # number of labels
        loss = torch.zeros(3, device=self.device)  # [box, obj, cls] losses

        # Compute all losses
        all_loss = []
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, gj, gi = indices[i]  # image, anchor, gridy, gridx
            if n_labels:
                pxy, pwh, pobj, pcls = pi[b, :, gj, gi].split((2, 2, 1, self.nc), 2)  # target-subset of predictions

                # Regression
                pbox = torch.cat((pxy.sigmoid() * 1.6 - 0.3, (0.2 + pwh.sigmoid() * 4.8) * self.anchors[i]), 2)
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(predicted_box, target_box)
                obj_target = iou.detach().clamp(0).type(pi.dtype)  # objectness targets

                all_loss.append([(1.0 - iou) * self.hyp['box'],
                                 self.BCE_base(pobj.squeeze(), torch.ones_like(obj_target)) * self.hyp['obj'],
                                 self.BCE_base(pcls, F.one_hot(tcls[i], self.nc).float()).mean(2) * self.hyp['cls'],
                                 obj_target,
                                 tbox[i][..., 2] > 0.0])  # valid

        # Lowest 3 losses per label
        n_assign = 4  # top n matches
        cat_loss = [torch.cat(x, 1) for x in zip(*all_loss)]
        ij = torch.zeros_like(cat_loss[0]).bool()  # top 3 mask
        sum_loss = cat_loss[0] + cat_loss[2]
        for col in torch.argsort(sum_loss, dim=1).T[:n_assign]:
            # ij[range(n_labels), col] = True
            ij[range(n_labels), col] = cat_loss[4][range(n_labels), col]
        loss[0] = cat_loss[0][ij].mean() * self.nl  # box loss
        loss[2] = cat_loss[2][ij].mean() * self.nl  # cls loss

        # Obj loss
        for i, (h, pi) in enumerate(zip(ij.chunk(self.nl, 1), p)):  # layer index, layer predictions
            b, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros((pi.shape[0], pi.shape[2], pi.shape[3]), dtype=pi.dtype, device=self.device)  # obj
            if n_labels:  # if any labels
                tobj[b[h], gj[h], gi[h]] = all_loss[i][3][h]
            loss[1] += self.BCEobj(pi[:, 4], tobj) * (self.balance[i] * self.hyp['obj'])

        return loss.sum() * bs, loss.detach()  # [box, obj, cls] losses

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        nt = targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices = [], [], []
        gain = torch.ones(6, device=self.device)  # normalized to gridspace gain

        g = 0.3  # bias
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float()  # offsets

        for i in range(self.nl):
            shape = p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)
            if nt:
                # # Matches
                r = t[..., 4:6] / self.anchors[i]  # wh ratio
                a = torch.max(r, 1 / r).max(1)[0] < self.hyp['anchor_t']  # compare
                # a = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                # t = t[a]  # filter

                # # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m)) & a
                t = t.repeat((5, 1, 1))
                offsets = torch.zeros_like(gxy)[None] + off[:, None]
                t[..., 4:6][~j] = 0.0  # move unsuitable targets far away
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh = t.chunk(3, 2)  # (image, class), grid xy, grid wh
            b, c = bc.long().transpose(0, 2).contiguous()  # image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.transpose(0, 2).contiguous()  # grid indices

            # Append
            indices.append((b, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, grid_y, grid_x indices
            tbox.append(torch.cat((gxy - gij, gwh), 2).permute(1, 0, 2).contiguous())  # box
            tcls.append(c)  # class

            # # Unique
            # n1 = torch.cat((b.view(-1, 1), tbox[i].view(-1, 4)), 1).shape[0]
            # n2 = tbox[i].view(-1, 4).unique(dim=0).shape[0]
            # print(f'targets-unique {n1}-{n2} diff={n1-n2}')

        return tcls, tbox, indices
