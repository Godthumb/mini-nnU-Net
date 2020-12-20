import torch.nn as nn
import torch
import torch.nn.functional as F

class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5, classes=3):

        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth
        self.classes = classes

    def forward(self, pred, target):
        '''

        :param pred:  (b, c, d, h, w)
        :param target: original_seg_arr (b, 1, d, h, w), will be translate to one_hot for loss computing
        :return:
        '''
        b =  pred.shape[0]
        # do softmax here
        pred = F.softmax(pred, dim=1)
        pred = pred.view(b, -1)
        ohe_target = self.create_one_hot(target, self.classes)
        ohe_target = ohe_target.view(b, -1)
        intersection = torch.sum(pred * ohe_target)
        soft_dice = (intersection + self.smooth) / (torch.sum(pred) +
                                                         torch.sum(ohe_target) - intersection + self.smooth)
        return 1 - soft_dice

    @staticmethod
    def create_one_hot(target, classes):
        one_hot_encoding = torch.zeros(target.shape[0], classes, target.shape[2], target.shape[3],
                                       target.shape[4], dtype=torch.float32).cuda()
        for i in range(classes):
            one_hot_encoding[:, i, :, :, :] = (target[:, 0] == i).float()
        return one_hot_encoding


class DC_and_CE_loss(nn.Module):
    def __init__(self, smooth, classes, weight_ce=1, weight_dice=1):
        super(DC_and_CE_loss, self).__init__()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.dc = SoftDiceLoss(smooth, classes)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        '''

        :param pred:  (b, c, d, h, w) network output
        :param target: (b, 1, d, h, w) original_seg_arr (b, 1, d, h, w)
        :return:
        '''
        dc_loss = self.dc(pred, target)
        # print(pred.shape)
        # print( target[:, 0].shape)
        ce_loss = self.ce(pred, target[:, 0].long())
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


if __name__ == '__main__':
    pred = torch.randn(2, 3, 128, 128, 128)
    target = torch.randint(0, 1, (2, 1, 128, 128, 128))  # 3分类
    l = DC_and_CE_loss(1e-5, 3)
    print(l(pred, target))
