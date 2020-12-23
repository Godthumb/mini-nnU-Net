import torch.nn as nn
import torch
import torch.nn.functional as F

class SoftBinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(SoftBinaryDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        '''
        :param pred:  (b, d, h, w)
        :param target: original_seg_arr (b, d, h, w)
        :return:
        '''
        b = pred.shape[0]
        pred = pred.view(b, -1)
        target = target.view(b, -1)
        intersection = pred * target
        soft_dice = (2 * intersection.sum(1) + self.smooth) / (pred.sum(1) + target.sum(1) + self.smooth)
        soft_dice = soft_dice.sum() / b
        return 1 - soft_dice

# ignore background
class SoftMultiDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5, classes=3, ignore_classes_index=None):

        super(SoftMultiDiceLoss, self).__init__()
        self.smooth = smooth
        self.classes = classes
        self.ignore_classes_index = ignore_classes_index
        self.bce_dice_loss = SoftBinaryDiceLoss()

    def forward(self, pred, target):
        '''

        :param pred:  (b, c, d, h, w)
        :param target: original_seg_arr (b, 1, d, h, w), will be translate to one_hot for loss computing
        :return:
        '''

        pred = F.softmax(pred, dim=1)  # (b, c, d, h, w)
        ohe_target = self.create_one_hot(target, self.classes) # (b, 1, d, h, w) => (b, c, d, h, w)

        total_dice_loss = 0
        for c in range(self.classes):
            if c != self.ignore_classes_index:
                total_dice_loss += self.bce_dice_loss(pred[:, c], ohe_target[:, c])
        return total_dice_loss / (ohe_target.shape[1] - 1) if self.ignore_classes_index != None \
            else total_dice_loss / ohe_target.shape[1]

    @staticmethod
    def create_one_hot(target, classes):
        one_hot_encoding = torch.zeros(target.shape[0], classes, target.shape[2], target.shape[3],
                                       target.shape[4], dtype=torch.float32).cuda()
        for i in range(classes):
            one_hot_encoding[:, i, :, :, :] = (target[:, 0] == i).float()
        return one_hot_encoding


class DC_and_CE_loss(nn.Module):
    def __init__(self, smooth, classes, weight_ce=1, weight_dice=1, ignore_classes_index=None):
        super(DC_and_CE_loss, self).__init__()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.dc = SoftMultiDiceLoss(smooth, classes, ignore_classes_index)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        '''

        :param pred:  (b, c, d, h, w) network output
        :param target: (b, 1, d, h, w) original_seg_arr
        :return:
        '''
        dc_loss = self.dc(pred, target)
        # print(pred.shape)
        # print( target[:, 0].shape)
        ce_loss = self.ce(pred, target[:, 0].long())
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result

class MultipleOutputLoss2(nn.Module):
    def __init__(self, loss, weight_factors=None):
        """
        use this if you have several outputs and ground truth (both list of same len) and the loss should be computed
        between them (x[0] and y[0], x[1] and y[1] etc)
        :param loss:
        :param weight_factors:
        """
        super(MultipleOutputLoss2, self).__init__()
        self.weight_factors = weight_factors
        self.loss = loss

    def forward(self, x, y):
        if self.weight_factors is None:
            weights = [1] * len(x)
        else:
            weights = self.weight_factors

        l = weights[0] * self.loss(x[0], y)  # weights [0, 0, 0, 1]
        for i in range(1, len(x)):
            if weights[i] != 0:
                l += weights[i] * self.loss(x[i], y)
        return l


if __name__ == '__main__':
    pred = torch.randn(2, 3, 128, 128, 128)
    target = torch.randint(0, 1, (2, 1, 128, 128, 128))  # 3分类
    l = DC_and_CE_loss(1e-5, 3)
    print(l(pred, target))
