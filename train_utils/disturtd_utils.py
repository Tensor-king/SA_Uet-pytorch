import torch


class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    # 建立左True，右上的混淆矩阵
    def update(self, a, b):
        n = self.num_classes
        if self.mat is None:
            # 创建混淆矩阵
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
        with torch.no_grad():
            # 寻找GT中为目标的像素索引 忽略255(FOV外的像素值)
            k = (a >= 0) & (a < n)
            inds = n * a[k].to(torch.int64) + b[k]
            # 左为True,上为predict
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)

    def reset(self):
        if self.mat is not None:
            self.mat.zero_()

    def compute(self):
        h = self.mat.float()
        # 计算全局预测准确率(混淆矩阵的对角线为预测正确的个数)
        acc_global = (torch.diag(h).sum() / h.sum()).item()
        # 计算sensitivity(recall)
        se = ((torch.diag(h) / h.sum(1))[1]).item()
        # 计算specificity
        sp = ((torch.diag(h) / h.sum(1))[0]).item()
        # 计算precision
        pr = ((torch.diag(h) / h.sum(0))[1]).item()
        # 计算每个类别的F1-score组成的列表 从0开始
        F1 = 2 * (pr * se) / (pr + se)

        # # 计算类别预测与真实目标的iou
        # iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))

        return acc_global, se, sp, F1, pr
