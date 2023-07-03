import torch.distributed as dist

import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from tqdm import tqdm

import train_utils.disturtd_utils as utils
from train_utils.dice_cofficient_loss import dice_coeff
# from focal_loss import FocalLoss


def reduce_value(value, average=True):
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
    else:
        world_size = 1
    if world_size < 2:  # 单GPU的情况
        return value

    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value /= world_size

        return value


def criterion(inputs, target, args):
    loss1 = 0

    # 为BCE设置权重
    weight = (torch.ones_like(target).float() * args.weight).cuda()
    weight[target == 0] = 1

    if type(inputs) is list:
        if args.sigmoid:
            loss2 = sum([nn.BCELoss(weight=weight)(i, target) for i in inputs])
        else:
            loss2 = sum([nn.BCEWithLogitsLoss(weight=weight)(i, target) for i in inputs])
    else:
        if args.sigmoid:
            loss2 = nn.BCELoss(weight=weight)(inputs, target)
        else:
            loss2 = nn.BCEWithLogitsLoss(weight=weight)(inputs, target)

    if args.dice:
        if type(inputs) is list:
            if not args.sigmoid:
                inputs = [torch.nn.Sigmoid()(i) for i in inputs]
            loss1 = sum([1 - dice_coeff(i, target) for i in inputs])
        else:
            if not args.sigmoid:
                inputs = torch.nn.Sigmoid()(inputs)
            loss1 = 1 - dice_coeff(inputs, target)
    return loss1 + loss2


def criterion_focal(inputs, target, args):
    loss_f = FocalLoss(alpha=[1, args.weight], smooth=False)
    if type(inputs) is list:
        inputs = [torch.nn.Softmax(dim=1)(i) for i in inputs]
        loss2 = sum([loss_f(i, target) for i in inputs])
    else:
        inputs = torch.nn.Softmax(dim=1)(inputs)
        loss2 = loss_f(inputs, target)

    return loss2


def evaluate(model, data_loader, num_classes, args):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    data_loader = tqdm(data_loader)
    mask = None
    predict = None
    dice_c = 0
    with torch.no_grad():
        for image, target in data_loader:
            image, target = image.cuda(), target.cuda()
            # (B,1,H,W)
            output = model(image)
            if type(output) is list:
                output = output[0]
            if not args.sigmoid and not args.use_focal:
                output = torch.sigmoid(output)
            if args.use_focal:
                output = nn.Softmax(dim=1)(output)
                output = output[:, 1:, :, :]
            truth = output.clone()
            output[output >= 0.5] = 1
            output[output < 0.5] = 0
            confmat.update(target.flatten(), output.long().flatten())
            dice_c += dice_coeff(output, target)
            mask = target.flatten() if mask is None else torch.cat((mask, target.flatten()))
            # 它是概率集合，不能是0，1集合
            predict = truth.flatten() if predict is None else torch.cat((predict, truth.flatten()))

    mask = mask.cpu().numpy()
    predict = predict.cpu().numpy()
    assert mask.shape == predict.shape, f"维度不对"
    AUC_ROC = roc_auc_score(mask, predict)

    return confmat.compute()[0], confmat.compute()[1], confmat.compute()[2], confmat.compute()[3], confmat.compute()[
        4], AUC_ROC, dice_c / len(data_loader)


def train_one_epoch(model, optimizer, data_loader, epoch, scheduler, args, scaler
                    ):
    model.train()
    total_loss = 0
    data_loader = tqdm(data_loader)
    for image, target in data_loader:
        image, target = image.cuda(), target.cuda()
        # 适用于BCE损失函数的标签形状 B 1 H W
        target = target.unsqueeze(1).float()
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            if not args.use_focal:
                loss = criterion(output, target, args)
            else:
                loss = criterion_focal(output, target, args)
        optimizer.zero_grad()

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

        data_loader.set_description(
            f"Epoch:{epoch}/{args.epochs}  train_loss:{loss.item()}")

        scheduler.step()
    return total_loss / len(data_loader)


def create_lr_scheduler(optimizer,
                        num_step: int,  # 一个epoch可以计算多少次batch
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-4):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        这里的相当于step
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
