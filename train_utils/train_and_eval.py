import os
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from tqdm import tqdm

import disturtd_utils as utils


def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes + 1)
    data_loader = tqdm(data_loader)
    mask = None
    predict = None
    with torch.no_grad():
        for image, target in data_loader:
            image, target = image.to(device), target.to(device)
            # (B,1,H,W)
            output = model(image)
            truth = output.clone()
            output[output >= 0.5] = 1
            output[output < 0.5] = 0
            confmat.update(target.flatten(), output.long().flatten())
            # dice.update(output, target)
            mask = target.flatten() if mask is None else torch.cat((mask, target.flatten()))
            # 它是概率集合，不能是0，1集合
            predict = truth.flatten() if predict is None else torch.cat((predict, truth.flatten()))

    mask = mask.cpu().numpy()
    predict = predict.cpu().numpy()
    assert mask.shape == predict.shape, f"维度不对"
    AUC_ROC = roc_auc_score(mask, predict)

    # fpr, tpr, thresholds = roc_curve(mask, predict)
    # plt.figure()
    # plt.plot(fpr, tpr, '-', label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC)
    # plt.title('ROC curve')
    # plt.xlabel("FPR (False Positive Rate)")
    # plt.ylabel("TPR (True Positive Rate)")
    # plt.legend(loc="lower right")
    # plt.savefig("ROC.png")

    return confmat.compute()[0], confmat.compute()[1], confmat.compute()[2], confmat.compute()[3], confmat.compute()[
        4], AUC_ROC


def train_one_epoch(model, optimizer, data_loader, device, epoch,
                    scaler=None):
    model.train()
    total_loss = 0

    data_loader = tqdm(data_loader)
    for image, target in data_loader:
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            # output的输出通道为1
            target = target.unsqueeze(1).float()
            loss = nn.BCELoss()(output, target)
        total_loss += loss.item()

        data_loader.set_description(f"Epoch[{epoch}/150]-train,train_loss:{loss.item()}")
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

    return total_loss / len(data_loader)


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
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


if __name__ == "__main__":
    path = "../DRIVE/aug/images"
    print(len(os.listdir(path)))
