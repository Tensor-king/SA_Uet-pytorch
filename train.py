import os
import random

import numpy as np
import torch
import torch.utils.data
import wandb

import compute_mean_std
import transforms as T
from datasets import DriveDataset, Chasedb1Datasets, StareDataset
from train_utils.train_and_eval import train_one_epoch, evaluate, create_lr_scheduler
from model.SA_Unet import SA_UNet
import torch.backends.cudnn


class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, vflip_prob=0.5,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.5 * base_size)
        max_size = int(2 * base_size)
        trans = [T.RandomResize(min_size, max_size)]  # 最小边为base_size
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:
            trans.append(T.RandomVerticalFlip(vflip_prob))
        trans.extend([
            T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self, crop_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(train, args, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    if args.dataset == "DRIVE":
        base_size = 565
        crop_size = 592
    elif args.dataset == 'CHASEDB1':
        base_size = 960
        crop_size = 1008
    elif args.dataset == 'STARE':
        base_size = 605
        crop_size = 512
    else:
        raise Exception("无效的数据集或者数据集字母需要大写")
    if train:
        return SegmentationPresetTrain(base_size, crop_size, mean=mean, std=std)
    else:
        return SegmentationPresetEval(crop_size, mean=mean, std=std)


def create_model():
    model = SA_UNet()
    return model


def setup_seed(seed=2022):
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def main(args):
    # 保证结果的可复现性
    setup_seed()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # segmentation nun_classes + background
    num_classes = args.num_classes

    # using compute_mean_std.py   自己计算数据集得到的mean和std
    mean, std = compute_mean_std.compute(args.mean_std)

    # DRIVE
    if args.dataset == 'DRIVE':
        train_dataset = DriveDataset(args.data_path,
                                     train=True,
                                     transforms=get_transform(train=True, args=args, mean=mean, std=std))

        val_dataset = DriveDataset(args.data_path,
                                   train=False,
                                   transforms=get_transform(train=False, args=args, mean=mean, std=std))
    elif args.dataset == 'CHASEDB1':
        # ChaseDB1
        train_dataset = Chasedb1Datasets(args.data_path,
                                         train=True,
                                         transforms=get_transform(train=True, args=args, mean=mean, std=std))

        val_dataset = Chasedb1Datasets(args.data_path,
                                       train=False,
                                       transforms=get_transform(train=False, args=args, mean=mean, std=std))
    elif args.dataset == "STARE":
        # Stare
        train_dataset = StareDataset(args.data_path,
                                     train=True,
                                     transforms=get_transform(train=True, args=args, mean=mean, std=std))

        val_dataset = StareDataset(args.data_path,
                                   train=False,
                                   transforms=get_transform(train=False, args=args, mean=mean, std=std))
    else:
        raise Exception("数据集错误或者需要大写")

    num_workers = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             num_workers=num_workers,
                                             pin_memory=True)

    model = create_model()
    model.to(device)

    params_to_optimize = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.Adam(
        params_to_optimize,
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    # 省显存,可能导致精度下降
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # 创建学习率更新策略
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[100], gamma=0.1)
    scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True, warmup_epochs=1)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=15, verbose=True, min_lr=0)
    # scheduler = lr_scheduler.MultiStepLR(optimizer, [50], gamma=0.1)

    # (Initialize logging)
    experiment = wandb.init(project=args.net)

    # 创建传输数据和文件的端口
    model_artifact = wandb.Artifact(
        args.net, type="model",
        description="hy-parameters and weights",
        metadata=vars(args))

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    # 用于决定多少次精度不增长的时候停止训练
    trigger = 0
    best_metric = {"AUC_ROC": 0.5}
    for epoch in range(args.start_epoch, args.epochs + 1):
        mean_loss = train_one_epoch(model, optimizer, train_loader, epoch, scheduler, args, scaler)

        acc, se, sp, F1, pr, AUC_ROC, dice = evaluate(model, val_loader, num_classes=num_classes,
                                                      args=args)

        # scheduler.step()
        experiment.log({
            "train_loss": mean_loss,
            "acc": acc,
            "sensitivity": se,
            "specificity": sp,
            "F1-score": F1,
            "AUC_ROC": AUC_ROC,
            "Dice": dice,
            "epoch": epoch
        })

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        if args.amp:
            save_file["scaler"] = scaler.state_dict()

        trigger += 1

        if AUC_ROC > best_metric["AUC_ROC"]:
            best_metric["AUC_ROC"] = AUC_ROC
            torch.save(save_file, "best_model.pth")
            trigger = 0

        if trigger >= args.early_stop or epoch == args.epochs:
            model_artifact.add_file('best_model.pth')
            experiment.log_artifact(model_artifact)
            break


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="hy-Pararmeters")
    parser.add_argument("--data-path", default="DRIVE", help="DRIVE root")
    parser.add_argument("--dataset", default="DRIVE", help="which dataset to train")
    parser.add_argument("--mean_std", default="DRIVE/aug/images", help="DRIVE root")
    # exclude background
    parser.add_argument("--num_classes", default=2, type=int)
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch_size", default=8, type=int)
    parser.add_argument("--epochs", default=150, type=int, metavar="N",
                        help="number of total epochs to train")

    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    # parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
    #                     help='momentum')
    parser.add_argument('--wd', '--weight_decay', default=0.0, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--early_stop', default=15, type=int)
    parser.add_argument('--loss_function', default="BCE and  dice", type=str)
    parser.add_argument('--dice', default=False, type=bool)
    parser.add_argument('--weight', default=1.5, type=int)
    parser.add_argument('--amp', default=False, type=bool)
    parser.add_argument('--sigmoid', default=False, type=bool)
    parser.add_argument("--net", default='SA_UNet', type=str)
    parser.add_argument("--use_focal", default=False, type=bool)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args_ = parse_args()
    main(args_)
