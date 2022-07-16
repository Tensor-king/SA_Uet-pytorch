import os

import torch
import torch.utils.data
import wandb
from torch.optim import lr_scheduler

import compute_mean_std
import transforms as T
from datasets import Chasedb1Datasets
from model.SA_Unet import SA_UNet
from train_utils.train_and_eval import train_one_epoch, evaluate


class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, vflip_prob=0.5,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        # min_size = int(0.5 * base_size)
        # max_size = int(1.2 * base_size)
        #
        # trans = [T.RandomResize(min_size, max_size)]  # 最小边为base_size

        trans = []
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


def get_transform(train, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), flag=True):
    base_size = 565
    crop_size = 1008

    if train:
        return SegmentationPresetTrain(base_size, crop_size, mean=mean, std=std)
    else:
        return SegmentationPresetEval(crop_size, mean=mean, std=std)


def create_model(num_classes):
    model = SA_UNet(in_channels=3, num_classes=num_classes, base_c=16)
    return model


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    # segmentation nun_classes + background
    num_classes = args.num_classes

    # using compute_mean_std.py   自己计算数据集得到的mean和std
    mean, std = compute_mean_std.compute()

    # DRIVE
    # train_dataset = DriveDataset(args.data_path,
    #                              train=True,
    #                              transforms=get_transform(train=True, mean=mean, std=std, flag=args.dataset))
    #
    # val_dataset = DriveDataset(args.data_path,
    #                            train=False,
    #                            transforms=get_transform(train=False, mean=mean, std=std, flag=args.dataset))

    # ChaseDB1
    train_dataset = Chasedb1Datasets(args.data_path,
                                     train=True,
                                     transforms=get_transform(train=True, mean=mean, std=std))

    val_dataset = Chasedb1Datasets(args.data_path,
                                   train=False,
                                   transforms=get_transform(train=False, mean=mean, std=std))

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=8,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             num_workers=num_workers,
                                             pin_memory=True)

    model = create_model(num_classes=num_classes)
    model.to(device)

    params_to_optimize = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.Adam(
        params_to_optimize,
        lr=args.lr
    )

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # 创建学习率更新策略
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[100], gamma=0.1)

    # (Initialize logging)
    experiment = wandb.init(project='SA_UNet_2022')

    # 创建传输数据和文件的端口
    model_artifact = wandb.Artifact(
        "SA_UNet_2022", type="model",
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

    trigger = 0
    best_metric = {"AUC_ROC": 0.5}
    for epoch in range(args.start_epoch, args.epochs + 1):
        mean_loss = train_one_epoch(model, optimizer, train_loader, device, epoch,
                                    scaler=scaler)
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        experiment.log({
            "train_loss": mean_loss,
            'lr': lr,
            "epoch": epoch
        })

        acc, se, sp, F1, pr, AUC_ROC = evaluate(model, val_loader, device=device, num_classes=num_classes)
        experiment.log({
            "Precision": pr,
            "acc": acc,
            "sensitivity": se,
            "specificity": sp,
            "F1-score": F1,
            "AUC_ROC": AUC_ROC,
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

        if epoch == args.epochs or trigger >= args.early_stop:
            model_artifact.add_file('best_model.pth')
            experiment.log_artifact(model_artifact)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch SA-UNET training")
    parser.add_argument("--data-path", default="CHASEDB1", help="DRIVE root")
    # exclude background
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=4, type=int)
    parser.add_argument("--epochs", default=150, type=int, metavar="N",
                        help="number of total epochs to train")

    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    # parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
    #                     help='momentum')
    # parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
    #                     metavar='W', help='weight decay (default: 1e-4)',
    #                     dest='weight_decay')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--early_stop', default=50, type=int)

    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use pytorch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
