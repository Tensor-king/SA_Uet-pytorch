## model

* SA-Unet

## Requirement

* torch--1.10.1
* wandb:Similar to tensorboard, record the result curve and save it to the cloud

## Precision

| Dataset  |   SE   |   SP   |  ACC   |  AUC   | F1     |
|----------|:------:|:------:|:------:|:------:|--------|
| DRIVE    | 0.8234 | 0.9840 | 0.9708 | 0.9872 | 0.8960 |
| CHASEDB1 | 0.8352 | 0.9885 | 0.9774 | 0.9917 | 0.9138 |

## Qusetion

* The original author did not use mask, but directly used 1st_label
* Data enhancement: the author code generates 3 * 3 * 20=180 pictures, and the author writes 256. I don't know what the
  situation is
* For the chasedb1 dataset, I changed the learning rate decay strategy. Using the method adopted in this paper, I
  can't achieve that accuracy
* My accuracy is higher than that of the author. If the evaluation index is designed incorrectly, please contact me:
  zyf1787594682@163.com

## How to train

* Before training, please pay attention to the learning rate decay measurement, crop_Size, calculation of normalized
  mean and standard deviation, batch_ Size is 4 and 8

