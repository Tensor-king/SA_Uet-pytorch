## model

* SA-Unet

## Requirement

* torch--1.10.1
* wandb:Similar to tensorboard, record the result curve and save it to the cloud

## Precision

| Dataset  |   SE   |   SP   |  ACC   |  AUC   | F1     |
|----------|:------:|:------:|:------:|:------:|--------|
| DRIVE    | 0.8234 | 0.9840 | 0.9708 | 0.9872 | 0.8221 |
| CHASEDB1 | 0.8352 | 0.9885 | 0.9774 | 0.9917 | ------ |

## Be Careful

* The original author did not use mask, Instead, it crop the image to 592*592, which causes most of its precision to be high, and if you use mask, this network will only achieve average results
* Data enhancement: the author code generates 3 * 3 * 20=180 pictures, and the author writes 256. I don't know what the
  situation is.
* Please use the data enhancement script to augment your own dataset, put it in the aug folder, and then train the network.


