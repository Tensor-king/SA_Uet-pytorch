# SA_Uet-pytorch

* test-results:To save the resulting curve
* compute_mean_std:Normalized input image

## Precision

| SE     | SP | ACC | AUC | F1 |
|--------| :---: | :---: | :---: | :---: |
| 0.8234 |0.9840 |0.9708|0.9872 |0.8960|

## Qusetion

* The original author did not use mask, but directly used 1st_ manual
* Data enhancement: the author code generates 3 * 3 * 20=180 pictures, and the author writes 256. I don't know what the
  situation is
* Only drive datasets were tested


