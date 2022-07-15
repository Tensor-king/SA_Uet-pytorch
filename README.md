# SA_Uet-pytorch

# test Results

* test-results:To save the resulting curve
* compute_mean_std:To Back one -time input image

## 精度

| SE     | SP | ACC | AUC | F1 |
|--------| :---: | :---: | :---: | :---: |
| 0.8234 |0.9840 |0.9708|0.9872 |0.8960|

## 问题

** 原作者未使用mask，直接使用的1st_manual
** 数据增强作者代码中最终产生3*3*20=180图片,作者写256，不知道为什么

