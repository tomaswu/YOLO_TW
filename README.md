# _TW

## GOAL
Commplete my yolo based on yolov3 and yolo v7.

## LOSS (v3)

$$lbox = \lambda_{coord} \sum^{s^2}_{i=0} \sum^B_{j=0}I_{i,j}^{obj} [se(x_i)+se(y_i)+se(\sqrt{w_i})+se(\sqrt{h_i})] \tag{1}$$

$$lobj = \sum^{s^2}_{i=0} \sum^B_{j=0} I_{i,j}^{obj} {\rm BCE}(c_{i,j}) +\lambda_{noobj} \sum^{s^2}_{i=0} \sum^B_{j=0}I_{i,j}^{obj} {\rm BCE}(c_{i,j}) \tag{2}$$

$$lcls = \lambda_{class} \sum^{s^2}_{i=0} I_{i,j}^{obj} \sum_{c \in classes} {\rm BCE}(p_{i,j})   \tag{3}$$

$$loss=lbox+lcls+lobj \tag{4}$$
其中，
+ $S$: 代表grid size,$S^2$代表13x13,26x26, 52x52
+ $B$:  box
+ $I_{i,j}^{obj}$: 如果在$i$,$j$处的box有目标，其值为1，否则为0，由anchor的置信度及阈值决定
+ $I_{i,j}^{noobj}$: 如果在i,j处的box没有目标，其值为1，否则为0，$I_{i,j}^{noobj}=1-I_{i,j}^{obj}$
+ $se$: 平方误差 $se(g(x))=(g(x)-g(\hat{x}))^2$
+ $C$: 是否含有目标，confidence
+ $p$: 分类预测，onehot
+ BCE: $${\rm BCE}(x) = - \hat x\log(x)-(1-\hat x)\log(1-x)$$

