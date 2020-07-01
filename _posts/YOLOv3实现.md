---
title: YOLOv3实现
date: 2020-03-20 21:52:28
tags:
categories: Notebook
mathjax: true
---

# 搭建网络

## 概览

关于YOLOv3的网络架构，参阅[这里](../YOLO/#网络架构). 为方便，下面再次展示YOLOv3的网络架构图。

{% asset_img v2-d2596ea39974bcde176d1cf4dc99705e_r.jpg %}

打开`yolov3.cfg`文件后，我们可以看到5种类型的Layer

```
[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky
```

注意：这里的`pad`和`batch_normalize`是bool类型

```
[shortcut]
from=-3
activation=linear
```

`shotcut`层是跳过连接(skip connection)，类似于ResNet中使用的连接。 from参数为-3，表示`shortcut`层的输出是通过将`shortcut`层的前一层和前面的第三层的特征图相加得到的。

```
[upsample]
stride=2
```

对前一层的特征图应用双线性上采样，采样因子为`stride`.

```
[route]
layers = -1, 61

[route]
layers = -1, 61
```

`route`层具有一个`layers`属性，它可以具有一个或两个值。当`layers`属性只有一个值时，它会输出由该值索引的层的特征图。在我们的示例中，它是-4，因此该层将输出位于`route`层前面的第4层的特征图。当层有两个值时，它会返回由其值所索引的层的特征图的连接。在我们的例子中，它是-1,61，该层输出来自前一层（-1）和第61层的特征图，它们沿着深度维度进行连接。

```
[yolo]
mask = 0,1,2
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
classes=80
num=9
jitter=.3
ignore_thresh = .7
truth_thresh = 1
random=1
```

有3个`yolo`层，其输入分别是上图中网络的输出——3种尺寸的特征图。`anchors`描述了9个box priors，但仅使用由`mask`索引的box priors. 

## 解析配置文件

我们要利用`yolov3.cfg`文件中的信息，创建相应的Layer `nn.Module`，并按网络拓扑拼接形成`nn.ModuleList`. 第一步就是解析配置文件。

```python
def parse_cfg(cfgfile):
    """
    解析cfg文件
    将每个block存储为字典。block的属性及其值在字典中作为键值对存储。
    把block添加到列表blocks中
    返回blocks
    """
    file = open(cfgfile,'r')
    lines = file.read().split('\n')#把cfg文件按行存储在list中
    lines = [x for x in lines if len(x) > 0]#去掉空行
    lines = [x for x in lines if x[0] != '#']#去掉注释
    lines = [x.rstrip().lstrip() for x in lines]#删除行首行末的空格

    block = {}
    blocks = []

    for line in lines:
        if line[0] == '[' :#新的block开始
            if len(block) != 0:
                blocks.append(block)#新的block开始，把上一个block添加到list中
                block = {}#初始化新的block
            block["type"] = line.lstrip('[').rstrip(']')

            if block["type"] == "convolutional":
                block["batch_normalize"] = 0
        else:
            key, value = line.split('=')
            block[key.rstrip()] = value.lstrip()

    if block:
        blocks.append(block)
    file.close()
    return blocks
```

## 创造构建Layer

现在我们将使用上述`parse_cfg`返回的列表，为配置文件中存在的Layer构造PyTorch模块(`nn.Module`).

```python
def create_network(blocks):
    module_list = nn.ModuleList()
    net_info = blocks[0]

    #上一个卷积层的filter数量，也就是当前卷积层的输入通道数
    prev_filters = 3    #3是输入通道数，即RGB
    #前面所有输出层的filter数量
    output_filters = [] #不包括输入通道数3

    for index, block in enumerate(blocks[1:]):
        module = nn.Sequential()
        #check the type of block
        #create a new module for the block
        #append to module_list
        if block["type"] == "convolutional":
            #get info about this block
            activation = block["activation"]
            batch_normalize = int(block["batch_normalize"])
            filters = int(block["filters"])
            padding = int(block["pad"])#NOTE:cfg里的pad是bool类型
            kernel_size = int(block["size"])
            stride = int(block["stride"])

            #TODO:这里是什么意思？论文中有提到过吗？
            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            #TODO:Conv层的bias是True还是False?
            if batch_normalize:
                module.add_module('conv_{}'.format(index), nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = False))
                module.add_module('bn_{}'.format(index), nn.BatchNorm2d(filters))
            else:
                module.add_module('conv_{}'.format(index), nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = True))
            if activation == "leaky":
                module.add_module('leaky_{}'.format(index), nn.LeakyReLU(0.1, inplace=True)) 
            prev_filters = filters
            output_filters.append(prev_filters)

        elif block["type"] == "upsample":
            stride = int(block["stride"])
            #这里的stride是Unsample的scale_factor
            #TODO:mode是nearest还是bilinear?
            module.add_module("upsample_{}".format(index), nn.Upsample(scale_factor=stride, mode='nearest'))
            output_filters.append(prev_filters)
        
        elif block["type"] == "route":
            layers = block["layers"].split(',')
            start = int(layers[0])
            try:
                end = int(layers[1])
            except:
                end = 0
            #convert positive annotation to negative annotation
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index
            route = EmptyLayer()
            module.add_module("route_{}".format(index), route)
            if end < 0: #如果end存在
                #concatenate feature maps
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]
            prev_filters = filters
            output_filters.append(prev_filters)

        elif block["type"] == "shortcut":
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)
            output_filters.append(prev_filters)

        #YOLO is the detection layer
        elif block["type"] == "yolo":
            mask = block["mask"].split(',')
            mask = [int(x) for x in mask]

            anchors = block["anchors"].split(',')
            anchors = [int(x) for x in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            num_classes = int(block["classes"])
            img_size = net_info["height"]

            yolo_layer = YoloLayer(anchors, num_classes, img_size)
            module.add_module("Yolo_{}".format(index), yolo_layer)
            output_filters.append(prev_filters)

        module_list.append(module)
    
    return net_info, module_list
```

我们有一个名为`EmptyLayer`的新层，顾名思义就是一个空层。

```python
route = EmptyLayer()

shortcut = EmptyLayer()
```

它被定义为

```python
class EmptyLayer(nn.Module):
    #Placeholder for route and shortcut layers
    def __init__(self):
        super(EmptyLayer, self).__init__()
```

现在，空层可能看起来很奇怪，因为它什么都不做。`route`层，就像任何其他层一样执行操作（使用前面的层/连接）。在PyTorch中，当我们定义一个新层时，它继承`nn.Module`，在`nn.Module`对象的`forward`函数写入层执行的操作。

为了设计`route` Layer，我们构建一个`nn.Module`对象。然后，我们可以在`forward`函数中编写代码来连接/获取特征图。最后，我们在网络的`forward`函数中执行该层的操作。

但是，如果连接代码相当简短（在特征图上调用`torch.cat`），那么设计一个如上所述的层将导致不必要的抽象，这只会增加代码。我们可以做一个空层来代替提出的`route`层，然后直接在`Darknet`的`nn.Module`对象的`forward`函数中执行连接。

## 创造构建YOLO Layer

建立`yolo`层：

```python
class YoloLayer(nn.Module):
    def __init__(self, anchors, num_classes, img_dim=416):
        super(YoloLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.ignore_thres = 0.5 
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.obj_scale = 1 
        self.noobj_scale = 100
        self.metrics = {}
        self.img_dim = img_dim
        self.grid_size = 0
```



`yolo`层涉及到[损失函数](../YOLO/#损失函数-1)的计算。为方便起见，这里再给出计算损失函数的公式：

<p>
$$
\begin{align*}
{\rm loss}=&\lambda_{\rm coord} \sum_{i=0}^{S_1\times S_1} \sum_{j=0}^{B}\mathbb{1}_{ij}^{\rm obj} [(t_{x_i}-\hat t_{x_i})^2 + (t_{y_i} - \hat t_{y_i})^2]\\
+& \lambda_{\rm coord} \sum_{i=0}^{S_1\times S_1} \sum_{j=0}^{B}\mathbb{1}_{ij}^{\rm obj} [(t_{w_i}-\hat t_{w_i})^2 + (t_{h_i} - \hat t_{h_i})^2]\\
-& \lambda_{\rm obj} \sum_{i=0}^{S_1\times S_1} \sum_{j=0}^{B}\mathbb{1}_{ij}^{\rm obj}
\log(c_{ij})
- \lambda_{\rm noobj} \sum_{i=0}^{S_1\times S_1} \sum_{j=0}^{B}\mathbb{1}_{ij}^{\rm noobj}
\log(1-c_{ij})\\
-& \lambda_{\rm class} \sum_{i=0}^{S_1\times S_1} \sum_{j=0}^{B}\mathbb{1}_{ij}^{\rm obj} \sum_{c\in {\rm classes}} [\hat p_{ij}(c)\log(p_{ij}(c)) + (1-\hat p_{ij}(c))\log(1-p_{ij}(c))]
\end{align*}
$$
</p>

由公式容易得到代码：

```python
def forward(self, x, targets=None, img_dim=None):
    #x is feature map, of shape N, C, H, W
    
    ###
    loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
    loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
    loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
    loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
    loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
    loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
    loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
    loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
    total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
    ###
```

现在的问题是$\mathbb{1}_{ij}^{\rm obj}$，也就是代码中的`obj_mask`如何求。我们这里调用`build_target`函数：

```python
iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
    pred_boxes=pred_boxes, pred_cls=pred_cls, target=targets,
    anchors=self.scaled_anchors, ignore_thres=self.ignore_thres
)
```

`pred_boxes`和`pred_cls`是从`forward`函数的输入`x`中提取的：

```python
prediction = (#(N, C, H, W) -> (N, #anchors, 5+#classes, H, W)
    x.view(num_samples, self.num_anchors, self.num_classes+5, grid_size, grid_size)\
        .permute(0, 1, 3, 4, 2).contiguous()#->(N, #anchors, H, W, 5+#classes)
)

#Get outputs
#x, y, w, h ,pred_conf of shape [N, #anchors, H, W]
x = torch.sigmoid(prediction[..., 0])   #center x
y = torch.sigmoid(prediction[..., 1])   #center y
w = prediction[..., 2]  #width
h = prediction[..., 3] #height
pred_conf = torch.sigmoid(prediction[..., 4])   #conf
#pred_cls of shape [N, #anchors, H, W, #classes]
pred_cls = torch.sigmoid(prediction[..., 5:])   #cls pred

#If grid size does not match current we compute new offsets
if grid_size != self.grid_size:
    self.compute_grid_offsets(grid_size, cuda=x.is_cuda)

#Add offset and scale with anchors
#pred_boxes of shape (N, #anchors, H, W, 4)
pred_boxes = FloatTensor(prediction[..., :4].shape)
pred_boxes[..., 0] = x.data + self.grid_x
pred_boxes[..., 1] = y.data + self.grid_y
pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h

#output of shape (N, #anchors*H*W, 5+#classes)
output = torch.cat(
    (
        pred_boxes.view(num_samples, -1, 4) * self.stride,  #N, #anchors*H*W, 4
        pred_conf.view(num_samples, -1, 1),                 #N, #anchors*H*W, 1
        pred_cls.view(num_samples, -1, self.num_classes),   #N, #anchors*H*W, #classes
    ),
    -1,
)
```

其中，`Add offset and scale with anchors`部分的公式参见[这里](../YOLO/#Direct-location-prediction)。公式中的$c_x,c _y$，也就是`self.grid_x, self.grid_y`是如何计算的呢？

```python
def compute_grid_offsets(self, grid_size, cuda=True):
    self.grid_size = grid_size
    g = self.grid_size
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    self.stride = self.img_dim / self.grid_size
    #Calculate offsets for each grid
    self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)
    self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)
    self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])
    self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
    self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))
```

> ```python
> >>> torch.arange(5)
> tensor([ 0,  1,  2,  3,  4])
> ```
>
> `torch.Tensor.repeat`: Repeats this tensor along the specified dimensions.
>
> ```python
> >>> x = torch.tensor([1, 2, 3])
> >>> x.repeat(4, 2)
> tensor([[ 1,  2,  3,  1,  2,  3],
>         [ 1,  2,  3,  1,  2,  3],
>         [ 1,  2,  3,  1,  2,  3],
>         [ 1,  2,  3,  1,  2,  3]])
> ```

好了，现在该看看`build_targets`函数是如何构建的了。

```python
def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres):
    ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

    #pred_boxes of shape [N, #anchors, H, W, 4(译码后的x, y, w, h)]
    #pred_cls of shape [N, #anchors, H, W, #classes]
    nB = pred_boxes.size(0)
    nA = pred_boxes.size(1)
    nC = pred_cls.size(-1)
    nG = pred_boxes.size(2)#grid size

    #Output tensors
    obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0)
    noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1)
    class_mask = FloatTensor(nB, nA, nG, nG).fill_(0)
    iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0)
    tx = FloatTensor(nB, nA, nG, nG).fill_(0)
    ty = FloatTensor(nB, nA, nG, nG).fill_(0)
    tw = FloatTensor(nB, nA, nG, nG).fill_(0)
    th = FloatTensor(nB, nA, nG, nG).fill_(0)
    tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)

    #Convert to position relative to box
    target_boxes = target[:, 2:6] * nG
    gxy = target_boxes[:, :2]
    gwh = target_boxes[:, 2:]
    #Get anchors with best iou
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
    best_ious, best_n = ious.max(0)
    #Separate target values
    b, target_labels = target[:, :2].long().t()
    gx, gy = gxy.t()
    gw, gh = gwh.t()
    gi, gj = gxy.long().t()
    #Set masks
    obj_mask[b, best_n, gj, gi] = 1
    noobj_mask[b, best_n, gj, gi] = 0

    #Set noobj mask to zero where iou exceeds ignore threshold
    for i, anchor_ious in enumerate(ious.t()):
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

    #Coordinates
    tx[b, best_n, gj, gi] = gx - gx.floor()
    ty[b, best_n, gj, gi] = gy - gy.floor()
    #Width and height
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
    #One-hot encoding of label
    tcls[b, best_n, gj, gi, target_labels] = 1
    #Compute label correctness and iou at best anchor
    class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
    iou_scores[b, best_n, gi, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)

    tconf = obj_mask.float()
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf
```

其中，`bbox_wh_iou`和`bbox_iou`都是计算两个bounding box之间的IoU的函数；`t()`是二维数组转置函数。

```python
def bbox_wh_iou(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area

def bbox_iou(box1, box2, x1y1x2y2=True):
    #TODO:box2不应该是bbox的数组吗？不应该是一个单独的box啊？

    if not x1y1x2y2:
        #Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        #Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    #Get the coordinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    #Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
        torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    #Union area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou
```

最后的问题是：<font color = purple>`build_targets`函数的参数`target`的维度是怎样的？</font> 我还没有搞清楚，不过`target`显然是从训练集中直接得到的数据。

## Darknet

```python
class Darknet(nn.Module):
    def __init__(self, cfgfile, img_size=416):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_network(self.blocks)
        self.img_size = img_size

    def forward(self, x, targets=None):
        #cache outputs for route layer. key:index value: feature map
        outputs = {}
        yolo_outputs = []
        loss = 0

        for index, block in enumerate(self.blocks[1:]):
            if block["type"] in ["convolutional", "upsample"]:
                x = self.module_list[index](x)
                outputs[str(index)] = x
            elif block["type"] ==  "route":
                layers = block["layers"].split(',')
                #convert negative annotation to positive annotation
                layers = [int(i) if int(i) > 0 else int(i)+index for i in layers]
                if len(layers) == 1:
                    x = outputs[str(layers[0])]
                    outputs[str(index)] = x
                else:
                    x1 = outputs[str(layers[0])]
                    x2 = outputs[str(layers[1])]
                    #x of shape(N,C,H,W), concatenate along dim C
                    x = torch.cat((x1, x2), dim=1)
                    outputs[str(index)] = x
            elif block["type"] == "shortcut":
                from_layer = int(block["from"])
                activation = block["activation"]
                #convert negative annotation to positive annotation
                from_layer = from_layer if from_layer > 0 else from_layer + index
                x1 = outputs[str(from_layer)]
                x2 = outputs[str(index-1)]
                #like residual block
                x = x1 + x2
                outputs[str(index)] = x
                #TODO:What about the activation???
            elif block["type"] == "yolo":
                img_dim = x.shape[2]
                x, layer_loss = self.module_list[index][0](x, targets, img_dim) #x of shape (N, #anchors*H*W, 5+#classes)
                loss += layer_loss
                yolo_outputs.append(x)
                outputs[str(index)] = x
        yolo_outputs = to_cpu(torch.cat(yolo_outputs, 1))   #yolo_outputs of shape (N, #feature_maps*#anchors*H*W, 5+#classes)
        return yolo_outputs if targets is None else (loss, yolo_outputs)
```

还记得我们为`shortcut`层和`route`层创建的`EmptyLayer`吗？它们的功能直接写在`Darknet`的`forward`函数中了。

# 检测

## 非极大值抑制

选取一个置信度阈值，过滤掉低阈值box，再经过NMS（非极大值抑制），就可以输出整个网络的预测结果了。

```python
def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    #predictions of shape (N, #feature_maps*#anchors*H*W, 5+#classes)

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):   #image_pred of shape (#feature_maps*#anchors*H*W, 5+#classes)
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 4] >= conf_thres] #shape (#conf>thres, 5+#classes)
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Object confidence times class confidence
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
        # Sort by it
        image_pred = image_pred[(-score).argsort()]
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)
        # Perform non-maximum suppression
        keep_boxes = []
        while detections.size(0):
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            label_match = detections[0, -1] == detections[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match #同类，且IoU大于阈值
            weights = detections[invalid, 4:5]
            # Merge overlapping bboxes by order of confidence TODO:为什么是merge？而不是保留conf最大的那一个
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)

    return output
```

注：本篇笔记参考、摘录了如下资料

[从零开始实现YOLO v3](https://zhuanlan.zhihu.com/p/36899263)

[eriklindernoren / PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3)