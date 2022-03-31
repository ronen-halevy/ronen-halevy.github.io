# A Guide To YOLOv3

## Introduction to Object Detection

The task of a CNN object detection model is dual: It provides both classifies objects within an image to dataset labels, and also provides an estimation to objects' bounding boxes locations. The diagram below illustrates an input image on the left, and classification with bounding box annotations results on the right.

**Animation: Image Class and Bounding Box Annotations**

<img src="https://github.com/ronen-halevy/ronen-halevy.github.io/blob/master/assets/images/yolo/image-shapes-annotations.gif" alt="drawing" width="100%"/>




A detection model normally outputs 2 vectors per each detected object:

- A Classification output vector, with estimated probabilities of each dataset label. Vector length is \\(N_classes\\), i.e. number of classes in sdataset. Decision is mostly taken by applying a softmax operator on the vector.
- A vector with the predicted location of a bounding box which encloses the object. The location can be represented in various formats as illustrated in the diagram below.

**Representation Formats:  (1) Bbox Vertices. (2) Bbox Center + Dimenssions**: 

![alt text](https://github.com/ronen-halevy/ronen-halevy.github.io/blob/master/assets/images/yolo/circle-image.jpg)

## Object Detection Models

YOLO was indeed an innovative break-through in the field of CNN Object Detection algorithims. This section briefly reviews 3 object detection algorithms before YOLO. 

### Plain CNN Model

This is a conventional CNN classification model, but classification output stage is now enhanced by a regression predictor for the prediction of a bounding box. Implementation is simple. However, it is limitted to a detection of a single object.

The illustrative diagram which follows presents an image with 3 shape objects. The model, at the best case, will detect one of the object shapes only. 
Such detection models, with a detection capability of a sinlge object are often reffered to as `Object Localization` models.

**Figure: Plain CNN Model**

![alt text](https://github.com/ronen-halevy/ronen-halevy.github.io/blob/master/assets/images/yolo/image-classification.jpg)


### Sliding Window Model

To address the single object detectio0n limitation, CNN is repeatedly activated inside the bounderies of a window, as it  slides along the image as illustrated in the animation diagram below.

To fit various object sizes, multiple window sizes should be activated, as illustrated in the animation. Alternatively,  (or in addition to), the sliding window should run over multiple scaleds of the image.

Location can be determined by window's region, and the offset of the bounding box within the sliding window position.

**Figure: Sliding Window Animation**

![alt text](https://github.com/ronen-halevy/ronen-halevy.github.io/blob/master/assets/images/yolo/sliding-window-detection.gif)

There are drawbacks to this model: The repeated deployment of the CNN model per each window position, constraints a heavy computation load. But not only that - since convolution span regions are limitted by window's size and position, which may be uncorrelated with image's regions of interest postion and sizes, objects may be cropped or missed by the model.

### R-CNN

R-CNN ([by Ross Girshick et al, UC Berkely, 2014](https://arxiv.org/abs/1311.2524)), which stands `Regions with CNN features`, addresses drawbacks of `Sliding Window` model. 
The idea of R-CNN in essence is of a 3 steps process:
1. Extract region proposals - 2000 regions were stated n original paper. The farther process is limitted to proposed regions. There are a number of algorithm which can make region proposals. The authors used [selective search by J. Uijlings, K. van de Sande, T. Gevers, and A. Smeulders.Selective search for object recognition.IJCV, 2013.](https://www.researchgate.net/publication/262270555_Selective_Search_for_Object_Recognition)
2. Deploy CNN with bounding box regression over each proposed region.
3. Classify each region - originally using Linear SVM, in later model's variants e.g. `Fast R-CNN`, `Softmax` was deployed.

**Figure: Region Proposals**

![alt text](https://github.com/ronen-halevy/ronen-halevy.github.io/blob/master/assets/images/yolo/image-classification-rcnn.jpg)

This is just a brief description of the algorithm, which at that time, contributed to a dramatic improvement of CNN detection models performance. R-CNN was later followed by improvments variants such as FASTR-CNN, [Girshick, Ross. "Fast r-cnn." Proceedings of the IEEE international conference on computer vision, 2015](https://arxiv.org/abs/1504.08083), FASTRR-CNN, [Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun, 2016](https://arxiv.org/abs/1506.01497). These models aimed to address R-CNN problems, amongst are real time performance issues,long training time for the 2000 regions and region selection process.


## A Brief Introduction to YOLOv3

This article is about YOLO (`You Only Look Once`), and specifically its 3rd version YOLOv3. [YOLOv3: An Incremental Improvement, Joseph Redmon, Ali Farhadi, 2018](https://arxiv.org/abs/1804.02767)

As presented above, the common practive in various algorithms prior to YOLO, was to run CNN ver many regions. This ptactices' computation cost is high.

YOLO does segment the image as well. However, YOLO's way of segmenting the image is entirely different: instead of running CNN over thousands of regions seperately, it runs CNN only once over the entire image.
This is a huge difference, which makes YOLO so much faster - YOLOv3 is 1000 times faster than R-CNN.

So how can YOLO be so fast, while still segmenting the images?

Answer: YOLO functionally segments the image to a grid of cells. But rather than running CNN seperately on each cell, it runs CNN ones.  

YOLO's CNN predicts a detection descriptors to each grid cell. A descriptor combines 2 vectors:
1. A Classification result vector which holds a classification probabilty for each of the dataset's classes.
2. The Bounding Box Location $x_1, x1, w, ,h$, along with the Objective Prediction which indicates probabilty of an object resides in the Bbox.


Animation below illustrates Bounding Boxes parameters, which consist of the center location \\(c_x, c_y\\), and Width and Height.


**Gridded Image Animation:** Center location \\(c_x, c_y\\), Width and Height

![alt text](https://github.com/ronen-halevy/ronen-halevy.github.io/blob/master/assets/images/yolo/grid-image-shapes.gif)



Following is a diagram which illustrates a detection descriptor which YOLO predicts per a detected object. The descriptor consists of 5 words, 4 of which describe the bounding box location, then the Objective probability, and then N classes probabilities.

**Detection Descriptor**

![alt text](https://github.com/ronen-halevy/ronen-halevy.github.io/blob/master/assets/images/yolo/yolov3-single-output-cell-fields.jpg)

As a matter of fact, YOLOv3 supports not just a single detection per cell, but it supports 3 detections per a cell. 
Accordingly, YOLO's predicted descriptor per cell is as illustrated in the diagram below.


**Detection 3 Descriptors**


![alt text](https://github.com/ronen-halevy/ronen-halevy.github.io/blob/master/assets/images/yolo/yolov3%20output-cell-with-fields.jpg)


So that was the output for a single grid cell. 
But CNN assigns such a detection descriptor to each of the grid cells. Considering a 13x13 grid, the CNN output looks like this:

**YOLOv3 - CNN Output**

![alt text](https://github.com/ronen-halevy/ronen-halevy.github.io/blob/master/assets/images/yolo/yolov3%20output%20-cube-13.jpg)

### Grid Construction

So, how is the grid constructed?

The grid is constructed by passing the image thru a CNN, with a downsampling stride. Image size is 416*416*3, so assuming 
a 32 strides downsampling (which is actually the case), the output dimenssions would be a 13 x 13 x N box.
Where:

$N=3 x (5+N_{classes})$ 

That output structure is the illustrated above box diagram.

To enhence detection performance for smaller objects, YOLOv3 CNN generates output in 3 grid scales simultaneously: a 13 x 13 grid (as depicted above), and also 26 x 26 and 52 x 52 grids.

## YOLOv3 Block Diagrams

YOLOv3 Block Diagrams

The below block diagrams describe YOLOv3 `Forwarding` and `Training` operation.
Following chapters of this article present a detailed description of the 2 operation modes.

**YOLOv3 Block Diagram: Forwarding**


![alt text](https://github.com/ronen-halevy/ronen-halevy.github.io/blob/master/assets/images/yolo/YOLOv3-Forwarding.jpg)

**YOLOv3 Block Diagram: Training**

![alt text](https://github.com/ronen-halevy/ronen-halevy.github.io/blob/master/assets/images/yolo/YOLOv3-Training.jpg)

The next 2 chapters detail the functionality of Training and Forwarding modes diagrams, following the presented above block diagrams.

## YOLOv3 Training Functionality

This section details YOLOv3 Training functionality, following the presented above Training block diagram:

1. Training Dataset
2. Pre-Process Image
3. CNN Model and Decode
4. Loss Calculation
5. Gradient Descent Update

### 1. Training Dataset

The training dataset consists of both images examples and their related metadata. The metadata is created per each of the 3 scales.
The metadata is arranged to have the same structure like that of the detection descriptor presented in the YOLOv3 introduction section. 

For convininence the diagram is posted herebelow again:

**Training Arranged Metadata**

![alt text](https://github.com/ronen-halevy/ronen-halevy.github.io/blob/master/assets/images/yolo/yolov3%20output-cell-with-fields.jpg)


Let's illustrate the generation of that metadata with an example:

Herebelow is a training image example:

**Training Image Example with Bounding Box Attonations**

https://github.com/ronen-halevy/ronen-halevy.github.io/blob/master/assets/images/yolo/yolov3-input-image-example.jpg


The table below presents the 4 objects' metadata:

| # | x   | y   | w   | h   | objective | Class     |
|---|-----|-----|-----|-----|-----------|-----------|
| 1 | 104 | 144 | 112 | 64  | 1         | Trapezoid |
| 2 | 250 | 180 | 98  | 104 | 1         | Circle    |
| 3 | 120 | 272 | 108 | 77  | 1         | Hexagon   |
| 4 | 278 | 336 | 115 | 83  | 1         | Ellipse   |

To construct the training label records, some data arrangements should be taken:

**Class Data** - Should be arranged as a list of $N_{class}$ priorities. Here $N_{class}$ is 6, since dataset consists of 6 classes: Trapezoid, Circle Heagon, Ellipse, Square and Triangle.

So presentation should be of a one hot format like so:

Trapezoid: 1, 0, 0, 0, 0, 0
Circle:    0, 1, 0, 0, 0, 0
Hexagon:   0, 0, 1, 0, 0, 0
Square:    0, 0, 0, 1, 0, 0

Still, to improve performance, we apply `Label Smoothing`, as was proposed in `Rethinking the Inception Architecture for Computer Vision`by Szegedy et al in [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567).

with Label Smmothing, the one-hot probability of y given x, (marked by \\((p(y|x) =delta_{y,x}\\) ), is smoothed according the formula below:

**Label Smoothing Formula**

\\(p{_smoothed}(y|x)=(1-\epsilon)\delta_{y,x}+\epsilon  * u(y)\\)

Where:

- \\(\delta(y|x) \\) is the original one-hot probability
- \\(\epsilon\\) is the smoothing parameter, taken as \\(\epsilon=0.01\\) 
- \\(u(y)\\) is the distribution over lables, here assumed uniform distribution, i.e. \\(u(y)=\frac{1}{6}\\).

Pluging that in gives:

Trapezoid: 0.990125, 1.25e-4, 1.25e-4, 1.25e-4, 1.25e-4, 1.25e-4
Circle:    1.25e-4, 0.990125, 1.25e-4, 1.25e-4, 1.25e-4, 1.25e-4
Hexagon:   1.25e-4, 1.25e-4, 0.990125, 1.25e-4, 1.25e-4, 1.25e-4
Square:    1.25e-4, 1.25e-4, 1.25e-4, 0.990125, 1.25e-4, 1.25e-4


The training data is used for loss function computation for the 3 grid scales.
To make training data ready for this loss computations, we pack the training labels in 3 label arrays, each relate to a grid scale.

The diagram below shows a 13x13 grid over the image:

**Training Image with Attonations with a 13x13 Grid**

https://github.com/ronen-halevy/ronen-halevy.github.io/blob/master/assets/images/yolo/yolov3-input-image-example-grid.jpg


The coarse 13x13 grid diagram shows that the 4 objects are located at cells (3, 4), (7, 5), (3, 8) and (8, 10).

Table below presents the objects related cells per each of the 3 grids.

| #Grid Size | Cell Location Object #1 | Cell Location Object #2 | Cell Location Object #3 | Cell Location Object #4 |
|------------|-------------------------|-------------------------|-------------------------|-------------------------|
| 13x13      | 3, 4                    | 7, 5                    | 3, 8                    | 8, 10                   |
| 26x26      | 6, 8                    | 14, 10                  | 6, 16                   | 16, 20                  |
| 52x52      | 12, 16                  | 28, 20                  | 12, 32                  | 32, 40                  |



To make training data ready for loss computations, we pack it in 3 label arrays, with shape:

\\\text{labels.shape=Batch} *  \text{Grid Size} * N_{boxes} * (5+N_{classes})\\)

In our example:

\\\text{coarse-lables.shape=Batch} *  13*13*3 * 11\\)

\\\text{medium-lables.shape=Batch} * 26*26*3 * 11\\)

\\\text{fine-lables.shape=Batch} * 52*52*3 * 11\\)



Now let's fill in the data to the lable arrays:

**coarse-grid lables**

The network path to the coarse grid output passes through 32 strides, so accordingly the related grid cells indices are:

index_x1, index_y1 = int(104/32), int(144/32)
index_x2, index_y2 = int(250/32), int(180/32)
index_x3, index_y3 = int(120/32), int(272/32)
index_x4, index_y4 = int(278/32), int(336/32)

index_x1, index_y1 = 3, 4
index_x2, index_y2 = 7, 5
index_x3, index_y3 = 3, 8
index_x4, index_y4 = 8, 10

**medium-grid lables**

The network path to the coarse grid output passes through 16 strides, so accordingly the related grid cells indices are:

index_x1, index_y1 = 6, 8
index_x2, index_y2 = 14, 10
index_x3, index_y3 = 6, 16
index_x4, index_y4 = 16, 20


**fine-grid lables**

The network path to the coarse grid output passes through 16 strides, so accordingly the related grid cells indices are:

index_x1, index_y1 = 12, 16
index_x2, index_y2 = 28, 20
index_x3, index_y3 = 12, 32
index_x4, index_y4 = 32, 40

Let Batch=0:

coarse-grid-lables[0,3,4,0,:] =  (0,104,144,112,64,1,0.990125,1.25e-4,1.25e-4,1.25e-4,1.25e-4,1.25e-4)
coarse-grid-lables[0,7,5,0,:] = (0,250,180,98,104,1,0.990125,1.25e-4,1.25e-4,1.25e-4,1.25e-4,1.25e-4)
coarse-grid-lables[0,3,8,0,:] = (0,120,272,108,77,1,0.990125,1.25e-4,1.25e-4,1.25e-4,1.25e-4,1.25e-4)
coarse-grid-lables[0,8,10,0,:] = (0,278,336,115,83,1,0.990125,1.25e-4,1.25e-4,1.25e-4,1.25e-4,1.25e-4)

medium-grid-lables[0,6,8,0,:] =  (0,104,144,112,64,1,0.990125,1.25e-4,1.25e-4,1.25e-4,1.25e-4,1.25e-4)
medium-grid-lables[0,14,10,0,:] = (0,250,180,98,104,1,0.990125,1.25e-4,1.25e-4,1.25e-4,1.25e-4,1.25e-4)
medium-grid-lables[0,6,18,0,:] = (0,120,272,108,77,1,0.990125,1.25e-4,1.25e-4,1.25e-4,1.25e-4,1.25e-4)
medium-grid-lables[0,16,20,0,:] = (0,278,336,115,83,1,0.990125,1.25e-4,1.25e-4,1.25e-4,1.25e-4,1.25e-4)

fine-grid-lables[0,6,8,0,:] =  (0,104,144,112,64,1,0.990125,1.25e-4,1.25e-4,1.25e-4,1.25e-4,1.25e-4)
coarse-grid-lables[0,15,10,0,:] = (0,250,180,98,104,1,0.990125,1.25e-4,1.25e-4,1.25e-4,1.25e-4,1.25e-4)
coarse-grid-lables[0,6,16,0,:] = (0,120,272,108,77,1,0.990125,1.25e-4,1.25e-4,1.25e-4,1.25e-4,1.25e-4)
coarse-grid-lables[0,16,20,0,:] = (0,278,336,115,83,1,0.990125,1.25e-4,1.25e-4,1.25e-4,1.25e-4,1.25e-4)


### 2. Pre-Process Image

The input images should be resized to 416 x 416 x 3, but preserve the original aspect ratio.

Here's a pseudo code for image resize. It is followed by an illustrative example.

Note: `tf.resize` has a `preserve_aspect_ratio` attribute, so one could consider using it.

```python

    yolo_h, yolo_w    = 416
    orig_h, orig_w _  = image.shape

    scale = min(yolo_w/orig_w, yolo_h/orig_h)
    scaled_w, scaled_h  = int(scale * orig_w), int(scale * orig_h)
    resized_image = np.resize(image, (yolo_w, yolo_h))

    padded_image = np.full(shape=[yolo_h, yolo_w, 3], fill_value=128.0)
    d_w, d_h = (yolo_w - orig_w) // 2, (yolo_h - orig_h) // 2
    padded_image[d_h:orig_h+d_h, d_w:orig_w+d_w, :] = resized_image
```  



<ins>Example</ins>


Here's an illustration of the above pseudo code.

**Input Image**

![alt text](https://github.com/ronen-halevy/ronen-halevy.github.io/blob/master/assets/images/yolo/image-resize-1.jpg)

```python
orig_h, orig_w _  = 200, 300

scale = min(416/300, 416/200)
scale = 1.386666667

scaled_w, scaled_h  = int(1.386666667 * 300), int(1.386666667 * 200)
scaled_w, scaled_h  = 416, 277

d_w, d_h = 0, 69
```

**Illustrating Animation**

![alt text](https://github.com/ronen-halevy/ronen-halevy.github.io/blob/master/assets/images/yolo/image-resize-new.gif)


## 3. CNN Model and Decode

This section presents YOLOv3 CNN, along with its output decoding part, both are parts of the model's graph.

YOLOv3 CNN is an FCN - A Fully Convolution Network, as it is composed of convolution modules only, without any fully connected component. 

The CNN is based on Darknet-53 network as its backbone.

Here below is a high level block scheme of the YOLOv3 CNN. It is followed by a more detailed diagram of same YOLOv3 CNN.

**YOLOv3 CNN Hige Level Block Diagram**

![alt text](https://github.com/ronen-halevy/ronen-halevy.github.io/blob/master/assets/images/yolo/yolov3-cnn-higer-level.jpg)

**YOLOv3 CNN Detailed Block Diagram**

![alt text](https://github.com/ronen-halevy/ronen-halevy.github.io/blob/master/assets/images/yolo/yolov3-model-p2.jpg)


Looking at the above diagrams, one can observe 3 sub-module types:

1. Darknet-53, CNN's backbone.
2. Three CNN paths, one per each grid scale
3. Decode modules which make a post-process on CNN's output, before loss function computation.

Next section drills inside the modules, providing detailed insights on architecture.


### Darknet-53**

Take a look at the Darknet-53 part in the above block diagram and note that:

- Darknet-53 is structured as a cascade of ConvBlocks and ResBlocks.
- The `x1`, `x2`, `x8`, `x4` notations on top of the ResNet blocks in the diagram above, indicate of the duplication number of the same module. 
- Each of the 5 ConvBlocks downsamples by 2 (stride=2), for a total stride 32 at the top stage, and 16 and 8 at the stages before. Those stages feed the coarse, medium and fine scale grids respectively.

Here below is a Block diagrams of ConvBlock. ResBlock follows after.



**ConvBlock**

![alt text](https://github.com/ronen-halevy/ronen-halevy.github.io/blob/master/assets/images/yolo/conv-block.jpg)

- As depicted by the diagram, ConvBlock is a structure which combines a Conv2D module, a BatchNormalization module and a Relu Activation at the top - except for the output stages, where activation is not applied.

- ConvBlocks have 2 flavors: either with or without downsampling. The downsampling with stride=2 flavor is implemented only within the Darknet-53 block.

- ConvBlocks' kernel size is 3 inside Darknet-53, while after that, kernel size alternates between 1 x 1 with N=512 and 3 x 3 with N=1024.




**ResBlock**

![alt text](https://github.com/ronen-halevy/ronen-halevy.github.io/blob/master/assets/images/yolo/residual-block.jpg)


The Res-Block sums up the output of 2 Conv-Blocks with the input data x, aka `skipped` input.
(If the ResBlocks look familiar - then it indeed is a reuse of the structure presented in the famous `ResNet` model. 

The `ResBlock` structure provides 2 contributions:

1. It helps solving the vanishing gradient problem. (**Vanishing Gradient Reminder:**, during training back propogation, the gradients are calculated using the chain rule, so a gradient is a multiplication product of previous gradients with its conv block partial derivative. In case the network is deep, gradient values become smaller in layers closer to the top of the network.). since ResBlock's `skip` connection gradient is unity, the vanishing gradient is prevented.
2. The mixing of skip layer with the convolutioned layers refines the feature extraction, which benefits from more details provided by the skipped data and features provided by convolutioned layers output.


**Why named Darknet-53?**

Darknet-53 is because it had 53 layers. However, count of all conv2D elements in Darknet-53, (considering 2 conv2D elements in a ResBlock), sums up to:  1+1+2+1+4+1+16+1+16+1+8=52, not 53.

Still, it had 53 layers, but the fully connected output layer at the top was omitted when deployed in YOLOv3.

### 3 Scale Paths

YOLOv3 most noticeable improvement wrt earlier YOLOs, was the detection in 3 scales. This improvement enhanced smaller object detection performance, which was a weakness of previous YOLO versions.

As depicted by the above CNN diagram, the 3 scale paths are similarly structured, each with 7 convolution blocks. However, each scale is reduced by a factor of 2 in input data strides and ConvBlocks' number of filters.

**The Concatenation Block**

The medium and fine grained paths have an extra Concatenation Block, (See diagram's 3rd and 4th rows from top). It concatenates data sourced by Darknet-32's intermidiate stage, together with upsampled data from the preceding scale level path. 

![alt text](https://github.com/ronen-halevy/ronen-halevy.github.io/blob/master/assets/images/yolo/yolov3-model-p2.jpg)

Why Concatenating?
The concatenation module's contribution is quite similar to the Res Block's effect on feature detection refinement. 
Still, concatenation is applied and not summation, since the 2 datas are sourced by different network stages, which leaves no point for summation.

### Decode Module

Decode module receives CNN's output, which concatenates bbox x,y,w,h,objective and class predictions.


- 
`Decode` processes CNN outputs, before being fed to Loss Function computation.


Table below summerizes the module's operations.




Preprocessing of Bounding Box Parameters

The bounding box outputs from the CNN, i.e. center coordinates and dimenssons are postprocessed calculated like so:

x=sigmoid(x)+cx 

y=sigmoid(y)+cy 

w =  ew∗anchorw 

h =  eh∗anchorh 

Where  cxandcy  are the containing cell upper left corner coordinates, sigmoid(x) and sigmoid(y) are offsets of bounding box center within the containing cell.

In the illustration diagram below, the object center is within cell (2,4), so accordingly:

cy=1 

cx=3 

The expression for bounding box width and height is given next:

w=exp(w)∗anchor_w

h=exp(h)∗anchor_h

Where the terms anchor_w and anchor_h are explained next.

The objectness which relates to the probability of an object within the cell and class prob, which represents the probability of each class, are subject to sigmoid too.

objectness=sigmoid(objectness)

class prob=sigmoid(class prob) RONEN TBD SOFTMAX



## CNN Output Process

The diagram below describes YOLOv3 process of CNN output

**YOLOv3 Post Process**

![alt text](https://github.com/ronen-halevy/ronen-halevy.github.io/blob/master/assets/images/yolo/YOLOv3%20Post%20Process.jpg)




































https://towardsdatascience.com/yolo-v3-object-detection-53fb7d3bfe6b#:~:text=More%20bounding%20boxes%20per%20image&text=On%20the%20other%20hand%20YOLO,it's%20slower%20than%20YOLO%20v2.

https://lilianweng.github.io/lil-log/2017/12/31/object-recognition-for-dummies-part-3.html

https://vivek-yadav.medium.com/part-1-generating-anchor-boxes-for-yolo-like-network-for-vehicle-detection-using-kitti-dataset-b2fe033e5807 :
, YOLOv2 does not assume the aspect ratios or shapes of the boxes. 

https://machinelearningmastery.com/how-to-perform-object-detection-with-yolov3-in-keras/


http://christopher5106.github.io/object/detectors/2017/08/10/bounding-box-object-detectors-understanding-yolo.html


effectively splits the image into grids of arbitrary size

from https://towardsdatascience.com/yolo-v3-object-detection-53fb7d3bfe6b#:~:text=More%20bounding%20boxes%20per%20image&text=On%20the%20other%20hand%20YOLO,it's%20slower%20than%20YOLO%20v2. :

YOLO is a fully convolutional network and its eventual output is generated by applying a 1 x 1 kernel on a feature map


https://nico-curti.github.io/NumPyNet/NumPyNet/layers/route_layer.html :
Route Layer
In the YOLOv3 model, the Route layer is usefull to bring finer grained features in from earlier in the network. This mean that its main function is to recover output from previous layer in the network and bring them forward, avoiding all the in-between processes. Moreover, it is able to recall more than one layer’s output, by concatenating them. In this case though, all Route layer’s input must have the same width and height. It’s role in a CNN is similar to what has alredy been described for the Shortcut layer.

In the YOLOv3 applications, it’s always used to concatenate outputs by channels: let out1 = (batch, w, h, c1) and out2 = (batch, w, h, c2) be the two inputs of the Route layer, then the final output will be a tensor of shape (bacth, w, h, c1 + c2), as described here. On the other hand, the popular Machine Learning library Caffe, let the user chose, by the Concat Layer, if the concatenation must be performed channels or batch dimension, in a similar way as described above.

Our implementation is similar to Caffe, even though the applications will have more resamblance with YOLO models.

An example on how to instantiate a Route layer and use it is shown in the code below:


