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

Answer: YOLO functionally segments the image to a grid of cells. But rather than running CNN seperately on each cell, it runs CNN ones, but objects are still referenced to the grid cell which contain their center is. We will illustrate that - don't worry.

**Gridded Image**

![alt text](https://github.com/ronen-halevy/ronen-halevy.github.io/blob/master/assets/images/yolo/yolov3-input-image-cells-shapes.jpg)


The CNN assigns a detection descriptor to each grid cell. The descriptor combines 2 vectors
1. A Classification result vector with predicted probabilty for each dataset's class
2. Bounding Box Location - $x_1, x1, w, ,h$, and also an Objective Prediction which indicates probabilty of an object in the cell.

Diagram below illustrates that.

**Detection Descriptor**

![alt text](https://github.com/ronen-halevy/ronen-halevy.github.io/blob/master/assets/images/yolo/yolov3-single-output-cell-fields.jpg)



As a matter of fact, YOLOv3 supports 3 detections per a cell - the corrected descriptors illustration diagram follows.

**Detection 3 Descriptors**

![alt text](https://github.com/ronen-halevy/ronen-halevy.github.io/blob/master/assets/images/yolo/yolov3%20output-cell-with-fields.jpg)


Now let's combine the descriptors with the entire grid of cells:

The CNN assigns a detection descriptor to each grid cell, so for a 13x13 grid, the CNN output looks like this:

**YOLOv3 - CNN Output**

![alt text](https://github.com/ronen-halevy/ronen-halevy.github.io/blob/master/assets/images/yolo/yolov3%20output%20-cube-13.jpg)

### Achieving the Grid Effect

So, how is the grid effect achieved?

The grid effect is achieved by passing the image thru a `FuLL CNN,  with 32 strides, such that the 416*416*3 input image results in a 13 x 13 x N shape output - this is the strcture illustrated in the above cube diagram, and 
$N=3 x (5+N_{classes})$ 

To enhence detection performance for smaller objects, YOLOv3 CNN generates simultaneously output in 3 grid scales: 13 x 13 (as depicted above), and also 26 x 26 and 52 x 52.


### YOLOv3 Block Diagrams

The below block diagrams describe YOLOv3 `Forwarding` and `Training` operation.
Following chapters of this article present a detailed description of the 2 operation modes.

**YOLOv3 Block Diagram: Forwarding**


![alt text](https://github.com/ronen-halevy/ronen-halevy.github.io/blob/master/assets/images/yolo/forwarding.gif)

**YOLOv3 Block Diagram: Training**

![alt text](https://github.com/ronen-halevy/ronen-halevy.github.io/blob/master/assets/images/yolo/training.gif)

## Image Resize

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



## YOLOv3 CNN

The network is FCN - Fully Convolution, so it is composed of convolution modules only, without any fully connected component. 

As mentioned before, the network generates outputs in 3 scales, with a 13x13, a 26x26 and a 52x52 grid size for the coarse, medium and fine grid scales respectively.

The backbone of YOLOv3 network is the Darknet-53 network - details on darknet-32 are provided in a next paragraph. 

So here's a higher level block scheme of the CNN:

**CNN Higer Level Diagram**

![alt text](https://github.com/ronen-halevy/ronen-halevy.github.io/blob/master/assets/images/yolo/yolov3-cnn-higer-level.jpg)


Let's now drill in towards the presentation of a more detailed diagram of the CNN.
Still, for the sake of simplicity, we'll present it gradually. We start with the 13x13 output path. After that, we will add the rest of the network and present the entire picture.

Here below is a detailed diagram of the 13 x 13 grid path. It is followed by explainations on the main building blocks. Still, the reader is assumed to be familiar with standard Conv Net standard modules - `Relu` and `Batch Normalization` otherwise the reader can look those up.

**YOLO CNN Coarse Grid Path**


![alt text](https://github.com/ronen-halevy/ronen-halevy.github.io/blob/master/assets/images/yolo/yolov3-model-p1.jpg)


### CNN Modules Description

**Darknet-53**

- `Darknet-53, as depicted by the diagram above, is structured as a cascade of 5 blocks of modules marked as `Res-Blocks`, with 6 additional blocks of modules marked as `Conv Blocks`. The 2 diagrams below present the structures of a `Conv Block` and a `Res-Block`.

- The `x1`, `x2`, `x8`, `x4` notations on top of the Res-Net blocks in the diagram above, indicate of the duplication number of the same module. 

- Summing up the total number of conv2D elements in Darknet-53, (considering 2 conv2D elements in a Res-Block), we get:  1+1+2+1+4+1+16+1+16+1+8=52

- BTW, name is Darknet-53 is because it had 53 layers, but the fully connected output layer at the top was omitted.

- Each of the 5 ConvBlocks downsamples by 2 (stride=2), for a total stride 32. Accordingly, the 416x416 input image results in a 13x13 output grid.

**Conv Block**

![alt text](https://github.com/ronen-halevy/ronen-halevy.github.io/blob/master/assets/images/yolo/conv-block.jpg)

- As depicted by the diagram, ConvBlock is a combination of Conv2D module, BatchNormalization module and a Relu activation.

- Note that the conv blocks have come in 2 flavors: either with downsampling, with stride equals 2 and zero-padding , (this flavor is implemented only within the Darknet-53 block), and without downsampling, where stride is 1.

- The ConBlock is terminated by a `Relu` activation, except for the output stages, where activation is not applied.

- The kernel size inside Darknet-53 is 3, while after that it alternates between 1 x 1 with N=512 and 3 x 3 with N=1024.

**Res-Block**

![alt text](https://github.com/ronen-halevy/ronen-halevy.github.io/blob/master/assets/images/yolo/residual-block.jpg)


The Res-Block sums up the output of 2 Conv-Blocks with the input data x, aka `skipped` input.
(If the ResBlocks look familiar - then it indeed is a reuse of the structure presented in the famous `ResNet` model. 

The `ResBlock` structure provides 2 contributions:

1. It helps solving the vanishing gradient problem. (**Vanishing Gradient Reminder:**, during training back propogation, the gradients are calculated using the chain rule, so a gradient is a multiplication product of previous gradients with its conv block partial derivative. In case the network is deep, gradient values become smaller in layers closer to the top of the network.). since ResBlock's `skip` connection gradient is unity, the vanishing gradient is prevented.
2. The mixing of skip layer with the convolutioned layers refines the feature extraction, which benefits from more details provided by the skipped data and features provided by convolutioned layers output.



# Expanding View To All 3 Scales

YOLOv3 most noticeable improvement is the 3 scales detection. This enhances smaller object detection performance, which btw, was a weakness of previous YOLO versions.

Here below is a block diagram of the complete YOLOv3 network. As depicted by the diagram, the 3 scale paths are quite similar, each consists of 7 convolution blocks, but each path runs in a different scale and number of filters.

The Medium path is fed by Darknet-35's 16 strides stage, while the fine path is fed by Darknet-35's 8 strides stage.

The medium and fine grained paths have an extra concatenation block, (See diagram's 3rd and 4th rows from top). It concatenates data sourced by Darknet-32's intermidiate stage, together with upsampled data from the preceding scale level path. 

![alt text](https://github.com/ronen-halevy/ronen-halevy.github.io/blob/master/assets/images/yolo/yolov3-model-p2.jpg)

**Why Concatenatin?**
The concatenation module's contribution is quite similar to the Res Block's effect on feature detection refinement. 
Still, concatenation is applied and not summation, since the 2 datas are sourced by different network stages, which leaves no point for summation.

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


