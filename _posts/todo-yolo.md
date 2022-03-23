# A Guide To YOLOv3

## Introduction to Object Detection

The task of a CNN object detection model is dual: It provides both classifies objects within an image to dataset labels, and also provides an estimation to objects' bounding boxes locations. The diagram below illustrates an input image on the left, and classification with bounding box annotations results on the right.

**Input Image (Left), Output Annotation (Right)**

![alt text](https://github.com/ronen-halevy/ronen-halevy.github.io/blob/master/assets/images/yolo/image-shapes-annotations.jpg)

A detection model normally outputs 2 vectors per each detected object:

- A Classification output vector, with estimated probabilities of each dataset label. Vector length is \\(N_classes\\), i.e. number of classes in sdataset. Decision is mostly taken by applying a softmax operator on the vector.
- A vector with the predicted location of a bounding box which encloses the object. The location can be represented in various formats as illustrated in the diagram below.

**Representation Formats**: 

- (1): $((x_1,y_1), (x_2,y_2))$
- (2): $((x_c,y_c), (w,h))$

![alt text](https://github.com/ronen-halevy/ronen-halevy.github.io/blob/master/assets/images/yolo/circle-image.jpg)

## Object Detection Models

As a baground to the presentation of YOLOv3 model, this section presents 

### Plain CNN Model

Using a conventional CNN classification model, by adding a regression predictor to it is a straight forward implementation. However, it is limitted to detect a single object only.
See illustrative diagram below: The image consists of 3 shape objects. Assume that the dataset set of labels is: ['square', 'ellipse', 'triangle', 'hexagon', 'circle']. The model, at the best case, will detect one of the object shapes only.
Such detection models, with a detection capability of a sinlge object are often reffered to as `Object Localization` models.

**Plain CNN Model**

![alt text](https://github.com/ronen-halevy/ronen-halevy.github.io/blob/master/assets/images/yolo/image-classification.jpg)


### Sliding Window Model

In this model, the CNN is activated in the bounderies of a square windows which slides in straight lines along the image as illustrated in the animation diagram below.
To fit various object size, multiple window sizes should be activated, as depicted in the animation, and/or multiple image scales.

Location can be determined by window's region, and the offset of the bounding box within the sliding window position.

**Sliding Window**

![alt text](https://github.com/ronen-halevy/ronen-halevy.github.io/blob/master/assets/images/yolo/sliding-window-detection.gif)

There are drawbacks to this model: The repeated deployment of the CNN model per each window position, constraints a heavy computation load. But not only that - since convolution span regions are limitted by window's size and position, which may be uncorrelated with image's regions of interest postion and sizes, objects may be cropped or missed by the model.

### R-CNN

R-CNN ([by Ross Girshick et al, UC Berkely, 2014](https://arxiv.org/abs/1311.2524)), which stands `Regions with CNN features`, addresses drawbacks of `Sliding Window` model. 
The idea of R-CNN in essence is of a 3 steps process:
1. Extract region proposals - 2000 regions were stated n original paper. The farther process is limitted to proposed regions. There are a number of algorithm which can make region proposals. The authors used [selective search by J. Uijlings, K. van de Sande, T. Gevers, and A. Smeulders.Selective search for object recognition.IJCV, 2013.](https://www.researchgate.net/publication/262270555_Selective_Search_for_Object_Recognition)
2. Deploy CNN with bounding box regression over each proposed region.
3. Classify each region - originally using Linear SVM, in later model's variants e.g. `Fast R-CNN`, `Softmax` was deployed.

**Region Proposals**

![alt text](https://github.com/ronen-halevy/ronen-halevy.github.io/blob/master/assets/images/yolo/image-classification-rcnn.jpg)

This is just a brief description of the algorithm, which at that time, contributed to a dramatic improvement of CNN detection models performance. R-CNN was later followed by improvments variants such as FASTR-CNN, [Girshick, Ross. "Fast r-cnn." Proceedings of the IEEE international conference on computer vision, 2015](https://arxiv.org/abs/1504.08083), FASTRR-CNN, [Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun, 2016](https://arxiv.org/abs/1506.01497). These models aimed to address R-CNN problems, amongst are real time performance issues,long training time for the 2000 regions and region selection process.


## A Brief Introduction to YOLOv3

This article is about YOLO (`You Only Look Once`), and specifically its 3rd version YOLOv3. [YOLOv3: An Incremental Improvement, Joseph Redmon, Ali Farhadi, 2018](https://arxiv.org/abs/1804.02767)

The dominant common practice of CNN object detecton models before YOLO (e.g. Sliding Window and R-CNN), was to divide the image to regions, and run CNN on each. The computation cost required is huge.

YOLO also uses the practice of segmentig the image, as due to that segmantation it can detect many objects in an image.
However, YOLO's way of segmenting the image is entirely different: instead of running CNN over thousands of regions seperately, it runs CNN only once over the entire image.
This is a huge difference, which makes YOLO so much faster - YOLOv3 is 1000 times faster than R-CNN.

So how is segmentation still achieved?

Let's see...


As the illustrated gridded image below depicts, the detected objects are referenced to the grid cell which contain their center is. 

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

The grid effect is achieved by passing the image thru a FuLL CNN,  with 32 strides, such that the 416*416*3 input image results in a 13 x 13 x N shape output - this is the strcture illustrated in the above cube diagram, and 
$N=3 x (5+N_{classes})$ 

To enhence detection performance for smaller objects, YOLOv3 CNN generates simultaneously output in 3 grid scales: 13 x 13 (as depicted above), and also 26 x 26 and 52 x 52.


### YOLOv3 Block Diagrams

The below block diagrams describe YOLOv3 `Forwarding` and `Training` operation.
Following chapters of this article present a detailed description of the 2 operation modes.

**YOLOv3 Block Diagrams: Forwarding and Training**

![alt text](https://github.com/ronen-halevy/ronen-halevy.github.io/blob/master/assets/images/yolo/yolov3-flow-diagram.jpg)

## Image Resize

The input images should be resized to 416 x 416 x 3, but preserve the original aspect ratio.

Here's a pseudo code for image resize. It is followed by an illustrative example.

Note: `tf.resize` has a `preserve_aspect_ratio` attribute, so one could consider using it.

`python

    yolo_h, yolo_w    = 416
    orig_h, orig_w _  = image.shape

    scale = min(yolo_w/orig_w, yolo_h/orig_h)
    scaled_w, scaled_h  = int(scale * orig_w), int(scale * orig_h)
    resized_image = np.resize(image, (yolo_w, yolo_h))

    padded_image = np.full(shape=[yolo_h, yolo_w, 3], fill_value=128.0)
    d_w, d_h = (yolo_w - orig_w) // 2, (yolo_h - orig_h) // 2
    padded_image[d_h:orig_h+d_h, d_w:orig_w+d_w, :] = resized_image
    
`

**Example**

Here's an illustration of the above pseudo code.

**Input Image**

![alt text](https://github.com/ronen-halevy/ronen-halevy.github.io/blob/master/assets/images/yolo/image-resize-1.jpg)

orig_h, orig_w _  = 200, 300

scale = min(416/300, 416/200)
scale = 1.386666667

scaled_w, scaled_h  = int(1.386666667 * 300), int(1.386666667 * 200)
scaled_w, scaled_h  = 416, 277

**Scaled Image**

![alt text]((https://github.com/ronen-halevy/ronen-halevy.github.io/blob/master/assets/images/yolo/image-resize-2.jpg)


**Padded Image Template**

![alt text]((https://github.com/ronen-halevy/ronen-halevy.github.io/blob/master/assets/images/yolo/image-resize-3.jpg)


d_w, d_h = 0, 69

**Padded Image**

![alt text]((https://github.com/ronen-halevy/ronen-halevy.github.io/blob/master/assets/images/yolo/image-resize-4.jpg)








![alt text](https://github.com/ronen-halevy/ronen-halevy.github.io/blob/master/assets/images/yolo/yolov3-Flow%20Forward%20Path.jpg)


Let's zoom into YOLO CNN output - it is illustrated in the following diagram:













Suppose we have this image:






























https://morioh.com/p/9bca8f92d016 :
How does YOLO work?

YOLO is based on a single Convolutional Neural Network (CNN). The CNN divides an image into regions and then it predicts the boundary boxes and probabilities for each region. It simultaneously predicts multiple bounding boxes and probabilities for those classes. YOLO sees the entire image during training and test time so it implicitly encodes contextual information about classes as well as their appearance.




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


