# A Guide To YOLOv3

## Introduction to Object Detection

The task of a CNN object detection model is dual: It provides both classifies objects within an image to dataset labels, and also provides an estimation to objects locations.

Accordingly, the model outputs 2 vectors per a detected object:

- A Classification output vector, with estimated probabilities of each dataset label. Vector length is \\(N_classes\\), i.e. number of classes in sdataset. Decision is mostly taken by applying a softmax operator on the vector.
- A vector with the predicted location of a bounding box which encloses the object. The location can be presented in various formats as illustrated in the diagram below.

**Bounding Box Annotation Formats**: 

- Image on the left: The image contains a circle object
- Annotations(1): Bounding Box annotated by corners coordinates.
- Annotations(2): Bounding Box annotated by center coordinates and boxe's dimenssions.

![alt text](https://github.com/ronen-halevy/ronen-halevy.github.io/blob/master/assets/images/yolo/circle-image.jpg)

## Object Detection Models

### Plain CNN Model

Using a conventional CNN classification model, by adding a regression predictor to it is a straight forward implementation. However, it is limitted to detect a single object only.
See illustrative diagram below: The image consists of 3 shape objects. Assume that the dataset set of labels is: ['square', 'ellipse', 'triangle', 'hexagon', 'circle']. The model, at the best case, will detect one of the object shapes only.
Such detection models, with a detection capability of a sinlge object are often reffered to as `Object Localization` models.

![alt text](https://github.com/ronen-halevy/ronen-halevy.github.io/blob/master/assets/images/yolo/image-classification.jpg)


### Sliding Window Model

In this model, the CNN is activated in the bounderies of a square windows which slides in straight lines along the image as illustrated in the animation diagram below.
To fit various object size, various windows sizes should be activated, as depicted in the animation, and/or sliding window should run over various image scales.

Objects location can be determined by window's position, and the offset of the bounding box within the sliding window position.

![alt text](https://github.com/ronen-halevy/ronen-halevy.github.io/blob/master/assets/images/yolo/sliding-window-detection.gif)

There are drawbacks to this model: The repeated deployment of the CNN model per each window position, constraints a heavy computation load. But not only that - since convolution span regions are limitted by window's size and position, which may be uncorrelated with image's regions of interest postion and sizes, objects may be cropped or missed by the model.

### R-CNN
R-CNN ([by Ross Girshick et al, UC Berkely, 2014](https://arxiv.org/abs/1311.2524)), which stands `Regions with CNN features`, addresses drawbacks of `Sliding Window` model. 
The idea of R-CNN in essence is of a 3 steps process:
1. Extract region proposals - 2000 regions were stated n original paper. The farther process is limitted to proposed regions. There are a number of algorithm which can make region proposals. The authors used [selective search by J. Uijlings, K. van de Sande, T. Gevers, and A. Smeulders.Selective search for object recognition.IJCV, 2013.](https://www.researchgate.net/publication/262270555_Selective_Search_for_Object_Recognition)
2. Deploy CNN with bounding box regression over each proposed region.
3. Classify each region - originally using Linear SVM, in later model's variants e.g. `Fast R-CNN`, `Softmax` was deployed.

This is just a brief description of the algorithm, which at that time, contributed to a dramatic improvement of CNN detection models performance. R-CNN was later followed by improvments variants such as FASTR-CNN, [Girshick, Ross. "Fast r-cnn." Proceedings of the IEEE international conference on computer vision, 2015](https://arxiv.org/abs/1504.08083), FASTRR-CNN, [Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun, 2016](https://arxiv.org/abs/1506.01497). These models aimed to address R-CNN problems, amongst are real time performance issues,long training time for the 2000 regions and region selection process.



## Background



For example, image classification is straight forward, but the differences between object localization and object detection can be confusing, especially when all three tasks may be just as equally referred to as object recognition.

Image classification involves assigning a class label to an image, whereas object localization involves drawing a bounding box around one or more objects in an image. Object detection is more challenging and combines these two tasks and draws a bounding box around each object of interest in the image and assigns them a class label. Together, all of these problems are referred to as object recognition.




Object detection 
https://bdtechtalks.com/2021/06/21/object-detection-deep-learning/
While an image classification network can tell whether an image contains a certain object or not, it won’t say where in the image the object is located. Object detection networks provide both the class of objects contained in an image and a bounding box that provides the coordinates of that object.

## Models For Object Detection

### Conventional CNN

### sliding Window

### R-CNN




This post describes YOLOv3, but focuses mainly on implementation aspects.

# How YOLOv3 Works
.
YOLOv3 predicts bounding boxes and class probabilities of an entire image in a unified convolution network computation pass. That means, the predictions process is executed over the entire image at the same pass.

Here's an illustration - 

**The input** is an image of some shapes:

![alt text](https://github.com/ronen-halevy/ronen-halevy.github.io/blob/master/assets/images/yolo/input-image-shapes.jpg)


**Expected Output**:


![alt text](https://github.com/ronen-halevy/ronen-halevy.github.io/blob/master/assets/images/yolo/image-shapes-annotations.jpg)



To do so, the input image is divided into an $size_x * size_y$ grid cells. Each grid cell is responsible for detecting objects which centers falls within it's bounderies. like so (images are resized to a uniform 415*416 size):

![alt text](https://github.com/ronen-halevy/ronen-halevy.github.io/blob/master/assets/images/yolo/yolov3-input-image-cells-shapes.jpg)


The grid effect is achieved by passing the image thru a FuLL CNN,  with 32 strides, such that the 416*416*3 input image results in a 13x 13 x N shape output.

The resultant output size is 13 x 13 x N, where each of the 1 x 1 x N cells corresponds to a division within the image.
The cells' width are:

$N=3 x (5+N_{classes})$ - we will get to that structure soon.

Diagrams below illustrate YOLOv3 Forwarding path:

![alt text](https://github.com/ronen-halevy/ronen-halevy.github.io/blob/master/assets/images/yolo/yolov3-Flow%20Forward%20Path.jpg)


Let's zoom into YOLO CNN output - it is illustrated in the following diagram:


![alt text](https://github.com/ronen-halevy/ronen-halevy.github.io/blob/master/assets/images/yolo/yolov3%20output%20-cube-13.jpg)


As depicted by the diagram, each of the grid cells holds 3 records of information, each corresponds to a detected object bounding box, i.e. YOLOv3 supports up to 3 bounding boxes per cell.

Next diagram drills deeper into the structure of the bounding boxes records:

![alt text](https://github.com/ronen-halevy/ronen-halevy.github.io/blob/master/assets/images/yolo/yolov3%20output-cell-with-fields.jpg)











Let's ilustrate it with diagrams:

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


