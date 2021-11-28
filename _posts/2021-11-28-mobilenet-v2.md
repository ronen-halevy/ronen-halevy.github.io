---
title: MobileNetV2: Inverted Residuals and Linear Bottlenecks
tags:  Detection Segmentation Guides
---

## Introduction

The Background for proposing this Neural Network model, was the challenge of implementing deeper CNNs to achieve better classification performance. Deeper CNNs resulted with improved performance. This is valid for a various of computer vision tasks such as recognition, detction, segmentation etc. On the other hand, when getting much deeper, problems such as vanishing/exploding gradients become more significant, with symptoms such as growing errors and accuracy degradation. 

In their paper from 2015, "Deep Residual Learning for Image Recognition"](https://arxiv.org/abs/1512.03385), Kaiming He, Xiangyu Zhang, Shaoqing Ren & Jian Sun proposed a model which enables the training very deep networks - the Resnet. The paper demonstrates Resnet-34, Resnet-50, Resnet-101 and Resnet-152, which deploy 34, 50, 101 and 152 parametric layers respectively.

## Resnet Functional Description

Resnet building block is called a `Residual Block` which contains a `skip` connection, aka a `short-cut` as depicted in Figure 1 and described next.



***Figure 1: Resnet Residual Block***

![Resnet Residual Block](https://github.com/ronen-halevy/ronen-halevy.github.io/blob/master/assets/images/cnn-models/resnet-residual-block.drawio.png)

As shown by Figure 1, the skip is added to plain connection just before the Relu module, thus providing nolinearity effect to the sum.

The Resnet stacks Residual Blocks back to back as illustrated by Figure 2.


***Figure 1: Resnet - A Stack of Residual Blocks***

![Resnet Residual Block](https://github.com/ronen-halevy/ronen-halevy.github.io/blob/master/assets/images/cnn-models/resnet-stack-of-residual-block.drawio.png)


## Notes on Vanishing Gradient problem

***A brief reminder of vanishing gradient problem*** - during the backpropagation, the network's weights parameters are gradually updated to minimize the loss function, according to Gradient Descent algorithm - or any of its variants.

Here's the Gradient Descent update equation, for uptating the weights of the kth layer at time \\(i+1\\):

$w_{i+1}^k=w_i^k+\alpha * \frac{d L}{dw_i^k} $


Where:

- \\(w_{i+1}^k\\) expresses the kth layer's weights at time \\(i+1\\).
- \\(\alph\\) is the learning rate
- \frac{d L}{dw_i^k} is the Loss gradient with respect to the weight at time \\(i+1\\).


The gradients are calculated using the derivative chain rule, where the optimization calculation is executed in a back propogating manner, starting from layer k=L, back till k=1. 

Let's illustrate the back propogation on a residual block, i.e. the Loss derivative with respect to \\(a^{k}\\), given \\(\frac{dL}{da^{(k+2)}}\\). 

See Diagram below, which is followed by the xhin rule detivative expression.

![Resnet Residual Block](https://github.com/ronen-halevy/ronen-halevy.github.io/blob/master/assets/images/cnn-models/chain-rule-resnet-stack-of-residual-block.drawio.png)

According to chain rule:

\\(\frac{dL}{da^{(k)}} =  \frac{dL}{da^{(k+2)}}\frac{da^{(k+2)}}{da^{(k)}}\\)

Consider that:

\\(a^{(k+2)}} =  g(F(a^{(k)}) +  a^{(k)})


Where g(x) is the activation function, ReLu in this example. ReLu is linear for positive arguments and zero otherwise, so let's concentrate on the nonw zero case.

In that case, the derivative of the `skipped` component is 1, so this component protects against vanishing gradient problem.

## Notes on Dimenssion Matching

If the dimenssion of the plain section increae, so it now differs from the skip input dimenssions, 2 approaches can be taken:
1. Add extra zero padding to the skip data.
2. Use projection, i.e. convolve with 1 x 1 kernel, just to expand dimensions.

Besides improving vanishing gradients issue, the `skip` connections carry lower level information from initial layers which correspnd to lower level features. 
Adding this information to the higher level more abstract information extracted by the layers which follow, contributes to better performance.
