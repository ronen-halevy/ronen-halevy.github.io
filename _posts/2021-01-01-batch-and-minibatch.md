---
layout: default
title: Batch and Mini Batch
nav_order: 11
---
# Batch and Minibatch
## Introduction

Gradient Descent and its variations are the most common algorithms used for fitting the DNN model during the Training phase. 

The basic formula of Gradient Descent parameter update is presented in Eq. 1:

### Eq. 1 Gradient Descent

\\(w_{t+1}=w_t-\alpha \cdot \triangledown L(w)\\)

Where \\(L(w)\\) is a Loss function. Loss function expresses the prediction's accuracy calculated for a every training data example.

Eq. 1 shows the formula for the recursive update of the network weight coefficient w: The new coefficient value w equals to the current w, substructed by the Loss value \\(L(w)\\) multipied by learning rate \\(\alpha\\). 
This variant of Gradient Descent, at which Eq. 1 is calculated for each data sample, is called **Stochastic Gradient Descent**, abrivated to **SGD**. Naturally, The gradients of the Loss function may be noisy, as a result of the random noise normally added to the input data samples.

The noisy gradient slows down convergence, as a result of deflections from the path on the sgradient's slope towards the gradient's minima, the progress towards the optimum is more hesitant and slow.
This noisy gradients problem may be solved by averaging the gradients over a batch of samples, so in case of an n -samples batch size, Eq. 1 is calculated once per n samples.  The rest of this post introduces the considerations involved in selecting the batch size.



## Batch and Minibatch Gradient Desent Computation

When speaking about Batch methods, there's a distinction between the **Batch** and **Minibatch** methods.
**Batch** computation relates to running the Gradient Descent update once for the entire m training examples. Accordingly, Eq. 1 changes to Eq. 2, where the gradient with respect to w is taken over the Cost function, denoted by J(w), which is an average of the loss function, as expressed in Eq. 3.

### Eq. 2: Batch Gradient Descent
\\(w=w-\alpha \cdot \frac{\partial J(w)}{\partial w}\\)

Where \\(J(w)\\) is a Cost function which expresses the averaged prediction's accuracy over the entire tarining m samples batch.

### Eq. 3: Cost Function

\\(J(w)=\frac{1}{m}\sum_{i=1}^{m}L(w)
\\)

The gradient claculated over the cost function rather than the loss function give is more accurate, less volunarable to noise. Still, the updates are slow, taken once every calculations of m samples. 

**Minibatch** is a compromise solution for the tradeoff between the SGD faster updates, and Batch algorithms accurate gradient estimaton. Normally, selected batch size would be a power of 2 number in the range of 32-512, which fits vectorized processors.


Some considerations on chosing batch size: (Goodfellow et al, ch. 8)

- Larger batches provide a mode accurate estimate of the gradient but, with less than a linear return. Let's explain that: As shown by Eq. 4, the Standard error of the mean loss, calculated over m loss values, is not decreased by a factor of \\(\frac{1}{m}\\) wrt the error of a none-batched sample, but only by a factor of \\(\frac{1}{\sqrt{m}}\\).
So accordingly, a gradient based on averaging the loss of 100 samples requires 100 times more computations, but the standard error reduces by a factor of 10 only. 
Let's see that:

### Eq. 4: Standard error of the mean loss

\\(SE({\hat{\mu}})  = \sqrt{var[\frac{1}{m}\sum_{i=1}^{m}L^{(i)}]}=\frac{1}{m} \cdot \sqrt{m} \sigma=\frac{1}{\sqrt{m}} \cdot \sigma \\)


- High Processing Computing (HPC) devices, e.g. GPUs, may be underutilized for extremely small batch sizes. This motivatess using some minimum batch size, otherwise there is no much gain in processing minibatches wrt SGD.
- Amount of required memory grows with batch size, so batch size is limitted by memory requirements.
- Some Hardware vectorized processors achieve better performance gain with power of 2 batch sizes.
- Small batches can offer regularization effect, perheps due to noise they add to the learning process.








