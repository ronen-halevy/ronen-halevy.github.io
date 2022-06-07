---
title: Batch Normalizations
tags: Keras Normalization guides
---

## Introduction



The goal to perform data normalization is to all input features to a common scale - mean 0 and varianc 1. Normalizing input data can considerably accelatrate training rate. This post follows a [paper](https://arxiv.org/abs/1502.03167) by Ioffe and Szegedy which BatchNormalization API is based on. Let's review the details.

## The Algorithm 

It is evident that normalization is required not just before the first input layer, but also before each hidden layers which follow. To start with, Eq. 1 presents the straight forward normalization formula.

Regarding the indices \\(i\\) and \\(k\\) - input example \\(x=(x^{(1)}....x^{(d)})\\) has d dimenssions (e.g. an image with d pixels). Minibatch holds m examples.

### Eq. 1: Normalization

\\(\hat {x_i}^{(k)} = \frac{ {x_i}^{(k)}-\mu_B^{(k)}}{\sqrt{  {\sigma_B^{(k)}}^2   + \epsilon}}\\)

Where:

- \\(k \ \epsilon \ [1,d]\\) denotes the dimension index within a single input example.

- \\(i \ \epsilon \ [1,m]\\) denotes the \\(i{th} \\) example within the m examples mini-batch.

- \\(\mu_B^{(k)}\\) denotes the mean calculated for the \\(k_{th}\\) dimension - find details below.

- \\({\sigma_B^{(k)}}^2\\) denotes the variance calculated for the \\(k_{th}\\) dimension -see details below.

- \\( \epsilon \\) is a stabilization factor, needed for the case \\(\sigma=0\\), negligible otherwise.


## Add Scaling and Offset Factors

Implementing the described above plain baNormalizing the input data values like so, would make data have the same zero mean and variance=1 for all hidden layers. To prevent this uniformity which might degenerate the network. To prevent that, the normalized data is passed through a linear operator like so:


\\(y^{(k)} =  \gamma^{(k)}  \hat {x_i}+ \beta^{(k)} \\)

Where:

- \\( \gamma\\) is a learned scaling factor.

- \\(\beta\\)  is a learned offset factor.

Both \\( \gamma\\) and \\(\beta\\)  are trainable by the optimizer during the parameters fitting stage, along with other model's parameters.



##  Mean and Variance Calculation within a mini-batch

The straight forward expresions for the mean and variance of an m examples mini-batch,  are given by Eq. 2 and 3:

### Eq. 2: Mean over the mini-batch

\\(\mu_B=\frac{1}{m}\sum_{i=1}^{m}x_i\\)

### Eq. 3: Variance over the mini-batch

\\(\sigma_B^2=\frac{1}{m}\sum_{i=1}^{m}(x_i-\mu_B)^2\\)


##  Mean and Variance Calculation within a mini-batch


Eq. 2 and 3 relates to the calculation done within an m examples minibatch. Howexver, it is required to process multiple training mini-batches, and average the values over them, as denoted by Eq. 3 and Eq. 4:

### Eq. 3: Mean over multiple mini-batches

\\(E(x) = E_B[\mu_B]\\)

Eq.3 averages the the mini-batches' means.


### Eq. 4: Variance over multiple mini-batches

\\(Var(x) = \frac{m}{m-1}E_B[{\sigma_B}^2]\\)

Eq. 4 is the unbiased variance estimate formula.

Note that  m is the size of each of the averaged mini-batches while \\({\sigma_B}^2\\) is the variance of a single mini-batch.


Keras uses moving average formula to calculate mean and variance over the mini-batches, as shown by Eq. 5 and 6:


### Eq. 5: Keras moving average calculation of Mean


\\(E(x) = E(x) * momentum + \mu_B * (1 - momentum)\\)

Where:

- E(x) is the moving average **Mean** over the mini-batches

- \\(\mu_B\\) is the momentum of a single mini-batch.

- momentum - is a constant coefficient, determined by the API. default to 0.99

### Eq. 6: Keras  moving average calculation of Variance


\\(Var(x) = Var(x) * momentum + {\sigma_B}^2 * (1 - momentum)\\)

Where:

- Var(x) is is the moving average **Variance** over the mini-batches.

- \\({\sigma_B}^2 \\) is the variance of a single mini-batch.

- momentum - (same as above), is a constant coefficient, determined by the API. default to 0.99


## Keras API - BatchNormalization


Here's the signiture of keras API

```python
tf.keras.layers.BatchNormalization(
    axis=-1,
    momentum=0.99,
    epsilon=0.001,
    center=True,
    scale=True,
    beta_initializer="zeros",
    gamma_initializer="ones",
    moving_mean_initializer="zeros",
    moving_variance_initializer="ones",
    beta_regularizer=None,
    gamma_regularizer=None,
    beta_constraint=None,
    gamma_constraint=None,
    **kwargs
)
```

## A Note On Parameter Storage Requirements


The calculation of batch normalization requires the storing og 4 parameters:


1. \\( \mathbf{\gamma}\\) - the scaling factor, which is an optimization trainable parameter.
2. \\(\mathbf{\beta}\\)  - the offset factor, an optimization trainable parameter.
3. **E(x)** - The mean averaged accross mini-batches - not a trainable parameter , but still needed to be stored.
4. **Var(x)** - The variance averaged accross mini-batches a - not a trainable parameter, but still needed to be stored.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1111


ADD CODE:
https://keras.io/guides/transfer_learning/

**Example: the BatchNormalization layer has 2 trainable weights and 2 non-trainable weights**

layer = keras.layers.BatchNormalization()
layer.build((None, 4))  # Create the weights

print("weights:", len(layer.weights))
print("trainable_weights:", len(layer.trainable_weights))
print("non_trainable_weights:", len(layer.non_trainable_weights))
weights: 4
trainable_weights: 2**
non_trainable_weights: 2


!!p.s. note (2022 june):
https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization

Importantly, batch normalization works differently during training and during inference.

During training (i.e. when using fit() or when calling the layer/model with the argument training=True), the layer normalizes its output using the mean and standard deviation of the current batch of inputs. That is to say, for each channel being normalized, the layer returns gamma * (batch - mean(batch)) / sqrt(var(batch) + epsilon) + beta, where:

epsilon is small constant (configurable as part of the constructor arguments)
gamma is a learned scaling factor (initialized as 1), which can be disabled by passing scale=False to the constructor.
beta is a learned offset factor (initialized as 0), which can be disabled by passing center=False to the constructor.
During inference (i.e. when using evaluate() or predict() or when calling the layer/model with the argument training=False (which is the default), the layer normalizes its output using a moving average of the mean and standard deviation of the batches it has seen during training. That is to say, it returns gamma * (batch - self.moving_mean) / sqrt(self.moving_var + epsilon) + beta.

self.moving_mean and self.moving_var are non-trainable variables that are updated each time the layer in called in training mode, as such:

moving_mean = moving_mean * momentum + mean(batch) * (1 - momentum)
moving_var = moving_var * momentum + var(batch) * (1 - momentum)
As such, the layer will only normalize its inputs during inference after having been trained on data that has similar statistics as the inference data.
