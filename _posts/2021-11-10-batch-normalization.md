---
title: Batch Normalizions
tags: Keras Normalization
---

# Batch Normalization


Normalizing input data can considerably accelatrate training rate. Keras implements Batch Normalization as described in ["Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
" by Sergey Ioffe, Christian Szegedy](https://arxiv.org/abs/1502.03167), so this post follows this paper accordingly.

Normalization is required not just at the input layer but also before each of the hidden layers. Normalize is commonly taken before the layer's activation stage. Here's the general

\\(\hat {x_i}^{(k)} = \frac{ {x_i}^{(k)}-\mu_B^{(k)}}{\sqrt{  {\sigma_B^{(k)}}^2   + \epsilon}}\\)

Where:
\\(k \. \epsilon [1,d]\\) denotes the \\(k{th} \\) eelement within the d elements example (e.g.  the \\(k{th} \\) pixel.

 \\(i \. \epsilon [1,m]\\) denotes the \\(i{th} \\) examples within the batch of m examples. within a single example image ).

\\(\mu_B^{(k)}\\) denotes the mean of the \\(k_{th}\\) element, calculated for the batch.

\\({\sigma_B^{(k)}}^2\\) denotes the variance of the \\(k_{th}\\) element, calculated for the batch.

\\( \epsilon \\)is a stabilization factor, for the case \\(\sigma=0\\).

Normalizing the input data values like so, would make data have the same zero mean and variance=1 for all hidden layers. To prevent this uniformity which might degenerate the network. To prevent that, the normalized data is passed through a linear operator like so:


\\(y^{(k)} =  \gamma^{(k)}  \hat {x_i}+ \beta^{(k)} \\)

Where 

\\( \gamma\\) and \\(\beta\\) are learnable parameters in the model.








