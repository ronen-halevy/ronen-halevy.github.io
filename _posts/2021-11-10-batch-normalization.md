
---
title: Batch Normalizions
tags: Keras Normalization Knowledgebase
---


# Batch Normalization

Normalizing input data can considerably accelatrate training rate. Normalization is required not just at the input layer but also before each of the hidden layers. Normalize is commonly taken before the layer's activation stage. Here's the general 

\\(\hat {x_i}^{(k)} = \frac{ {x_i}^{(k)}-\mu_B^{(k)}}{\sqrt{  {sigma_B^{(k)}}^2   + \epsilon}}\\)

Where:

\\({x_i}^{(k)}\\) denotes the \\(i_{th}\epsilon [1,d]\\)  element of the \\(k_{th} \epsilon  [1,m]\\) batch's example,

\\\mu_B^{(k)}\\} denotes the mean of the \\(k_{th}\\) batch  
\\\{sigma_B^{(k)}}^2\\} denotes the variance of the \\(k_{th}\\) batch  


Loffer and Szegedey (1) suggest that normalization is needed before each layer. The straight forward normalization formula is:



Where:







Deep Neural Networks is complicated by the fact that the distribution of each layer’s inputs changes during training, as the parameters of the previous 
layers change. This slows down the training by requiring lower learning rates and careful parameter initialization, and makes it notoriously hard to 
train models with saturating nonlinearities. We refer to this phenomenon as internal covariate shift, and address the problem by normalizing layer inputs. 
Our method draws its strength from making normalization a part of the model architecture and performing the normalization for each training mini-batch. 
Batch Normalization allows us to use much higher learning rates and be less careful about initialization, and in some cases eliminates the need for Dropout. 
Applied to a stateof-the-art image classification model, Batch Normalization achieves the same accuracy with 14 times fewer training steps,
and beats the original model by a significant margin. Using an ensemble of batch-normalized networks, we improve upon the best published result 
on ImageNet classification: reaching 4.82% top-5 test error, exceeding the accuracy of human raters.


Batch Normalization can be applied to any set of activa-tions  in  the  network.   Here,  we  focus  on  transforms  thatconsist of an affine transformation followed by an element-wise nonlinearity:z =g(Wu + b)whereWandbare learned parameters of the model, andg(·)is  the  nonlinearity  such  as  sigmoid  or  ReLU.  Thisformulation covers both fully-connected and convolutionallayers.  We add the BN transform immediately before thenonlinearity, by normalizingx =Wu + b. We could havealso  normalized  the  layer  inputsu,  but  sinceuis  likelythe output of another nonlinearity,  the shape of its distri-bution is likely to change during training, and constrainingits first and second moments would not eliminate the co-variate shift.  In contrast,Wu + bis more likely to havea symmetric, non-sparse distribution, that is “more Gaus-

# Batch-Normalized Convolutional Networks
Batch Normalization enables higher learning ratesIn traditional deep networks, too high a learning rate mayresult in the gradients that explode or vanish, as well as get-ting stuck in poor local minima. Batch Normalization helpsaddress these issues.  By normalizing activations through-out the network, it prevents small changes in layer parame-ters from amplifying as the data propagates through a deepnetwork.   For  example,  this  enables  the  sigmoid  nonlin-earities to more easily stay in their non-saturated regimes,which is crucial for training deep sigmoid networks but hastraditionally been hard to accomplish.Batch Normalization also makes training more resilient tothe parameter scale. Normally, large learning rates may in-crease the scale of layer parameters,  which then amplifythe gradient during backpropagation and lead to the modelexplosion. However, with Batch Normalization, backprop-agation through a layer is unaffected by the scale of its pa-rameters. Indeed, for a scalara,BN(Wu) =BN((aW)u)and thus∂BN((aW)u)∂u=∂BN(Wu)∂u, so the scale does not af-fect the layer Jacobian nor, consequently, the gradient prop-agation.  Moreover,∂BN((aW)u)∂(aW)=1a·∂BN(Wu)∂W, so largerweights lead tosmallergradients, and Batch Normalizationwill stabilize the parameter growth.We further conjecture that Batch Normalization may leadthe layer Jacobians to have singular values close to 1, whichis known to be beneficial for training (Saxe et al., 2013).Consider  two  consecutive  layers  with  normalized  inputs,and the transformation between these normalized vectors:̂z =F(̂x). If we assume that̂xand̂zare Gaussian and un-correlated, and thatF(̂x)≈Ĵxis a linear transformationfor the given model parameters, then botĥxand̂zhave unitcovariances, andI=Cov[̂z] =JCov[̂x]JT=JJT. Thus,Jis orthogonal, which preserves the gradient magnitudesduring backpropagation.  Although the above assumptionsare not true in reality,  we expect Batch Normalization tohelp make gradient propagation better behaved.   This re-mains an area of further study.4. Experiments4.1. Activations over timeTo  verify  the  effects  of  internal  covariate  shift  on  train-ing,  and the ability of Batch Normalization to combat it,we considered the problem of predicting the digit class onthe MNIST dataset (LeCun et al., 1998a).  We used a verysimple network, with a 28x28 binary image as input, and3 fully-connected hidden layers with 100 activations each.Each hidden layer computesy =g(Wu + b)with sigmoidnonlinearity,  and  the  weightsWinitialized  to  small  ran-dom  Gaussian  values.   The  last  hidden  layer  is  followed




## References:

Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
Sergey IoffeSIOFFE@GOOGLE.COMChristian SzegedySZEGEDY@GOOGLE.COMGoogle, 1600 Amphitheatre Pkwy, Mountain View
