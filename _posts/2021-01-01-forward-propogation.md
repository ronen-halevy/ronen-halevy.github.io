---
layout: default
nav_order: 6
title: Introduction to Forward Propogation
---
# Forward Propogation

## Introduction

This is the second in series of 3 deep learning intro posts:
1. Introduction to Deep Learning which introduces the Deep Learning technology background, and presents network's building blocks and terms.
2. Forward Propogation, which presents the mathematical equations of the prediction path.

In this post we will examine the forwarding equations of the input data through the network. This is the network's prediction data path, at which the network's weights are static, and only the input data changes. As an oposite to Forward Propogation which calculates the prediction value based on the input data, the Backwards Propogation process calcuates the network's weights. The latter is presented in the next post.

## Network Schemes 

Figure 1 illustrates a Neural Network. The data is forwarded through 15 densely interconnected nodes. 

The forwarding propogation journey is executed in a layer by layer order, so nodes of layer l calculate their activation output, which is the input data of layer (l+1). 
Throughut this post we will present the forwarding equations based on Figure 1 as an example network.
 
 
 ### Figure 1: Neural Network
 ![Supervise Learning Outlines](../assets/images/neural-networks/deep-neural-network.png)




## Forwarding Propogation with Scalar Equations

Based on the network exapmple of Figure 1, this section presents the forwarding equations of each of the 5 layers, where the input of layer l is the output of layer l-1, for l=2-5. 
Note that the subscripts and superscript conventions are as the following:
- Superscript index in square brackets: Layer index
- Subscript index: node index within the layer
- 2 Subscripts (used for weights indexing): First index is the source node index, and second index is of the destination node.

The goal of this detailed description is to give a detailed forwarding example, with a detailed notations of all the parameters involved and their assigned indices.

So here are the 5 layers equations, listed within the cascaded nodes' sketches:



 ### Layer 1 Forwarding  Equations
 
![neuron_cascaded_operator](../assets/images/neural-networks/L1_neuron_cascaded_operator.png)


 ### Layer 2 Forwarding  Equations
 
![neuron_cascaded_operator](../assets/images/neural-networks/L2_neuron_cascaded_operator.png)

 
 ### Layer 3 Forwarding  Equations
![neuron_cascaded_operator](../assets/images/neural-networks/L3_neuron_cascaded_operator.png)

 ### Layer 4 Forwarding  Equations
 
![neuron_cascaded_operator](../assets/images/neural-networks/L4_neuron_cascaded_operator.png)


 ### Layer 5 Forwarding  Equations
 
![neuron_cascaded_operator](../assets/images/neural-networks/L5_neuron_cascaded_operator.png)
 
 
 ## Forwarding Propogation with Vector (Matrix) Equations
 
 
The previous section presented the detailed forwaring equations in a scalar form. Goal was to give a detailed example of all the operators and parameters. This section lists the vectorized forwarding equations corresponding to Figure 1, the 5 layers example network. The equations are equivalent to those presented in the previous section, but now in the more computationally efficient and also presentationally compact vectorized form. After specifing the vectorized equations for all 5 layers, the generalized layer forwarding equations are specified.
Note there is no single matrix equation which solves the entire network, but a vectorized seperated equations each layer, as presented next.
Figure 2 presents the vectorized forwarding flow, specifing the vectorized operation at each layer. Following it are the vectorized equation which are a breakdown of the vectorized equations. The input to the first layer, denoted by the vector \\(\bar{x}\\) so far, is now denoted by \\(\bar{a}^{[0]}\\), so that layer 1 notations are similar to all other layers.
 
  
 ![neuron_cascaded_operator](../assets/images/neural-networks/forwarding-vectorized-flow.png)



### Eq. 1-a: Layer 1 Weighted input

\\(\begin{bmatrix}
z_1^{[1]} \\\\\\ 
z_2^{[1]} \\\\\\ 
z_3^{[1]}  \\\\\\ 
z_4^{[1]}
\end{bmatrix}=
\begin{bmatrix}
w_{11}^{[1]}  & w_{21}^{[1]} & w_{31}^{[1]} \\\\\\ 
w_{12}^{[1]}  & w_{22}^{[1]} & w_{32}^{[1]} \\\\\\ 
w_{13}^{[1]}  & w_{23}^{[1]} & w_{33}^{[1]} \\\\\\ 
w_{14}^{[1]}  & w_{24}^{[1]} & w_{34}^{[1]}
\end{bmatrix} \begin{bmatrix}
a_1^{[0]}  \\\\\\ 
a_2^{[0]} \\\\\\
a_3^{[0]}
\end{bmatrix}+\begin{bmatrix}
b_1^{[1]} \\\\\\ 
b_2^{[1]} \\\\\\ 
b_3^{[1]} \\\\\\ 
b_4^{[1]} 
\end{bmatrix}
\\)

### Eq. 1-b: Layer 1 activation

\\(\begin{bmatrix}
a_1^{[1]} \\\\\\ 
a_2^{[1]} \\\\\\ 
a_3^{[1]} \\\\\\ 
a_4^{[1]}
\end{bmatrix}=\begin{bmatrix}
g_1^{[1]}(z_1^{[1]}) \\\\\\ 
g_2^{[1]}(z_2^{[1]}) \\\\\\ 
g_3^{[1]}(z_3^{[1]}) \\\\\\ 
g_4^{[1]}(z_4^{[1]})
\end{bmatrix}\\)

### Eq. 2-a: Layer 2 Weighted input

\\(\begin{bmatrix}
z_1^{[2]} \\\\\\ 
z_2^{[2]} \\\\\\
z_3^{[2]} \\\\\\ 
z_4^{[2]}
\end{bmatrix}=
\begin{bmatrix}
w_{11}^{[2]}  & w_{21}^{[2]} & w_{31}^{[2]} & w_{41}^{[2]}\\\\\\
w_{12}^{[2]}  & w_{22}^{[2]} & w_{32}^{[2]} & w_{42}^{[2]}\\\\\\ 
w_{13}^{[2]}  & w_{23}^{[2]} & w_{33}^{[2]} & w_{43}^{[2]}\\\\\\ 
w_{14}^{[2]}  & w_{24}^{[2]} & w_{34}^{[2]} & w_{44}^{[2]}
\end{bmatrix} \begin{bmatrix}
a_1^{[1]}  \\\\\\ 
a_2^{[1]} \\\\\\
a_3^{[1]} \\\\\\
a_3^{[1]}
\end{bmatrix}+\begin{bmatrix}
b_1^{[2]} \\\\\\ 
b_2^{[2]} \\\\\\ 
b_3^{[2]} \\\\\\ 
b_4^{[2]} 
\end{bmatrix}\\)

### Eq. 2-b: Layer 2 activation

\\(\begin{bmatrix}
a_1^{[2]} \\\\\\ 
a_2^{[2]} \\\\\\ 
a_3^{[2]} \\\\\\ 
a_4^{[2]}
\end{bmatrix}=\begin{bmatrix}
g_1^{[2]}(z_1^{[2]}) \\\\\\ 
g_2^{[2]}(z_2^{[2]}) \\\\\\ 
g_3^{[2]}(z_3^{[2]}) \\\\\\ 
g_4^{[2]}(z_4^{[2]})
\end{bmatrix}\\)


### Eq. 3-a: Layer 3 Weighted input


\\(\begin{bmatrix}
z_1^{[3]} \\\\\\ 
z_2^{[3]} \\\\\\ 
z_3^{[3]}  \\\\\\ 
z_4^{[3]}
\end{bmatrix}=
\begin{bmatrix}
w_{11}^{[3]}  & w_{21}^{[3]} & w_{31}^{[3]} & w_{41}^{[3]}\\\\\\ 
w_{12}^{[3]}  & w_{22}^{[3]} & w_{32}^{[3]} & w_{42}^{[3]}\\\\\\ 
w_{13}^{[3]}  & w_{23}^{[3]} & w_{33}^{[3]} & w_{43}^{[3]}\\\\\\ 
w_{14}^{[3]}  & w_{24}^{[3]} & w_{34}^{[3]} & w_{44}^{[3]}
\end{bmatrix} \begin{bmatrix}
a_1^{[2]}  \\\\\\ 
a_2^{[2]} \\\\\\
a_3^{[2]} \\\\\\
a_3^{[2]}
\end{bmatrix}+\begin{bmatrix}
b_1^{[3]} \\\\\\ 
b_2^{[3]} \\\\\\ 
b_3^{[3]} \\\\\\ 
b_4^{[3]} 
\end{bmatrix}\\)

### Eq. 3-b: Layer 3 activation



\\(\begin{bmatrix}
a_1^{[3]} \\\\\\ 
a_2^{[3]} \\\\\\ 
a_3^{[3]} \\\\\\ 
a_4^{[3]}
\end{bmatrix}=\begin{bmatrix}
g_1^{[3]}(z_1^{[4]}) \\\\\\ 
g_2^{[3]}(z_2^{[4]}) \\\\\\ 
g_3^{[3]}(z_3^{[4]}) \\\\\\ 
g_4^{[3]}(z_4^{[4]})
\end{bmatrix}\\)


### Eq. 4-a: Layer 4 Weighted input


\\(\begin{bmatrix}
z_1^{[4]} \\\\\\ 
z_2^{[4]}
\end{bmatrix}=
\begin{bmatrix}
w_{11}^{[4]}  & w_{21}^{[4]} & w_{31}^{[4]} & w_{41}^{[4]} \\\\\\ 
w_{12}^{[4]}  & w_{22}^{[4]} & w_{32}^{[4]} & w_{42}^{[4]}
\end{bmatrix} \begin{bmatrix}
a_1^{[3]}  \\\\\\ 
a_2^{[3]} \\\\\\
a_3^{[3]} \\\\\\
a_3^{[3]}
\end{bmatrix}+\begin{bmatrix}
b_1^{[4]} \\\\\\ 
b_2^{[4]}
\end{bmatrix}\\)


### Eq. 4-b: Layer 4 activation


\\(\begin{bmatrix}
a_1^{[4]} \\\\\\ 
a_2^{[4]}
\end{bmatrix}=\begin{bmatrix}
g_1^{[4]}(z_1^{[4]}) \\\\\\ 
g_2^{[4]}(z_2^{[4]}) \\\\\\ 
g_3^{[4]}(z_3^{[4]}) \\\\\\ 
g_4^{[4]}(z_4^{[4]})
\end{bmatrix}\\)


### Eq. 5-a: Layer 5 Weighted input


\\(z_1^{[5]}=
\begin{bmatrix}
w_{11}^{[5]}  & w_{21}^{[5]}
\end{bmatrix} \begin{bmatrix}
a_1^{[4]} \\\\\\
a_2^{[4]}
\end{bmatrix}+b_1^{[5]}\\)


### Eq. 5-b: Layer 5 activation

\\(a_1^{[5]}=
g^{[5]}(z_1^{[5]})\\)

Next section extends Eq. 5: while the above section regards to the Feed Forwarding of a single node, next section presents the Feed Forward equations for an entire layer l.

 ## Vectorized Feed Forward Equations
 
 Eq. 6 shows the Feed Forwarding epressions for any layer l, 0<l<L, and a single data exempale vector \\(\bar(x)\\) denoted here by \\(\bar(a)^{[0]}\\) 
 
 ### Eq. 6: Vectorized Feed Forward Equations for Layer l 
 
 #### Eq. 6a: Vectorized Feed Forward Equations  - Weighted input
 $$
 \bar{z}^{[l]}=\bar{w}^{[l]}\bar{a}^{[l-1]}+\bar{b}^{[l]}
 $$
 
 #### Eq. 6b: Vectorized Feed Forward Equations - activation

$$a^{[l]}=
g^{[l]}(z^{[l]})$$


Eq.6 vectors and matrix dimenssions are:
 - \\(\bar{z}^{[l]}\\) : n(l) x 1
 - \\(\bar{w}^{[l]}\\) : n(l) x n(l-1)
 - \\(\bar{a}^{[l-1]}\\) : n(l-1) x 1
 - \\(\bar{b}^{[l]}\\) : n(l) x 1

Where n(l) is the number of nodes in layer l.
 
Next section extends Eq. 6 a bit more: while the above section regarded the input vector as a vector of size 1 x n(l-1), next section presents the Feed Forward equations for an input data set with m examples.

## Vectorized Feed Forward Across Multiple Examples

Eq. 6a and Eq. 6b are the forwarding equations for a single data input vector. To even more generalized case is the forwarding equation for all training exam or any batch of examples. These multi-examples equations are basically the same as Eq. 6, except the dimensions of the various vector change to a matrix structure as shown in Eq. 7a and Eq. 7b. Accordingly, we the matrix will be denoted in capital letters

### Eq. 7: Vectorized Feed Forward Equations for Layer l Across m Examples

#### Eq. 7a: Vectorized Feed Forward Equations Across m Examples - Weighted input
 $$
 \bar{Z}^{[l]}=\bar{w}^{[l]}\bar{A}^{[l-1]}+\bar{b}^{[l]}
 $$
 
 #### Eq. 7b: Vectorized Feed Forward Equations Across m Examples - activation

$$\bar{A}^{[l]}=
g^{[l]}(\bar{Z}^{[l]})$$


Where \\(A^{[l]}\\) and  \\(Z^{[l]}\\) are now matrices, each column of which relates to an input data example \\(m\epsilon{M}\\) , like so:

\\(\bar{Z}^{[l]}=\begin{bmatrix}z_1^{[l]{(1)}}& z_1^{[l]{(2)}} & . & . & z_1^{[l]{(m)}}\\\\\\  
 z_2^{[l]{(1)}}& z_2^{[l]{(2)}} & . & . & z_2^{[l]{(m)}}\\\\\\  
 z_3^{[l]{(1)}}& z_3^{[l]{(2)}} & . & . & z_3^{[l]{(m)}}\\\\\\ 
 .& . & . & . &. \\\\\\ 
 . & . &.  & . & . \\\\\\ 
 z_n^{[l]{(1)}}&z_n^{[l]{(2)}}  & . & . & z_n^{[l]{(m)}}\end{bmatrix}\\)


\\(\bar{A}^{[l]}=\begin{bmatrix}a_1^{[l]{(1)}}& a_1^{[l]{(2)}} & . & . & a_1^{[l]{(m)}}\\\\\\  
 a_2^{[l]{(1)}}& a_2^{[l]{(2)}} & . &  .& a_2^{[l]{(m)}}\\\\\\
 a_3^{[l]{(1)}}& a_3^{[l]{(2)}} & . & . & a_3^{[l]{(m)}}\\\\\\ 
 .& . &  .&.  &. \\\\\\
 . & . & . & . & .\\\\\\ 
 a_n^{[l]{(1)}}&a_n^{[l]{(2)}}  & . & . & a_n^{[l]{(m)}}\end{bmatrix}\\)


Eq.7 matrix dimenssions are:

 - \\(\bar{Z}^{[l]}\\) : n(l) x m
 - \\(\bar{w}^{[l]}\\) : n(l) x n(l-1)
 - \\(\bar{A}^{[l-1]}\\) : n(l-1) x m 
 - \\(\bar{b}^{[l]}\\) : n(l) x 1

Where:
- n(l) is the number of nodes in layer l, 
- m is the number of training examples, 

So, as an example, \\(z_2^{[l]{(m)}}\\) means: z of second nodes in lth layer and mth example.


Note about matrix addition:   In Eq. 7a, the dimensions of first summand In Eq. 7a n(l) x m, so the n(l) x 1  vector \\(\bar{b}^{[l]}\\) is added us broadcasting, i.e. it is added to each of the m columns.

## Overall Feed Forward

Eq. 7b expresses the value of the activation matrix \\(A^{[l]}\\) as a function of \\(Z^{[l]}\\). The latter, \\(Z^{[l]}\\), is expressed as a function of \\(A^{[l-1]}\\), i.e. previous layer's activation matrix. and so forth. So, express in a single eqaution the expression for \\(A^{[l]}\\) as a composition of all its predecessor layers.

Just one cosmetic modification to simplify the equation: Assume the bias in Eq. 7b is included in the weights matrix w (the first row holds the l biases), and accordingly the first row of A matrxis all 1s.
Example:
\\(\begin{bmatrix}
z_1^{[4]} \\\\\\ 
z_2^{[4]}
\end{bmatrix}=
\begin{bmatrix}
w_{11}^{[4]}  & w_{21}^{[4]} & w_{31}^{[4]} & w_{41}^{[4]} \\\\\\ 
w_{12}^{[4]}  & w_{22}^{[4]} & w_{32}^{[4]} & w_{42}^{[4]}
\end{bmatrix} \begin{bmatrix}
a_1^{[3]}  \\\\\\ 
a_2^{[3]} \\\\\\
a_3^{[3]} \\\\\\
a_3^{[3]}
\end{bmatrix}+\begin{bmatrix}
b_1^{[4]} \\\\\\ 
b_2^{[4]}
\end{bmatrix}\\)

Will now be:

\\(\begin{bmatrix}
z_1^{[4]} \\\\\\ 
z_2^{[4]}
\end{bmatrix}=
\begin{bmatrix}
b_{1}^{[4]}  & w_{11}^{[4]}  & w_{21}^{[4]} & w_{31}^{[4]} & w_{41}^{[4]} \\\\\\ 
w_{2}^{[4]}  & w_{22}^{[4]} & w_{32}^{[4]} & w_{42}^{[4]}
\end{bmatrix} \begin{bmatrix}
1  \\\\\\ 
a_1^{[3]}  \\\\\\ 
a_2^{[3]} \\\\\\
a_3^{[3]} \\\\\\
a_3^{[3]}
\end{bmatrix}\\)


Now let's have the entire composition:

### Eq. 8: Vectorized Feed Forward Combined Equation

\\(A^{[l]}=g^{[l]}(w^{[l]}g^{[l-1]}(w^{[l-1]}g^{[l-2]}(w^{[l-2]}g^{[l-3]}(......g^{[1]}(w^{[1]}A^{[0]}))))\\)

We will get to Eq. 8 in the Backwards Propogation post.

## Next steps
 
The next post in this series is about Backwards propogation, which is activated durimg the traing phase, aka fitting, to calculate optimized values for the network's wheights and biases.


