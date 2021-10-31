---
layout: default
nav_order: 5
title: Logistic Regression Cost Function
---
## Appendix A: Detailed Development of Logistic Regression Cost Function

For convinience, let's re-write the Logistic Regression formulas 

#### Eq. 6: Logistic Regression Formula

#### 6a Logistic Regression Formula for y=1

$$p(y=1| x, w,b) = \hat{y}=\sigma(b+w^Tx) = \frac{1}{1+e^{^{-(b+w^Tx)}}}$$

#### 6b Logistic Regression Formula for y=0


$$p(y=0| x, w,b) = 1 - p(y=1| x, w,b) = 1-\hat{y}$$


Consequently, we can combine 6a and 6b to have an expression for \\(y\varepsilon [0,1]\\):

#### Eq. 7: Combined Logistic Regression Formula

$$p(y|x.b,w) =  \hat{y}^y(1- \hat{y})^{y-1} = \sigma(b+w^Tx) = \frac{1}{1+e^{^{-(b+w^Tx)}}}^y (1-\frac{1}{1+e^{^{-(b+w^Tx)}}})^(y-1)$$


It is easy to see that Eq. 7 is maximized if the prediction is correct, i.e. in case \\(\hat{y}\\)=1 and y=1, or in case \\(\hat{y}\\)=0 and y=0. 

In our scenario, the Training data set consists of m such data points {\hat{y}^i, y^i} i=1:m. The optimal predictor would be the one which maximizes the likelihhod og the Training data set's predictions, ie:

sequence equals the multiplication of the m So. the best predictor whold be the one which we have m data points, each have m data points, so we are looking for the set of parameters {b,w} which maximize all the obewe have the Training data set, which is a set of such observations. The challenge is to find the set of parameters 

We would like to find the If we could find the  the a good predictor, than we had a better chance So now the challenge is to find the parameters (b. w) with which the value of 

To find the 
For Logistic regression, instead of defining a the cost function, and then finding the set of parameters which minimize the cost, as was demonstrated for Linear Regression,  - which would not be convex, we will find the parameters which maximze the likelihood function

The likelihood function calculates the probability of the observed data. Maximum Likelihood refers to the optimization which finds the parameters which maximize the likelihood function, as shown in Eq. 8.

#### Eq. 8: Likelihood Function
$$L(b, w| y, x) = (p(Y| X, w,b) = 
\prod_{i=1}^{m}p(y^i|x^i, b,w)= 
\prod_{i=1}^{m}(\hat{y}^{(i)})^{y^{(i)}}(1-\hat{y}^{(i)})^{1-y^{(i)}}$$

So we are looking for the set of parameters {b,w} which mazimize the likelihhod function. Equivalently to that, we can look instead for the set of parameters {b,w} which minimize the negative of likelihhod function. We will go on the minimization option, which is consistent with the Cost minimization paradigm. 

Eq. 8 is not concave. Note that the non-concave charectaristic is common to the exponential family, which are only logarithmically concave. 
With that in mind, and considering that logarithms are strictly increasing functions, maximizing the log of the likelihood is equivalent to maximizing the likelihood. Not only that, but taking the log makes things much more covinient, as the multiplication are converted to a sum. 
So Here we take the natural log of Eq. 8 likelihood equation.

#### Eq. 9: Log Likelihood Function

$$logL(b,w|y,x)=\sum_{i=1}^{m}logp(y_i|x_i, b,w)$$

Pluging Eq. 7 into Eq 9, and following the common convention of denoting the log-likelihood with a lowercase l, we get: 

#### Eq. 10: Log Likelihood Function

$$l(b,w|y,x)=\sum_{i=1}^{m}logp(y_i|x_i, b,w)=\sum_{i=1}^{m}log( h_{b,w}(x_i)^y_i(1- h_{b,w}(x_i))^{y_i-1})$$


Consider the Logarithm power rule:

#### Eq. 11: Logarithm power rule:

$$log(x^ y) = y log(x)$$


Plug Eq. 11 into Eq 10 and get:
#### Eq. 12: Log Likelihood Function - Simplified


$$l(b,w|y,x)=\sum_{i=1}^{m}log( h_{b,w}(x_i)^y_i(1- h_{b,w}(x_i))^{y_i-1})=\sum_{i=1}^{m}y_ilogh_{b,w}(x_i)+(y_i-1)log(1-h_{b,w}(x_i))$$


Eq. 12 is the Likelihhod Function. Our optimal parameters set (b, w}, should maximize the Likelihood function, aka the Maximum Likelihood Estimator (MLE).
Equivalently to finding the MLE, is to find the minimal negative of the Likelihood. With the search of a minimum, we keep the same minimization paradigm like we did for the Cost function. So we will add a minus sign to Eq. 12, and call it Cost function - as shown in Eq. 13. Q.E.D.


#### Eq. 13: Cost Function

$$J(b,w) = -l(b,w)=\sum_{i=1}^{m}-y_ilogh_{b,w}(x_i)+(1-y_i)log(1-h_{b,w}(x_i))$$

Q.E.D.



