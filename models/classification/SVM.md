# Support Vector Machines (SVM) - Detailed Theory:
## 1. Intuition:

SVM is a supervised machine learning algorithm primarily used for classification (though it can 
be used for regression). The core idea is to find a hyperplane that best separates data into 
classes. In the context of SVM, the best hyperplane is the one that maximizes the margin between 
two classes.

## 2. Maximizing the Margin:
Imagine plotting your data on a graph, and you want to find a line (in 2D), a plane (in 3D), or 
a hyperplane (in more than 3 dimensions) that best divides the data into classes. This "divider" 
should be equidistant from the closest data point of each class, and these points are known as 
"support vectors". The distance between the support vectors and the hyperplane is the margin.

The goal of SVM is to maximize this margin. 

## 3. Mathematical Formulation:
For a linearly separable dataset with data points $x$ and labels $y$ (where $y$ can be -1 or 1 
for a binary classification):
The equation of the hyperplane is given by:

$$
w \cdot x+b=0
$$
Where:
- $w$ is the normal vector to the hyperplane.
- $b$ is the bias.
The goal is to find the optimal $w$ and $b$.

For the decision boundary (the actual hyperplane separating the data):
$$
y_i\left(w \cdot x_i+b\right)-1 \geq 0
$$
For all $i$ data points.
The width of the margin, $M$, is:
$$
M=\frac{2}{\|w\|}
$$
We want to maximize $M$, which is equivalent to minimizing $\frac{1}{2}\|w\|^2$.

## 4. Constrained Optimization:

We frame the SVM problem as a constrained optimization problem:
Minimize:
$$
\frac{1}{2}\|w\|^2
$$
Subject to:
$$
y_i\left(w \cdot x_i+b\right)-1 \geq 0
$$
For all $i$.
This is a quadratic optimization problem with linear constraints.

5. Lagrange Multipliers and Dual Form:
To solve the above optimization problem, we use Lagrange multipliers. We introduce a Lagrange 
multiplier $\alpha_i$ for each constraint:
The Lagrangian is:
$$
L(w, b, \alpha)=\frac{1}{2}\|w\|^2-\sum_{i=1}^N \alpha_i\left[y_i\left(w \cdot 
x_i+b\right)-1\right]
$$
Where $N$ is the number of data points.
The dual problem is to maximize the Lagrangian with respect to the multipliers, $\alpha$, while 
minimizing it with respect to $w$ and $b$.

## 6. Kernel Trick:
For non-linearly separable data, we can map data to a higher-dimensional space where it becomes 
linearly separable. This mapping is done using a kernel function without explicitly transforming 
each data point. Popular kernels include the polynomial kernel, radial basis function (RBF), and 
sigmoid kernel. 

## 7. Soft Margin SVM:

In the real world, data is often noisy and may not be linearly separable. Soft Margin SVM 
introduces a slack variable, $\xi$, to allow some misclassifications. The goal then becomes 
minimizing:
$$
\frac{1}{2}\|w\|^2+C \sum_{i=1}^N \xi_i
$$
Where $C$ is a regularization parameter that determines the trade-off between maximizing the 
margin and minimizing classification errors.

