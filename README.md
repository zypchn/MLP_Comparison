# Multi-Layer-Perceptron

## Introduction
Multi-Layer Perceptron (MLP) is an artificial neural network widely used for solving classification and regression tasks. MLP consists of fully connected dense layers that transform input data from one dimension to another. The purpose of an MLP is to model complex relationships between inputs and outputs. <br />
MLP has 3 main components:
- **Input Layer**: Each neuron in this layer corresponds to an input feature.
- **Hidden Layers**: An MLP can have any number of hidden layers, with each layer containing any number of nodes. These layers process the information received from the input layer.
- **Output Layer**: The output layer generates the final prediction or result.

Multi-Layer Perceptron Visualized: <br />
![download](https://media.geeksforgeeks.org/wp-content/uploads/nodeNeural.jpg)

### How it works?
**1) Forward Propagation**: The neuron computes the weighted sum of the inputs: $z = \sum_i w_ix_i + b$ <br />
Where:
- $x_i$ corresponds to the input feature
- $w_i$ is the corresponding weight
- $b$ is the bias term
​<br/>

**2) Activation Function**: The weighted sum $z$ is pased through an activation function. Common activation functions are:
  - **Sigmoid**: $\sigma(z) = \frac {1}{1 + e^{-z}}$
  - **ReLU (Rectified Linear Unit):** $f(z) = max(0,z)$
  - **Tanh (Hyperbolic Tangent):** $tanh(z) = \frac {2}{1+e^{-2z}} - 1$

**3) Loss Function**: Once the output is generated, the next step is to calculate the loss, using a loss function. For a classification problem, the commonly used *binary cross-entropy* loss function is: $L = -\frac{1}{N} \sum_{i=1}^{N} [ y_i log(\hat{y_i}) + (1-y_i)log(1-\hat{y_i}) ]$ <br />
Where:
- $y_i$ is the actual label
- $\hat{y_i}$ is the predicted label
- $N$ is the number of samples
<br />

**3) Backpropagation**: The gradients of the loss function with respect to each weight and bias are calculated using the chain rule of calculus. Then, the error is propagated back through the network, layer by layer.

**4) Optimization**: The network updates the weights and biases by moving in the opposite direction of the gradient to reduce the loss. There are various optimization algorithms, one of which being Stochastic Gradient Descent (SGD): $w = w-\eta \cdot \frac {\partial L}{\partial w}$
SGD updates the weights based on a singe sample or a small batch of data, the same approach we used in this study.

The purpuse of this study was to conduct a comparative analysis to demonstrate the capabilities of different MLP models. The models mainly differed on 2 fields: number of hidden layers and activation functions.

## Methods
The first step of the study was to find a dataset that can be quickly trained, that means approximately 1000-1500 instances and not so many features. For those reasons, I choose: [Bank Note Authentication UCI Data](https://www.kaggle.com/datasets/ritesaluja/bank-note-authentication-uci-data). The dataset contains features extracted from real-life banknote-like images. 5 features were extraced from the images: *variance*, *skewness*, *kurtosis*, *entropy*, *class*. Some of the terms explain themselves, but for the ones that don't:
- Skewness: Measure of the asymmetry of the probability distribution of a real-valued random variable about its mean
- Kurtosis: The sharpness of the peak of a frequency-distribution curve
<br />

Then, I defined 4 models with the following architectures:
- 2 Hidden Layers with *Tanh* as the activation function
- 2 Hidden Layers with *ReLU* as the activation function
- 3 Hidden Layers with *Tanh* as the activation function
- 3 Hidden Layers with *ReLU* as the activation function

Each hidden layer in every architecture has 5 nodes per layer. The other factors, like train/test split, or initial parameters are same for every architecture.

For the implementation process, I used the following libraries: 
- numpy:
- pandas:
- sklearn:

After every model was trained, I selected the one with the best performance. The selection process contains an arbitrary formula that I created: (accuracy / num_steps). It should be noted that I trained every model until it has a loss less than 0.25, so the *num_steps* is the first step that it exceeded the 0.25 threshold. The best model was then implemented using PyTorch to see if it can achieve the same results with the same parameters and activation function.
My expectation before the experiments was that 2 hidden layers with the ReLU activation function to be the best performing one.

## Results

### Resources:
- https://www.geeksforgeeks.org/multi-layer-perceptron-learning-in-tensorflow/

