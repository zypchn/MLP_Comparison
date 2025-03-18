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
The dataset was splitted as 80% training and 20% as test.

Then, I defined 4 models with the following architectures:
- 2 Hidden Layers with *Tanh* as the activation function
- 2 Hidden Layers with *ReLU* as the activation function
- 3 Hidden Layers with *Tanh* as the activation function
- 3 Hidden Layers with *ReLU* as the activation function

Each hidden layer in every architecture has 5 nodes per layer. The other factors, like train/test split, or initial parameters are same for every architecture. Every model was trained with the learning rate of 1e-2.

For the implementation process, I used the following libraries: 
- numpy: 1.26.4
- pandas: 2.2.3
- sklearn: 1.2.2

After every model was trained, I selected the one with the best performance. The selection process contains an arbitrary formula that I created: (accuracy*100 / num_steps). It should be noted that I trained every model until it has a loss less than 0.20, so the *num_steps* is the first step that it exceeded the 0.20 threshold. The initial num_steps for each model was set to 5000. If a model failed to exceed the threshold, the number of steps was increased to 10000. If it still failed, the training process was stopped. This means that any model unable to surpass the 0.20 threshold within 10000 iterations was penalized with -1 point. The best model was then implemented using PyTorch to see if it can achieve the same results with the same parameters and activation function.
My expectation before the experiments was that 2 hidden layers with the ReLU activation function to be the best performing one.

## Results
The table that compares the architectures with their accuracy, number of steps and selection score are shown below:
| Architecture | Avg. Accuracy | Number of Steps | Selection Score (penalties included) |
|--|--|--|--|
| 2 Hidden Layers & Tanh | 0.9673 | 644 | 0.1502
| 2 Hidden Layers & ReLU | 0.9309 | 10000 | -0.9906
| 3 Hidden Layers & Tanh | 0.9745 | 3645 | 0.0267
| 3 Hidden Layers & Relu | 0.9200 | 10000 | -0.9908

Each model's metrics (avg. accuracy, precision, recall, F1 score), confusion matrix and classification report are also shown below:

- *2 Hidden Layers* and *Tanh*: <br />
![image](https://github.com/user-attachments/assets/8ab1561c-7a38-4acf-912f-54524a6788d0)

- *2 Hidden Layers* and *ReLu*: <br />
![image](https://github.com/user-attachments/assets/27d4d3f7-3bc7-4d1d-ad29-ad9ca95cee7f)

- *3 Hidden Layers* and *Tanh*: <br />
![image](https://github.com/user-attachments/assets/6dafdbd3-c27d-425e-9c5a-1b6f094757de)

- *3 Hidden Layers* and *ReLu*: <br />
![image](https://github.com/user-attachments/assets/69208cfc-d4dc-44fe-af50-b7e1c51c2a29)

- *2 Hidden Layers* and *Tanh* (PyTorch): <br />
![image](https://github.com/user-attachments/assets/d7efaf8b-0cdf-47fc-be8f-9872525086d4)



## Discussions
Although every model performs well with more that 90% accuracy, there are some things to be considered. First of all, I did not preprocess the data. ut considering that all models performed well on this dataset, it may be said that tha data was preprocessed beforehand. We actually don't know if it's the case or not because the data card is empty. Secondly, models with *ReLU* activation function took far too long to reach to the threshold than their counterparts (*tanh*). Possible reasons for that may include:
- The ReLU function outputs values in the range [0, ∞), which can lead to dying ReLU issues (where neurons get stuck at zero and stop learning). This slows down convergence, especially if the initialization isn't optimal.
- In shallow networks, ReLU might take longer to optimize because it does not have a symmetric gradient flow like Tanh.

Possible fixes regarding ReLU convergence issue:
- Use Leaky ReLU to avoid dead neurons
- Use Adam optimizer
- Change the learning rate

Finall PyTorch implementation worked better in terms of loss and various metrics. It outperformed it's counterpart on every measurement. This could happen because of the fact that we used tensors instead pf numpy arrays, or that PyTorch functions are more stable that the ones we implemented manually.

Further experimentation is required to explore these solutions.

### Resources:
- https://www.geeksforgeeks.org/multi-layer-perceptron-learning-in-tensorflow/

