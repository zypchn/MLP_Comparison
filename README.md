# Multi-Layer-Perceptron

## Introduction
Multi-Layer Perceptron (MLP) is an artificial neural network widely used for solving classification and regression tasks. MLP consists of fully connected dense layers that transform input data from one dimension to another. The purpose of an MLP is to model complex relationships between inputs and outputs. <br />
MLP has 4 main components:
- Input Values
- Weights and Biases
- Weighted Sum
- Step Function <br />

Perceptron Visualized: <br />
![download](https://github.com/user-attachments/assets/20002179-b518-48a8-8a29-840920bc98d8)

### How it works?
I will explain the steps of the perceptron below:
1) A weight is assigned to each input node of a perceptron, indicating the importance of that input in determining the output.
2) Compute the weighted sum: $$z = w_1x_1 + w_2x_2 + ... w_nx_n = X^TW$$ <br />
Where $X^T$ is the transposed input matrix, and $W$ is the weight matrix. The bias should be added after the summation, so the final result will be: $X^TW + b$$
3) The weighted sum is given to the step function (namely, Heaviside step function). The step function compares this weighted sum to a threshold. If the input is larger than the threshold value, the output is 1; otherwise, it's 0. <br />
$0 \quad if \quad z < Threshold \brace 
1 \quad if \quad z \geq Threshold$
4) Update the weights using the perceptron learning rule formula: $w_i,_j = w_i,_j + \eta (y_j - \hat{y}_j)x_i$.
Where:
- $$w_i,_j$$ is the weight between the $i^{th}$ input and $$j^{th}$ output neuron,
- $x_i$ is the $i^{th}$ input value,
- $y_i$ is the actual value and $\hat{y}_j)x_i$ is the predicted value,
- $\eta$ is the learning rate, controlling how much the weights are adjusted.
​

