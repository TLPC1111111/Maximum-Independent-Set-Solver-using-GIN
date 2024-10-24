# Maximum Independent Set Solver using GIN

This project implements a neural network-based solver for the Maximum Independent Set (MIS) problem using Graph Isomorphism Networks (GIN) from the PyTorch Geometric library. The network architecture consists of multiple GIN convolutional layers followed by fully connected layers to process graph-structured data.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [License](#license)

## Installation

To get started, you need to install the required libraries. You can do this using pip:

```bash
pip install torch torch-geometric numpy
```

Make sure you have PyTorch installed according to your system's specifications. You can find the installation instructions for PyTorch here.

## Usage


### 1. Clone the Repository
```bash
git clone https://github.com/TLPC1111111/Maximum-Independent-Set-Solver-using-GIN.git
cd Q-Learning-Graph-Neural-Network-GNN-Model
```

### 2. Install Dependencies
```bash
pip install torch torch-geometric tensorboard
```

### 3. Training the Model
To train the model, simply run:

```bash
python train.py
```

## MLP Class
The `MLP` class defines a multi-layer perceptron (MLP) architecture that is used as the neural network component in GIN convolutions. You can instantiate this class with a list specifying the number of neurons in each layer.

```python
mlp = MLP(size_list=[1, 64, 64, 1], batch_norm=True, dropout=0.5)
```

## MISSolver Class
The `MISSolver` class encapsulates the entire GIN architecture designed to solve the MIS problem. You can create an instance of the `MISSolver` class and then call its `forward` method to compute the loss and conditional expectation.

```python
solver = MISSolver()
loss, mis = solver(w, edge_index, batch)
```
- **Parameters:**:
  - `w`:Node feature matrix.
  - `edge_index`:Edge index tensor representing the graph structure.
  - `batch`:Batch vector that indicates the graph each node belongs to.

## Model Architecture
The `MISSolver` is composed of several GIN convolutional layers followed by batch normalization and activation functions:

1.**GINConv Layers**:
- Input: 1D feature vector for each node.
- Several GIN layers with 64 hidden units each.
- Final GIN layer outputs a single feature for each node.

2.**Batch Normalization**: Applied after each convolutional layer to stabilize learning.

3.**Activation Functions**: ReLU activation is used throughout the network.

4.**Loss Calculation**: The model calculates the loss based on the defined loss_fn, which computes the MIS-related loss.

## Training
Training the model involves defining a loss function and an optimizer. Here is a simple example of how you might set this up:

```python
optimizer = torch.optim.Adam(solver.parameters(), lr=0.001)

# Example training loop
for epoch in range(num_epochs):
    optimizer.zero_grad()
    loss, mis = solver(w, edge_index, batch)
    loss.backward()
    optimizer.step()
```

Make sure to replace `w`, `edge_index`, and `batch` with your actual data.


## Evaluation
After training, you can evaluate the model by checking the selected nodes in the independent set returned by the `conditional_expectation` method:

```python
selected_nodes = solver.conditional_expectation(w, prob_dense, adj, loss_thresholds, gammas, prob_mask)
```




