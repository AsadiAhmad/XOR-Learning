# XOR-Learning
Learning XOR problem with 1 hidden layer and two perceptron with sigmoid activation function.

<div display=flex align=center>
  <img src="/Images/XOR_Learning.png" width="600px"/>
</div>

## Tech :hammer_and_wrench: Languages and Tools :

<div>
  <img src="https://github.com/devicons/devicon/blob/master/icons/python/python-original.svg" title="Python" alt="Python" width="40" height="40"/>&nbsp;
  <img src="https://github.com/devicons/devicon/blob/master/icons/jupyter/jupyter-original.svg" title="Jupyter Notebook" alt="Jupyter Notebook" width="40" height="40"/>&nbsp;
  <img src="https://assets.st-note.com/img/1670632589167-x9aAV8lmnH.png" title="Google Colab" alt="Google Colab" width="40" height="40"/>&nbsp;
  <img src="https://github.com/devicons/devicon/blob/master/icons/numpy/numpy-original.svg" title="Numpy" alt="Numpy" width="40" height="40"/>&nbsp;
  <img src="https://github.com/devicons/devicon/blob/master/icons/matplotlib/matplotlib-original.svg"  title="MatPlotLib" alt="MatPlotLib" width="40" height="40"/>&nbsp;
  <img src="https://cdn.worldvectorlogo.com/logos/seaborn-1.svg"  title="seaborn" alt="seaborn" width="40" height="40"/>&nbsp;
  <img src="https://github.com/devicons/devicon/blob/master/icons/scikitlearn/scikitlearn-original.svg"  title="Sci-kit Learn" alt="Sci-kit Learn" width="40" height="40"/>&nbsp;
</div>

- Python : Popular language for implementing Neural Network
- Jupyter Notebook : Best tool for running python cell by cell
- Google Colab : Best Space for running Jupyter Notebook with hosted server
- Numpy : Best Library for working with arrays in python
- MatPlotLib : Library for showing the charts in python
- SeaBorn : Best Library for beautifing MatPlotLib charts
- Sci-kit Learn : Best Library for implementing Neural Networks

## Run the Notebook on Google Colab

You can easily run this code on google colab by just clicking this badge [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AsadiAhmad/XOR-Learning/blob/main/XOR_Learning.ipynb)

## 📊 Mathematical Formulation

### **1. Forward Propagation (Two Layers)**

#### Hidden Layer (2 neurons)
Let the input vector be \(X = (x_1, x_2)\) and the target output \(y\).

- Weights for the hidden layer:  
  \( w_{11}, w_{12} \) for neuron 1  
  \( w_{21}, w_{22} \) for neuron 2  
- Bias terms for hidden neurons: \( b_1, b_2 \)

**Net Input to the Hidden Neurons:**
\[
z_1 = w_{11}x_1 + w_{12}x_2 + b_1
\]
\[
z_2 = w_{21}x_1 + w_{22}x_2 + b_2
\]

**Sigmoid Activation:**
\[
h_1 = \sigma(z_1) = \frac{1}{1 + e^{-z_1}}
\]
\[
h_2 = \sigma(z_2) = \frac{1}{1 + e^{-z_2}}
\]

---

#### Output Layer (1 neuron)
- Weights for the output neuron: \(v_1, v_2\)  
- Bias for the output neuron: \(b_o\)

**Net Input to Output Neuron:**
\[
z_o = v_1h_1 + v_2h_2 + b_o
\]

**Output (Sigmoid Activation):**
\[
\hat{y} = \sigma(z_o) = \frac{1}{1 + e^{-z_o}}
\]

---

### **2. Loss Function (Binary Cross-Entropy)**

The error between predicted and true outputs is calculated using the binary cross-entropy loss:

\[
L(y, \hat{y}) = -\sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]
\]

---

### **3. Backpropagation (Gradient Calculation)**

#### Error at Output Layer:
\[
\delta_o = \hat{y} - y
\]

#### Gradient for Output Layer Weights:
\[
\frac{\partial L}{\partial v_j} = \delta_o \cdot h_j
\]

#### Error at Hidden Layer:
\[
\delta_j = \delta_o \cdot v_j \cdot h_j \cdot (1 - h_j)
\]

#### Gradient for Hidden Layer Weights:
\[
\frac{\partial L}{\partial w_{ij}} = \delta_j \cdot x_i
\]

---

### **4. Weight Update (Gradient Descent Rule)**

\[
w_{ij} \leftarrow w_{ij} - \eta \frac{\partial L}{\partial w_{ij}}
\]

\[
v_j \leftarrow v_j - \eta \frac{\partial L}{\partial v_j}
\]

Where \( \eta \) is the learning rate.

---

## License

This project is licensed under the MIT License.
