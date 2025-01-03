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

## Mathematical Formulation

### Hidden Layer (2 neurons)

```math
z_1 = w_{11}x_1 + w_{12}x_2 + b_1
```

```math
z_2 = w_{21}x_1 + w_{22}x_2 + b_2
```

```math
h_1 = \sigma(z_1) = \frac{1}{1 + e^{-z_1}}
```

```math
h_2 = \sigma(z_2) = \frac{1}{1 + e^{-z_2}}
```

### Output Layer (1 neuron)

```math
z_o = v_1h_1 + v_2h_2 + b_o
```

```math
\hat{y} = \sigma(z_o) = \frac{1}{1 + e^{-z_o}}
```

## License

This project is licensed under the MIT License.
