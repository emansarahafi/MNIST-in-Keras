# MNIST Digit Recognition with Keras

A comprehensive tutorial for building and training a neural network to recognize handwritten digits using the MNIST dataset and Keras.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/emansarahafi/MNIST-in-Keras/blob/main/MNIST_in_Keras_EmanSarahAfi.ipynb)

## 📋 Overview

This project demonstrates how to build a simple yet effective neural network for digit recognition using Keras. The tutorial covers the complete machine learning workflow from data loading and preprocessing to model training and evaluation.

**Original Author:** Xavier Snelgrove  
**Modified by:** Eman Sarah Afi

## 🎯 Features

- Load and visualize the MNIST dataset
- Data preprocessing and normalization
- Build a multi-layer perceptron (MLP) neural network
- Train the model with categorical crossentropy loss
- Evaluate model performance on test data
- Visualize correct and incorrect predictions

## 🚀 Getting Started

### Prerequisites

- Python 3.x
- pip (Python package manager)

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/emansarahafi/MNIST-in-Keras.git
   cd MNIST-in-Keras
   ```

2. **Set up a virtual environment (recommended):**

   ```bash
   pip install virtualenv
   virtualenv kerasenv
   source kerasenv/bin/activate  # On Windows: kerasenv\Scripts\activate
   ```

3. **Install required packages:**

   ```bash
   pip install numpy jupyter keras matplotlib
   ```

### Running the Notebook

Launch Jupyter Notebook:

```bash
jupyter notebook MNIST_in_Keras_EmanSarahAfi.ipynb
```

Or open directly in [Google Colab](https://colab.research.google.com/github/emansarahafi/MNIST-in-Keras/blob/main/MNIST_in_Keras_EmanSarahAfi.ipynb).

## 🧠 Model Architecture

The neural network consists of:

- **Input Layer:** 784 neurons (28×28 flattened images)
- **Hidden Layers:** Dense layers with dropout for regularization
- **Output Layer:** 10 neurons (one for each digit 0-9) with softmax activation
- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy

## 📊 Dataset

The [MNIST dataset](http://yann.lecun.com/exdb/mnist/) contains:

- 60,000 training images
- 10,000 test images
- Handwritten digits (0-9)
- 28×28 pixel grayscale images

## 🔍 What You'll Learn

- Loading datasets with Keras
- Data preprocessing and normalization techniques
- Converting labels to one-hot encoding
- Building sequential models in Keras
- Training neural networks
- Evaluating model performance
- Visualizing predictions and misclassifications

## 📈 Results

The model achieves strong performance on the MNIST test set, with detailed accuracy metrics provided in the notebook. You can inspect both correct and incorrect predictions to understand the model's behavior.

## 🛠️ Technologies Used

- **Keras**: High-level neural networks API
- **NumPy**: Numerical computing
- **Matplotlib**: Data visualization
- **Jupyter**: Interactive notebook environment

## 📚 Additional Resources

- [Keras Documentation](https://keras.io)
- [Keras GitHub Repository](https://github.com/fchollet/keras)
- [MNIST Database](http://yann.lecun.com/exdb/mnist/)
- [Cross-Entropy Loss Explained](https://en.wikipedia.org/wiki/Cross_entropy)

## 📝 License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.

## 🙏 Acknowledgments

- Original tutorial by Xavier Snelgrove for the University of Toronto
- Based on the `mnist_mlp.py` example from the Keras source code
- MNIST dataset created by Yann LeCun and colleagues
