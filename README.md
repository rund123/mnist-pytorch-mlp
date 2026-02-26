# MNIST Handwritten Digit Classification -- PyTorch MLP

This project implements a **Fully Connected Neural Network (MLP)** using
**PyTorch** to classify handwritten digits from the MNIST dataset.\
The model is trained, validated, and evaluated on the standard MNIST
benchmark dataset.

------------------------------------------------------------------------

##  Project Overview

This project demonstrates:

-   Loading and preprocessing image data using `torchvision`
-   Building a multi-layer perceptron (MLP) from scratch
-   Implementing structured training and validation pipelines
-   Evaluating model performance on a test dataset
-   Plotting training and validation loss curves
-   Saving trained model weights for future reuse

------------------------------------------------------------------------

##  Dataset

This project uses the **MNIST** dataset:

-   70,000 grayscale images
-   Image size: 28 × 28 pixels
-   10 classes (digits 0--9)
-   60,000 training images
-   10,000 test images

### Preprocessing Steps

1.  `ToTensor()` → Converts images to PyTorch tensors\
2.  `Normalize((0.5,), (0.5,))` → Normalizes pixel values to improve
    convergence

------------------------------------------------------------------------

##  Model Architecture

The MLP architecture consists of:

-   **Input Layer:** 784 neurons (28×28 flattened image)
-   **Hidden Layer 1:** 512 neurons + ReLU
-   **Hidden Layer 2:** 256 neurons + ReLU
-   **Hidden Layer 3:** 128 neurons + ReLU
-   **Output Layer:** 10 neurons (digit classes)

### Training Configuration

-   **Loss Function:** CrossEntropyLoss\
-   **Optimizer:** SGD\
-   **Momentum:** 0.9\
-   **Learning Rate:** 0.001\
-   **Epochs:** 25\
-   **Batch Size:** 250

------------------------------------------------------------------------

##  Installation

Clone the repository:

git clone
https://github.com/`<your-username>`{=html}/mnist-pytorch-mlp.git\
cd mnist-pytorch-mlp

Install dependencies:

pip install -r requirements.txt

------------------------------------------------------------------------

##  Usage

Run the training script:

python mnist_mlp.py

The script will:

-   Train the model\
-   Validate performance each epoch\
-   Print training and validation accuracy\
-   Plot loss curves\
-   Save the trained model to:

saved_model/mnist_mlp.pth

------------------------------------------------------------------------

##  Results

-   Training and validation loss decrease steadily over epochs\
-   Final test accuracy: \~93--94%\
-   Accuracy can be improved further with hyperparameter tuning

------------------------------------------------------------------------

##  Future Improvements

-   Replace MLP with a Convolutional Neural Network (CNN)\
-   Use Adam optimizer\
-   Add a learning rate scheduler\
-   Implement early stopping\
-   Add model checkpointing\
-   Improve experiment tracking

------------------------------------------------------------------------
