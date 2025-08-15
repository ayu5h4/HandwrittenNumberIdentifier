
# Handwritten Number Identifier

This repository contains a neural network model that predicts handwritten numbers. The model is built using PyTorch and employs a Convolutional Neural Network (CNN) with a tinyVgg architecture.


## 💿 Dataset

The model is trained on the **MNIST dataset**, which is a large database of handwritten digits that is commonly used for training and testing in the field of machine learning. The dataset is downloaded directly from `torchvision.datasets`.

-----

## 🤖 Model Architecture

The model uses a **tinyVgg architecture**. You can learn more about this architecture and how CNNs work at the [CNN Explainer](https://poloclub.github.io/cnn-explainer/). The model consists of two convolutional blocks followed by a classifier.

-----

## 🛠️ Installation

To run this project, you'll need to have Python and PyTorch installed. You can install the necessary dependencies using pip:

```bash
pip install torch torchvision matplotlib
```

-----

## 🚀 Usage

The primary file in this repository is the `Handwritten_number_Detection.ipynb` Jupyter Notebook.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ayu5h4/handwrittennumberidentifier.git
    cd handwrittennumberidentifier
    ```
2.  **Open the Jupyter Notebook:**
    ```bash
    jupyter notebook Handwritten_number_Detection.ipynb
    ```
3.  **Run the cells:** You can run the cells in the notebook to train the model from scratch and see the predictions on the test set.

You can also use the pre-trained model `handwtitten_number_detector_model.pth`. The notebook includes code to load the saved model and make predictions.

-----

## ✨ Results

The model achieves high accuracy in predicting handwritten digits. I have visualized the results in the notebook:

-----

## 👨‍💻 Author

  * **Ayush Ghodake**
  * **GitHub:** [ayu5h4](https://www.google.com/search?q=https://github.com/ayu5h4)
