# Explainable AI (XAI) Techniques for MNIST Classification

This repository contains a collection of algorithms and techniques aimed at improving the interpretability and understanding of machine learning models, particularly focusing on the task of classifying handwritten digits from the MNIST dataset. The primary goal is to shed light on how these models make predictions, with a special emphasis on Activation Maximization (AM) techniques.

## Overview

Explainable AI (XAI) is becoming increasingly important as machine learning models are integrated into more critical applications. Understanding the reasoning behind model predictions not only builds trust but also provides insights that can help improve model design. This project focuses on the MNIST digit classification task, leveraging Activation Maximization to visualize what excites certain neurons most, thereby giving us a glimpse into the model's learned features.

### Features

- **Activation Maximization (AM)**: Implementation of AM to understand and visualize the features that activate specific neurons in the network, enhancing interpretability.
    - **Simple AM with L2 Regularization**: Utilizes gradient ascent to optimize input images that maximize neuron activation, with L2 regularization to maintain the visual quality of generated images.
    - **AM via Deep Generative Networks (DGNs)**: Explores the use of DGNs to generate images that maximize neuron activation, optimizing in the latent space of a pretrained generator for more controlled and interpretable outputs.

## Getting Started

### Prerequisites

- Python 3.x
- PyTorch
- Matplotlib
- Jupyter Notebook (Optional, for running .ipynb examples)

## Acknowledgments

- [MNIST dataset](http://yann.lecun.com/exdb/mnist/): Handwritten digit dataset used for classification tasks.
- Community contributions and insights in the field of Explainable AI.

## References

This project has been informed and inspired by a wide range of sources in the field of Explainable AI. One key resource has been:

- Samek, W., Montavon, G., Vedaldi, A., Hansen, L. K., & Müller, K. R. (2019b). Explainable AI: Interpreting, explaining and visualizing deep learning. In Lecture Notes in Computer Science. https://doi.org/10.1007/978-3-030-28954-6

Specific papers and resources that have influenced this work include:
- Nguyen, A., Yosinski, J., & Clune, J. (2019). Understanding Neural Networks via Feature Visualization: A survey. arXiv (Cornell University). https://arxiv.org/pdf/1904.08939.pdf

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
