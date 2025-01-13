Multi-Scale Transformer Analysis for Areca-Nut Harvestig
This project focuses on building an image classification model for Areca nut harvesting using deep learning. The goal is to classify images of Areca nuts into different categories based on their ripeness level. We use deep learning models such as MultiVision Transformer and Swin Transformer to predict whether the nuts are ripe, overripe, or diseased.

Table of Contents
Project Overview
Dataset
Setup and Installation
Models
MultiVision Transformer (MVT)
Swin Transformer
Training
Prediction with Streamlit
Results
Future Work
License
Project Overview
The main goal of this project is to classify Areca nut images into 5 categories:

Diseased, to be harvested
Overripen, to be harvested
Ripen, to be harvested
Semi-ripen, not to be harvested
Unripen, not to be harvested
We train deep learning models to predict these categories from images, and then deploy the trained models using Streamlit for web-based predictions.

Dataset
The dataset used in this project is stored in the ./dataset/processed directory. The dataset is organized into subfolders, each representing a class label with images of Areca nuts belonging to that category. Each image is labeled with one of the following classes:

Diseased, to be harvested
Overripen, to be harvested
Ripen, to be harvested
Semi-ripen, not to be harvested
Unripen, not to be harvested
The images are resized to 224x224 pixels and are transformed into tensor format using PyTorch's transforms.

Setup and Installation
To get started, clone this repository and install the necessary dependencies:

Clone the repository:
bash
Copy code
git clone https://github.com/your-username/areca-nut-harvesting-classification.git
cd areca-nut-harvesting-classification
Install dependencies:
bash
Copy code
pip install -r requirements.txt
The dependencies include:

torch (PyTorch)
torchvision
timm
optuna
streamlit
PIL (Pillow)
numpy
matplotlib
Models
This project uses two deep learning models: MultiVision Transformer (MVT) and Swin Transformer. Both are trained to classify the Areca nut images.

MultiVision Transformer (MViT)
The MultiVision Transformer (MViT) is a custom model built using a transformer architecture. It consists of the following components:

Patch Embedding Layer: Divides the image into patches and embeds them.
Transformer Blocks: The core of the model, with multi-head self-attention and feed-forward layers.
Classification Head: A fully connected layer to output the class predictions.
Swin Transformer
The Swin Transformer is a pre-trained model from the timm library, fine-tuned for the Areca nut classification task. It works by:

Using a pretrained Swin Transformer as the backbone.
Modifying the head for classification by adding a fully connected layer to predict the class.
Training
To train the models, we use Optuna for hyperparameter optimization and PyTorch for model training. The training pipeline involves the following steps:

Loading and splitting the dataset: The dataset is split into training (80%) and testing (20%) sets.
Data Augmentation: To handle class imbalance, we use WeightedRandomSampler to ensure balanced sampling during training.
Model Training: The model is trained for 10 epochs, with the following hyperparameters optimized using Optuna:
embed_dim, num_heads, depth, feedforward_dim, dropout, learning_rate, momentum
After training, the model is saved as a .pth file.

Hyperparameter Optimization
The model's hyperparameters are optimized using Optuna. The search space includes:

Embed Dimension: 64, 128, 256
Number of Attention Heads: 2, 4, 8
Depth of the Transformer: 4-8 layers
Feedforward Dimension: 128, 256, 512
Dropout: 0.1-0.3
Learning Rate: 1e-4 to 1e-2
Momentum: 0.8 to 0.99
After training, the best hyperparameters are selected for final model evaluation.

Prediction with Streamlit
The trained models are deployed using Streamlit, which provides an interactive interface to upload an image and receive predictions. The app is set up to accept image uploads and display the predicted class (e.g., "Ripen, to be harvested").

Running the Streamlit App
Make sure the model files (mvt.pth and swin_transformer_model.pth) are in the ./output/ folder.
Run the following command to start the Streamlit app:
bash
Copy code
streamlit run app.py
This will open a web interface where you can upload an image and get predictions.

Results
The models are evaluated based on their accuracy on the test set. The final test accuracy for both models is reported after training.

MultiVision Transformer: Achieved an accuracy of X%.
Swin Transformer: Achieved an accuracy of Y%.
The models' performance is saved, and the best model is used for making predictions.

Future Work
Future improvements can include:

Adding more data for training to improve model generalization.
Experimenting with other deep learning architectures like CNN-based models or hybrid models.
Fine-tuning the models further for better performance.
Deploying the models to a cloud service for production use.
License
This project is licensed under the Apache 2.0 License - see the LICENSE file for details.
