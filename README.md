# GENERATING_NEW_FACES_USING_VARIATIONAL_ENCODERS
This project demonstrates the use of an Autoencoder and Variational Autoencoder (VAE) for image reconstruction using the CelebA dataset. The implementation is done using TensorFlow and Keras.

Table of Contents
Installation
Dataset
Project Structure
Usage
Model Architecture
Training
Results
Contributing
License
Installation
To run this project, you need to have Python 3.x installed along with the necessary libraries. Follow the steps below to set up the environment:

Clone the repository:

bash
Copy code
git clone https://github.com/your-username/celeba-autoencoder.git
cd celeba-autoencoder
Install the required packages:

bash
Copy code
pip install -r requirements.txt
Install Kaggle CLI:

bash
Copy code
pip install kaggle
Setup Kaggle API:

Place your kaggle.json file (containing your Kaggle API credentials) in the ~/.kaggle/ directory.
Dataset
The project uses the CelebA dataset, which can be downloaded directly using the Kaggle API. Ensure you have the necessary permissions to download the dataset.

Download the dataset:

bash
Copy code
kaggle datasets download -d jessicali9530/celeba-dataset
Unzip the dataset:

python
Copy code
from zipfile import ZipFile

with ZipFile('celeba-dataset.zip', 'r') as zipObj:
    zipObj.extractall('./data/')
Project Structure
graphql
Copy code
├── data/
│   ├── img_align_celeba/          # Directory containing the CelebA images
├── weights/                       # Directory for saving model weights
│   ├── AE/                        # Autoencoder weights
│   ├── VAE/                       # Variational Autoencoder weights
├── GenAI_Mini_project.ipynb       # Main Jupyter notebook for the project
├── README.md                      # Readme file
├── requirements.txt               # Python dependencies
└── kaggle.json                    # Kaggle API credentials (ensure this is in the correct directory)
Usage
To run the project, open the GenAI_Mini_project.ipynb notebook in Jupyter and follow the steps below:

Install necessary libraries:

python
Copy code
!pip install -U -q kaggle
Set up directories:

python
Copy code
import os

WEIGHTS_FOLDER = './weights/'
DATA_FOLDER = './data/img_align_celeba/'

if not os.path.exists(WEIGHTS_FOLDER):
    os.makedirs(os.path.join(WEIGHTS_FOLDER, "AE"))
    os.makedirs(os.path.join(WEIGHTS_FOLDER, "VAE"))
Load and preprocess the dataset:

python
Copy code
from glob import glob
import numpy as np

filenames = np.array(glob(os.path.join(DATA_FOLDER, '*/*.jpg')))
NUM_IMAGES = len(filenames)
print("Total number of images : " + str(NUM_IMAGES))

INPUT_DIM = (128, 128, 3)  # Image dimension
BATCH_SIZE = 512
Z_DIM = 200  # Dimension of the latent vector (z)
Create data generator:

python
Copy code
from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_flow = ImageDataGenerator(rescale=1./255).flow_from_directory(
    DATA_FOLDER,
    target_size=INPUT_DIM[:2],
    batch_size=BATCH_SIZE,
    shuffle=True,
    class_mode='input',
    subset='training'
)
Model Architecture
The project includes the implementation of both an Autoencoder and a Variational Autoencoder (VAE). The model architectures are defined in the notebook.

Autoencoder
python
Copy code
def build_autoencoder(input_dim, conv_filters, conv_kernel_size, conv_strides, latent_dim):
    ...
    return model
Variational Autoencoder (VAE)
python
Copy code
def build_vae_encoder(input_dim, output_dim, conv_filters, conv_kernel_size, conv_strides):
    ...
    return model
Training
To train the models, run the training cells in the notebook. The models will be trained using the provided dataset and the weights will be saved in the weights/ directory.

Example:
python
Copy code
from keras.callbacks import ModelCheckpoint

checkpoint_ae = ModelCheckpoint(
    os.path.join(WEIGHTS_FOLDER, 'AE/ae_weights.h5'),
    save_weights_only=True,
    save_best_only=True,
    verbose=1
)

history_ae = ae_model.fit(
    data_flow,
    epochs=100,
    callbacks=[checkpoint_ae]
)
Results
The notebook includes sections for visualizing the results of the trained models, such as reconstructed images and loss plots.

Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes.

License
This project is licensed under the MIT License - see the LICENSE file for details.

