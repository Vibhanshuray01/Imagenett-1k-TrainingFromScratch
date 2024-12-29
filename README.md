ResNet50 Training on ImageNet-1k Using PyTorch Lightning

This project focuses on training a ResNet50 model from scratch on the ImageNet-1k dataset. The primary goal is to achieve 70% Top-1 Accuracy through an efficient training pipeline implemented using PyTorch Lightning. The project leverages EC2 GPU instances for high-performance computing and uses local storage for the dataset.

Table of Contents:

Project Structure
Features
Requirements
Setup and Installation
Dataset Preparation
Training Instructions
Monitoring
Expected Results
Contributing
License

Project Structure
.
├── src
│   ├── config.py              # Contains all training configurations
│   ├── dataset.py             # Dataset loaders and transformations
│   ├── model.py               # ResNet50 model implementation using PyTorch Lightning
│   ├── train.py               # Training script
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── .env                       # Environment variables (e.g., HF_TOKEN)



Features
ResNet50 Architecture: Implements ResNet50 with PyTorch’s built-in torchvision library.
Training Techniques:
Label Smoothing
MixUp and CutMix Augmentations
Cosine Annealing Learning Rate Scheduler
PyTorch Lightning: Simplifies the training pipeline with modular code.
Multi-GPU Support: Optimized for GPU-based training using Lightning’s accelerator="gpu".
Model Checkpoints: Automatically saves the best model based on validation accuracy.
Local Dataset Support: Efficient loading from mounted EC2 volumes.


Requirements
Hardware:
An AWS EC2 instance with GPU support (e.g., g4dn.2xlarge).
At least 500 GB of disk space for the ImageNet dataset.
Software:
Python 3.8 or later
PyTorch 2.0 or later
NVIDIA CUDA Toolkit (matching PyTorch version)
AWS CLI (for EC2 setup and volume attachment)


Setup and Installation

Clone the Repository:
git clone <your-repo-url>
cd <your-repo-directory>
Install Dependencies: Create a virtual environment and install required packages:


python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

Set Environment Variables: Add your Hugging Face token (if required) in a .env file:
HF_TOKEN=<your-hugging-face-token>


Dataset Preparation
Ensure the ImageNet dataset is properly unzipped and stored on your EC2 instance.


Expected Directory Structure:

/mnt/imagenet/
├── train/
│   ├── n01440764/  # Class folders
│   │   ├── image1.JPEG
│   │   ├── image2.JPEG
│   └── ...
├── val/
│   ├── n01440764/  # Class folders
│   │   ├── image1.JPEG
│   │   ├── image2.JPEG
│   └── ...
To mount the volume with this dataset:



sudo mount /dev/nvme1n1 /mnt/imagenet
Training Instructions
Launch an EC2 GPU Instance:

Use an instance like g4dn.2xlarge with Ubuntu.
Attach the volume containing the dataset.
Run the Training Script:


python train.py
Monitor Training Progress: PyTorch Lightning logs training and validation metrics. Use the CLI or integrate tools like TensorBoard for visualization:


tensorboard --logdir lightning_logs/
Monitoring
During training, the following metrics are tracked:

Top-1 Accuracy (val_acc1)
Top-5 Accuracy (val_acc5)
Validation Loss (val_loss)
Learning Rate (via LearningRateMonitor)
Checkpoints are saved after each epoch based on the best validation accuracy:


resnet50-epoch-{epoch_number}-val_acc1-{val_acc1}.ckpt
Expected Results
With 100 epochs and a full dataset:
Top-1 Accuracy: ~70%
Top-5 Accuracy: ~90%
Training time depends on the batch size and hardware configuration.


Contributing
Contributions are welcome! Please follow these steps:

Fork the repository.
Create a feature branch (git checkout -b feature-name).
Commit your changes (git commit -m 'Add feature').
Push to the branch (git push origin feature-name).
Open a pull request.


License
NA