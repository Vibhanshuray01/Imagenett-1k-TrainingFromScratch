# ResNet50 Training on ImageNet-1k Using PyTorch Lightning

This project focuses on training a ResNet50 model from scratch on the ImageNet-1k dataset. The primary goal is to achieve 70% Top-1 Accuracy through an efficient training pipeline implemented using PyTorch Lightning. The project leverages EC2 GPU instances for high-performance computing and uses local storage for the dataset.

---

## Table of Contents:
1. Features
2. Requirements
3. Setup and Installation
4. Dataset Preparation
5. Training Instructions
6. Monitoring
7. Expected Results
8. Contributing
9. License

---

## Features:
- **ResNet50 Architecture:** Built using PyTorch's `torchvision` library for robustness and scalability.
- **Training Techniques:**
  - Label Smoothing
  - MixUp and CutMix Augmentations
  - Cosine Annealing Learning Rate Scheduler
- **PyTorch Lightning:** Simplifies the training process with a modular and scalable approach.
- **GPU Optimization:** Configured for single or multi-GPU training (e.g., EC2 g4dn.2xlarge instances).
- **Model Checkpointing:** Automatically saves the best model based on validation accuracy.
- **Efficient Dataset Loading:** Supports direct loading from mounted EC2 volumes for faster training.

---

## Requirements:
### Hardware:
- AWS EC2 GPU instance (e.g., g4dn.2xlarge).
- 500 GB of storage for the ImageNet dataset.

### Software:
- Python 3.8 or later
- PyTorch 2.0 or later
- NVIDIA CUDA Toolkit (matching PyTorch version)
- AWS CLI for EC2 setup

---

## Setup and Installation:
1. **Clone the Repository:**

   git clone <your-repo-url>
   cd <your-repo-directory>


Install Dependencies:

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt


Dataset Preparation:
Ensure the ImageNet dataset is properly extracted and stored on your EC2 volume.

Mount the dataset volume:
sudo mkdir -p /mnt/imagenet
sudo mount /dev/nvme1n1 /mnt/imagenet


Training Instructions:
Launch an EC2 GPU Instance:

Use an instance like g4dn.2xlarge with Ubuntu.
Attach the volume containing the dataset.
Run the Training Script:
python train.py


Here’s the updated and simplified README.md content that avoids the formatting issues with the project structure while maintaining clarity and relevance:

markdown
Copy code
# ResNet50 Training on ImageNet-1k Using PyTorch Lightning

This project focuses on training a ResNet50 model from scratch on the ImageNet-1k dataset. The primary goal is to achieve 70% Top-1 Accuracy through an efficient training pipeline implemented using PyTorch Lightning. The project leverages EC2 GPU instances for high-performance computing and uses local storage for the dataset.

---

## Table of Contents:
1. Features
2. Requirements
3. Setup and Installation
4. Dataset Preparation
5. Training Instructions
6. Monitoring
7. Expected Results
8. Contributing
9. License

---

## Features:
- **ResNet50 Architecture:** Built using PyTorch's `torchvision` library for robustness and scalability.
- **Training Techniques:**
  - Label Smoothing
  - MixUp and CutMix Augmentations
  - Cosine Annealing Learning Rate Scheduler
- **PyTorch Lightning:** Simplifies the training process with a modular and scalable approach.
- **GPU Optimization:** Configured for single or multi-GPU training (e.g., EC2 g4dn.2xlarge instances).
- **Model Checkpointing:** Automatically saves the best model based on validation accuracy.
- **Efficient Dataset Loading:** Supports direct loading from mounted EC2 volumes for faster training.

---

## Requirements:
### Hardware:
- AWS EC2 GPU instance (e.g., g4dn.2xlarge).
- 500 GB of storage for the ImageNet dataset.

### Software:
- Python 3.8 or later
- PyTorch 2.0 or later
- NVIDIA CUDA Toolkit (matching PyTorch version)
- AWS CLI for EC2 setup

---

## Setup and Installation:
1. **Clone the Repository:**
   ```bash
   git clone <your-repo-url>
   cd <your-repo-directory>
Install Dependencies:

bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
Set Environment Variables: (if required, for Hugging Face or other tokens)

Create a .env file and add:
bash
Copy code
HF_TOKEN=<your-hugging-face-token>
Dataset Preparation:
Ensure the ImageNet dataset is properly extracted and stored on your EC2 volume.
Directory structure:
bash
Copy code
/mnt/imagenet/
├── train/
│   ├── <class_name>/image1.JPEG
│   └── ...
├── val/
    ├── <class_name>/image1.JPEG
    └── ...
Mount the dataset volume:
bash
Copy code
sudo mkdir -p /mnt/imagenet
sudo mount /dev/nvme1n1 /mnt/imagenet
Training Instructions:
Launch an EC2 GPU Instance:

Use an instance like g4dn.2xlarge with Ubuntu.
Attach the volume containing the dataset.
Run the Training Script:

bash
Copy code
python train.py
Monitor Training Progress:

Use tools like TensorBoard for visualization:
bash
Copy code
tensorboard --logdir lightning_logs/
Monitoring:
During training, the following metrics are logged:

Top-1 Accuracy (val_acc1)
Top-5 Accuracy (val_acc5)
Validation Loss (val_loss)
Learning Rate (via LearningRateMonitor)
Checkpoints are saved automatically:

Example: resnet50-epoch-{epoch_number}-val_acc1-{val_acc1}.ckpt
Expected Results:
With 100 epochs and the full ImageNet dataset:

Top-1 Accuracy: ~70%
Top-5 Accuracy: ~90%
Training time depends on your hardware and batch size.


Contributing:
Gokkul Nath - gokkulnath@gmail.com

License:
NA