# Solar Explorar - Roof Segmentation

A U-Net based deep learning project to segment roofs from satellite imagery.

## Features
- Custom U-Net architecture for binary segmentation.
- Data augmentation using Albumentations.
- Automated mask generation from JSON annotations.
- Inference script with dynamic thresholding and heatmap visualization.

## Setup
1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd solar_explorar
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
- **Training**: Run `python train_unet.py` to train the model.
- **Validation/Inference**: Run `python lets_validize.py` to see results on sample images.
- **Mask Generation**: Run `python create_masks.py` if you have new annotated data in `raw_images`.

## Results
The current model achieves significant confidence (up to 0.91) after 100 epochs on a small dataset.
