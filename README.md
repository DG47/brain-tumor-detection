# Brain Tumor Detection Using Convolutional Neural Networks (CNN)

## Overview

This project implements a **Convolutional Neural Network (CNN)** to detect brain tumors from MRI images. The model is trained on a dataset of brain MRI scans and can classify images as **tumorous** or **non-tumorous**. The project involves data preprocessing, data augmentation, model building, training, evaluation, and visualization of results.

---

## Table of Contents

- [Project Description](#project-description)
- [Dataset](#dataset)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Project Structure](#project-structure)
- [Data Preprocessing](#data-preprocessing)
  - [Data Augmentation](#data-augmentation)
  - [Image Preprocessing](#image-preprocessing)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Evaluating the Model](#evaluating-the-model)
- [Visualizing Results](#visualizing-results)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Project Description

Brain tumors are life-threatening conditions that require early detection for effective treatment. This project aims to aid in the detection of brain tumors by utilizing deep learning techniques, specifically Convolutional Neural Networks, to classify MRI images.

The model takes MRI images as input and outputs a probability indicating whether a tumor is present. The project includes:

- **Data Augmentation** to increase the dataset size and improve model generalization.
- **Image Preprocessing** to focus on the brain region and normalize the images.
- **CNN Model** designed with convolutional, pooling, and fully connected layers.
- **Model Training and Evaluation** with visualization of training progress.
- **Integration with TensorBoard** for real-time monitoring.

---

## Dataset

The dataset consists of MRI images categorized into two classes:

- **Yes**: Images containing brain tumors.
- **No**: Images without brain tumors.

The original dataset is relatively small, with 253 images. To improve model performance, data augmentation techniques are applied to increase the number of training samples.

**Note**: Due to size constraints, the dataset is not included in this repository. You can obtain a similar dataset from [Kaggle - Brain Tumor MRI Images Dataset](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection).

---

## Getting Started

### Prerequisites

- **Python 3.6 or higher**
- **TensorFlow 2.x**
- **Keras**
- **OpenCV**
- **NumPy**
- **Matplotlib**
- **scikit-learn**

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/brain-tumor-detection-cnn.git
   cd brain-tumor-detection-cnn
   ```

2. **Create a Virtual Environment (Optional)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the Required Packages**

   ```bash
   pip install -r requirements.txt
   ```

   *Ensure that the `requirements.txt` file includes all necessary dependencies.*

---

## Project Structure

```
brain-tumor-detection-cnn/
│
├── data/
│   ├── augmented_data/
│   │   ├── yes/       # Augmented images with tumors
│   │   └── no/        # Augmented images without tumors
│   ├── yes/           # Original images with tumors
│   └── no/            # Original images without tumors
│
├── notebooks/
│   └── brain_tumor_detection.ipynb  # Jupyter notebook with code and explanations
│
├── models/
│   └── brain_tumor_detection_model.h5  # Saved trained model
│
├── logs/
│   └── ...    # TensorBoard logs
│
├── assets/
│   └── ...    # Images for README and documentation
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Data Preprocessing

### Data Augmentation

Due to the limited size of the dataset, data augmentation is applied to increase the number of samples and balance the classes. The augmentation techniques used include:

- **Rotation**: Random rotations between -10 and 10 degrees.
- **Width and Height Shifts**: Random horizontal and vertical translations.
- **Shear Transformation**: Shearing the images along the axis.
- **Brightness Adjustment**: Varying the brightness levels.
- **Horizontal and Vertical Flips**: Flipping images to create mirror images.
- **Zooming**: Random zoom within the image.
- **Fill Mode**: Filling empty areas created by transformations.

The `augment_data()` function in `brain_tumor_detection.ipynb` handles the augmentation process.

### Image Preprocessing

Each image undergoes the following preprocessing steps:

1. **Cropping**: Using the `crop_image()` function to focus on the brain region by detecting the largest contour.
2. **Resizing**: Resizing images to a uniform size of (240, 240, 3).
3. **Normalization**: Scaling pixel values to the range [0, 1].
4. **Labeling**: Assigning labels (1 for tumorous, 0 for non-tumorous).

---

## Model Architecture

The CNN model is built using TensorFlow's Keras API and consists of:

- **Input Layer**: Accepts images of shape (240, 240, 3).
- **Convolutional Layers**: Extract features using filters of varying sizes.
- **Batch Normalization**: Accelerates training and improves performance.
- **Activation Functions**: Uses ReLU for non-linearity.
- **Pooling Layers**: Reduces spatial dimensions to prevent overfitting.
- **Flatten Layer**: Converts 2D matrices to a 1D vector.
- **Dense Layer**: Fully connected layer for classification.
- **Output Layer**: Uses sigmoid activation for binary classification.

---

## Training the Model

The model is compiled and trained with the following configurations:

- **Optimizer**: Adam optimizer with a learning rate of 0.001.
- **Loss Function**: Binary cross-entropy for binary classification.
- **Metrics**: Accuracy to evaluate the model performance.
- **Early Stopping**: Monitors validation loss to prevent overfitting.
- **TensorBoard Callback**: Logs training metrics for visualization.

**Training Command**:

```python
history = model.fit(
    x=X_train,
    y=y_train,
    batch_size=32,
    epochs=20,
    validation_data=(X_valid, y_valid),
    callbacks=[early_stopping, tensorboard_callback]
)
```

---

## Evaluating the Model

The model's performance is evaluated using:

- **Validation Set**: Assess loss and accuracy on unseen data.
- **Confusion Matrix**: Visualize true vs. predicted labels.
- **Classification Report**: Includes precision, recall, F1-score.
- **ROC Curve**: Analyze the trade-off between true positive rate and false positive rate.

---

## Visualizing Results

### Training Metrics

Plotting training and validation loss and accuracy over epochs:

```python
def plot_training_history(history):
    # Code to plot loss and accuracy
    # ...
    plt.show()
```

### TensorBoard

Use TensorBoard for interactive visualizations:

```bash
# In Jupyter Notebook or Colab
%load_ext tensorboard
%tensorboard --logdir logs/
```

---

## Usage

1. **Prepare the Dataset**

   - Place the original images in `data/yes/` and `data/no/` directories.

2. **Run the Notebook**

   - Open `brain_tumor_detection.ipynb` in Jupyter Notebook or Jupyter Lab.
   - Execute the cells sequentially to preprocess data, build, train, and evaluate the model.

3. **Visualize Training**

   - Launch TensorBoard to monitor training progress.
   - Access TensorBoard at `http://localhost:6006/` in your web browser.

4. **Make Predictions**

   - Use the trained model to predict on new MRI images.
   - Load the model:
     ```python
     from tensorflow.keras.models import load_model
     model = load_model('models/brain_tumor_detection_model.h5')
     ```

---

## Results

- **Accuracy**: The model achieves a validation accuracy of over 90% (actual results may vary).
- **Loss**: Training and validation loss decrease over epochs, indicating effective learning.
- **Confusion Matrix**:

  |               | Predicted Positive | Predicted Negative |
  |---------------|--------------------|--------------------|
  | **Actual Positive** | TP                 | FN                 |
  | **Actual Negative** | FP                 | TN                 |

- **ROC Curve**: Shows the model's ability to distinguish between classes.

---

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the Repository**

   ```bash
   git clone https://github.com/DG47/brain-tumor-detection-cnn.git
   ```

2. **Create a Branch**

   ```bash
   git checkout -b feature/your-feature
   ```

3. **Commit Changes**

   ```bash
   git commit -am 'Add new feature'
   ```

4. **Push to the Branch**

   ```bash
   git push origin feature/your-feature
   ```

5. **Open a Pull Request**

   - Submit your pull request for review.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Dataset**: [Kaggle - Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection)
- **Inspiration**: Medical imaging and deep learning applications.
- **References**:
  - [TensorFlow Documentation](https://www.tensorflow.org/)
  - [Keras API Reference](https://keras.io/api/)
  - [OpenCV Documentation](https://docs.opencv.org/)

---

**Disclaimer**: This project is for educational purposes and should not be used as a diagnostic tool without proper validation and regulatory approval.

---

**Contact**:

- **Author**: Dhruv Gupta
- **Email**: [todhruvg@gmail.com](todhruvg@gmail.com)
- **GitHub**: [DG47](https://github.com/DG47)

Feel free to reach out with any questions or suggestions!
