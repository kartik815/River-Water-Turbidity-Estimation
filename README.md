# River Water Turbidity Estimation Using Digital Image Processing and Deep Learning

> **Project type:** Digital Image Processing / Image Classification Case Study  
> **Platform:** Google Colab  
> **Primary framework:** PyTorch + Torchvision  
> **Deep-learning model:** EfficientNet-B0 (ImageNet-pretrained, fine-tuned for five turbidity classes)  
> **Additional segmentation model:** SegFormer-B0 (pretrained on ADE20K, used for exploratory river-water extraction)

---

## 1. Project Overview

This project investigates whether visual characteristics of water images can be used to classify river-water turbidity into five categories. It combines conventional Digital Image Processing (DIP) operations, handcrafted feature extraction, and deep learning.

The workflow has two main parts:

1. **Image analysis and model development:** Explore the dataset using image enhancement, restoration, colour analysis, wavelet transforms, compression, morphology and segmentation. Extract handcrafted features and evaluate a traditional machine-learning baseline. Then train and evaluate an EfficientNet-B0 image classifier.
2. **Exploratory real-river image experiment:** Apply a pretrained SegFormer semantic-segmentation model to a natural river photograph, isolate the predicted water region, and compare EfficientNet predictions on the full image, a cropped water region, and a background-masked water crop.

The classifier predicts a **turbidity category**. It does not directly measure NTU, and its output probabilities should not be interpreted as calibrated confidence or laboratory measurements.

## 2. Objectives

- Study how image-processing techniques reveal or modify visual information in water samples.
- Examine colour, intensity, texture, edges and multi-scale image details that may be associated with turbidity.
- Build a conventional feature-based classification baseline.
- Train a transfer-learning image classifier using EfficientNet-B0.
- Evaluate the classifier on a held-out test split designed to reduce leakage between related source images.
- Explore whether semantic segmentation can isolate river water from surrounding objects before classification.
- Identify the limitations of laboratory-image classification when applied to natural river photographs.

## 3. Dataset

The project dataset contains **11,577 images** arranged into five turbidity classes and **34 source groups**.

### 3.1 Class labels

| Folder / class | Turbidity category | NTU range |
|---|---|---:|
| Class 1 | Excellent | 0–7 NTU |
| Class 2 | Good | 10–25 NTU |
| Class 3 | Acceptable | 30–55 NTU |
| Class 4 | Contaminated | 60–120 NTU |
| Class 5 | Highly contaminated | 140–180 NTU |

The provided label ranges contain gaps. Therefore, the five classes should be treated as the dataset's predefined categories; the project does not infer values for the NTU intervals between them.

### 3.2 Dataset organisation

The notebook expects a directory containing one folder per class. The class-folder names must match the `PROJECT_CLASSES` variable used in the notebook.

```text
Dataset/
├── Class 1/
├── Class 2/
├── Class 3/
├── Class 4/
└── Class 5/
```

The notebook also associates images with `source_group` identifiers. These identifiers are important for the group-based train/validation/test split.

### 3.3 Train, validation and test split

The dataset contains 34 source groups. The notebook uses a group-based split, allocating **24 source groups to training, 5 to validation and 5 to testing** (one validation and one test group per class, with the remaining groups assigned to training).

Images belonging to the same source group are kept together rather than being randomly distributed across splits. This reduces the risk that highly similar images from one original source appear in both training and evaluation data.

> **Reproducibility note:** The exact number of images in each split depends on how many images belong to each source group. Record the final `len(train_dataset)`, `len(val_dataset)` and `len(test_dataset)` values printed by the notebook here before publishing the repository. Do not infer image counts from the number of groups.

## 4. Digital Image Processing

DIP operations were used to understand the image data and investigate possible visual cues for turbidity. Most of these operations were exploratory demonstrations; they were not all applied as preprocessing to the final EfficientNet training images.

| Technique | Purpose in this project |
|---|---|
| Brightness and contrast adjustment | Examine how intensity and contrast changes affect the appearance of water samples. |
| Histogram equalization | Redistribute grayscale intensity values to increase global contrast. |
| CLAHE | Enhance local contrast while limiting excessive amplification of noise. |
| Gaussian noise and Gaussian filtering | Demonstrate smoothing and the effect of reducing Gaussian noise. |
| Salt-and-pepper noise and median filtering | Demonstrate removal of isolated extreme-value pixels while retaining edges relatively well. |
| RGB, HSV and grayscale analysis | Inspect colour channels, hue, saturation, value and intensity statistics. |
| Wavelet decomposition | Separate coarse image information from horizontal, vertical and diagonal details at multiple scales. |
| JPEG compression | Examine the visual effect of lossy compression at different quality settings. |
| Morphological operations | Demonstrate erosion, dilation, opening and closing on thresholded masks. |
| Image segmentation | Compare global thresholding, adaptive thresholding and K-means pixel clustering. |
| Handcrafted feature extraction | Calculate colour statistics, GLCM texture features, Local Binary Pattern (LBP) histograms, Canny edge density and wavelet-energy features. |

### Why DIP was useful

These operations helped us inspect the characteristics of the dataset beyond simply viewing the original images. For example, histograms describe intensity distributions, HSV separates colour information from brightness, GLCM and LBP describe local texture, and wavelets capture image details at different scales.

The experiments also showed that an image-processing operation can introduce artefacts or amplify noise. Consequently, an enhanced image is not automatically a better input for classification. The original training images and their preprocessing pipeline must remain consistent when evaluating the final model.

## 5. Models and Methodology

### 5.1 Handcrafted features and conventional baseline

The feature-extraction stage converts each image into a numerical feature vector. The extracted information includes:

- RGB, HSV and grayscale statistics;
- GLCM texture measures such as contrast, dissimilarity, homogeneity, energy and correlation;
- LBP texture-pattern histograms;
- Canny edge density;
- Wavelet approximation and detail energies.

These features were used with a conventional Random Forest baseline. In the reported experiment, the expanded handcrafted-feature baseline achieved **51.58% test accuracy** and **40.42% macro F1-score**.

This provides a comparison for the deep-learning experiment on this dataset and split. It does not establish that deep learning is always better than conventional methods; performance depends on the data, features, split and model configuration.

### 5.2 EfficientNet-B0 classifier

The main classifier is **EfficientNet-B0**, loaded with ImageNet-pretrained weights and fine-tuned for the project's five classes.

#### Architecture overview

EfficientNet-B0 is a convolutional neural network (CNN). Its feature extractor uses mobile inverted bottleneck convolution blocks (MBConv), depthwise convolutions and squeeze-and-excitation (SE) channel attention. The network progressively extracts low-level and high-level visual features.

The standard EfficientNet-B0 feature-extractor stages are summarised below:

| Part | Main operation | Output channels | Repeats |
|---|---|---:|---:|
| Stem | 3×3 convolution, stride 2 | 32 | 1 |
| Stage 1 | MBConv1, 3×3 | 16 | 1 |
| Stage 2 | MBConv6, 3×3 | 24 | 2 |
| Stage 3 | MBConv6, 5×5 | 40 | 2 |
| Stage 4 | MBConv6, 3×3 | 80 | 3 |
| Stage 5 | MBConv6, 5×5 | 112 | 3 |
| Stage 6 | MBConv6, 5×5 | 192 | 4 |
| Stage 7 | MBConv6, 3×3 | 320 | 1 |
| Head | 1×1 convolution | 1280 | 1 |
| Classifier | Global average pooling, dropout and fully connected layer | 5 outputs | 1 |

The MBConv expansion factor is indicated by the number after `MBConv`. For example, MBConv6 expands the input channels by a factor of six within the block. The exact internal operations and tensor dimensions are managed by Torchvision's EfficientNet-B0 implementation.

#### Project-specific classifier modification

The original ImageNet classifier predicts 1,000 categories. We replace its final fully connected layer with a layer containing **five output units**, one for each turbidity class:

```python
model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
model.classifier[1] = torch.nn.Linear(
    model.classifier[1].in_features, NUM_CLASSES
)
```

All model parameters are set to be trainable in the notebook, so the experiment fine-tunes the feature extractor as well as the new classifier layer.

#### Training configuration

| Parameter | Value |
|---|---:|
| Model | EfficientNet-B0 |
| Initial weights | ImageNet pretrained |
| Number of classes | 5 |
| Input image size | 244 × 244 (as configured in the notebook) |
| Batch size | 50 |
| Epochs | 30 |
| Optimizer | Adam |
| Learning rate | 0.0001 |
| Loss function | Cross-Entropy Loss |
| Compute device | CUDA GPU when available; otherwise CPU |
| Mixed precision | Enabled when using CUDA |

Images are converted to RGB, resized, converted to tensors and normalized using the ImageNet mean and standard deviation. The same `image_transform` is used for validation, testing and single-image inference.

The notebook records training and validation loss and accuracy after every epoch. It saves the best model according to validation loss and also saves a resumable checkpoint containing the model, optimizer, scaler and training history.

### 5.3 SegFormer water-region extraction experiment

A further idea was to remove surrounding objects from a natural river photograph and focus the classifier on the river water itself. Examples of surrounding content include vegetation, rocks, banks, shadows and other background regions.

For this exploratory experiment, we used **SegFormer-B0 fine-tuned on ADE20K**:

```text
nvidia/segformer-b0-finetuned-ade-512-512
```

SegFormer is a transformer-based semantic-segmentation model. Unlike an image classifier that assigns one label to an entire image, a semantic-segmentation model predicts a class for each pixel.

The notebook uses the model's label map to locate the `water` class, then:

1. Predicts segmentation logits for the uploaded river photograph.
2. Resizes the logits to the original working image dimensions.
3. Selects the water-class pixels to create a binary mask.
4. Applies morphological closing to fill small gaps.
5. Uses connected-component analysis to retain the largest detected water region.
6. Extracts a bounding crop and creates a second crop with non-water pixels set to black.
7. Runs the original EfficientNet-B0 classifier on the full image, water crop and masked water crop.

This experiment tests whether removing background content changes the classifier's prediction. It is exploratory: segmentation errors, reflections, shadows, vegetation and the difference between natural river photographs and laboratory training samples can all affect the result.

## 6. Results

### 6.1 EfficientNet-B0 held-out test results

The reported test results for the selected EfficientNet-B0 checkpoint are:

| Metric | Result |
|---|---:|
| Test accuracy | 87.74% |
| Macro F1-score | 87.67% |
| Macro ROC-AUC | 0.9832 |
| Test images | 1,745 |

The confusion matrix shows that the main classification errors occur between neighbouring categories:

- Class 1: 227 of 314 images correctly classified (approximately 72%).
- Class 2: 235 of 362 images correctly classified (approximately 65%).
- Class 3: 381 of 381 images correctly classified in this test set.
- Class 4: 346 of 346 images correctly classified in this test set.
- Class 5: 342 of 342 images correctly classified in this test set.

The ROC-AUC values shown in the notebook's plot are approximately 1.000, 0.934, 0.982, 1.000 and 1.000 for Classes 1–5 respectively. The macro ROC-AUC is approximately 0.9832.

These are results on the project's held-out test split. Perfect classification for some classes in this particular test set should not be interpreted as proof of perfect performance on all future water images.

### 6.2 Training behaviour and overfitting

The learning curves show training accuracy approaching 100%, while validation accuracy fluctuates and validation loss increases. This indicates substantial overfitting: the model fits the training images very well but does not show similarly stable improvement on validation data.

The best checkpoint is selected by validation loss, not simply by the final training epoch. The selected checkpoint should be used for the reported held-out test evaluation.

### 6.3 Comparison with the handcrafted baseline

| Approach | Test accuracy | Macro F1 |
|---|---:|---:|
| Handcrafted features + Random Forest | 51.58% | 40.42% |
| EfficientNet-B0 | 87.74% | 87.67% |

EfficientNet-B0 performed substantially better than the handcrafted Random Forest baseline in this experiment. A likely reason is that the CNN learns hierarchical visual representations directly from image data rather than relying only on a fixed set of manually designed features.

This is a comparison under the project's experimental setup, not a universal performance guarantee.

## 7. Limitations

- **Dataset domain:** The classifier was trained on the supplied dataset, which primarily contains laboratory-style water samples. Natural river photographs can have different lighting, reflections, camera characteristics, backgrounds and water appearances.
- **Overfitting:** The training and validation curves indicate a substantial generalization gap.
- **Group coverage:** Group-based splitting reduces source leakage, but the test set contains only a limited number of source groups. Results may vary with a different group split.
- **Class confusion:** Classes 1 and 2, and Classes 2 and 3, are confused in the held-out test results.
- **No direct NTU measurement:** The model predicts one of five categories. It does not replace a calibrated turbidity meter or laboratory measurement.
- **Segmentation errors:** SegFormer can incorrectly label shadows, reflections, vegetation or other regions as water, or miss parts of the river.
- **Exploratory real-world test:** A prediction on a natural river photograph is not a validated real-world result unless it is compared with a measured NTU value.
- **Image-processing side effects:** Enhancement, thresholding and filtering can remove or exaggerate details. The DIP demonstrations should not be assumed to improve classifier performance without a controlled experiment.

## 8. Future Improvements

1. Collect a larger and more diverse dataset of natural river images, with turbidity measurements recorded at the time and location of image capture.
2. Use more independent source groups for validation and testing, and repeat evaluation across multiple group splits.
3. Address overfitting with data augmentation, weight decay, dropout, learning-rate scheduling, early stopping and selective layer freezing.
4. Investigate class balancing and class-aware sampling if the dataset distribution requires it.
5. Calibrate predicted probabilities and report per-class precision, recall, F1-score and uncertainty.
6. Compare EfficientNet-B0 with other CNN architectures and carefully tuned conventional baselines.
7. Improve water segmentation and evaluate it against manually annotated river masks.
8. Test the complete segmentation-plus-classification pipeline on a labelled collection of real river images.
9. If continuous turbidity estimation is required, collect numeric NTU labels and investigate regression or ordinal-classification approaches.

## 9. Notebook Structure

| Notebook section | Purpose |
|---|---|
| Dataset loading and inspection | Read image paths, class labels and source-group information. |
| Turbidity knowledge base and validation | Define the five classes and validate labels and source groups. |
| Image enhancement | Brightness/contrast, histogram equalization and CLAHE. |
| Image restoration | Synthetic Gaussian and salt-and-pepper noise, Gaussian and median filtering, PSNR. |
| Colour-space analysis | RGB, HSV, grayscale channels, histograms and statistics. |
| Wavelet analysis | Multi-level wavelet decomposition and coefficient energy. |
| Compression | JPEG quality comparison and PSNR. |
| Morphology | Otsu thresholding, erosion, dilation, opening and closing. |
| Segmentation | Otsu, adaptive thresholding and K-means. |
| Handcrafted feature extraction | Colour, GLCM, LBP, Canny and wavelet features. |
| Group-based data splitting | Create training, validation and test sets without splitting source groups. |
| EfficientNet-B0 setup | Dataset class, transforms, DataLoaders, pretrained model and optimizer. |
| Training and validation | Fine-tune the model, track metrics and save checkpoints. |
| Model evaluation | Learning curves, test metrics, confusion matrices and ROC-AUC. |
| Single-image prediction | Upload an image and predict its turbidity category. |
| River segmentation experiment | Use SegFormer to isolate water and compare classifier inputs. |

## 10. Running the Project

### 10.1 Platform

The experiments were developed in **Google Colab**, which provides a hosted Jupyter notebook environment and optional GPU acceleration.

**Notebook:** [Open the Google Colab notebook](https://colab.research.google.com/drive/1ypbA0CSGXHj-cXUlOm6orqV2SxZ2YDNF)

Access to the notebook and dataset depends on Google Drive sharing permissions. Make a copy of the notebook before editing or running it if you do not have edit access.

### 10.2 Dataset setup

1. Place the dataset in Google Drive.
2. Ensure the five class folders and their names match the notebook configuration.
3. Update the `SOURCE_ROOT` path if your dataset is stored elsewhere.
4. Run the dataset inspection and validation cells before training.
5. Check the printed image counts and source-group split.

The notebook uses the following example paths:

```python
SOURCE_ROOT = Path("/content/drive/MyDrive/Dataset")
LOCAL_ROOT = Path("/content/Dataset")
```

It copies the dataset to Colab's local storage to improve image-loading performance.

### 10.3 Dependencies

The notebook uses Python libraries including:

- `torch` and `torchvision` for deep learning and EfficientNet-B0;
- `numpy` and `pandas` for numerical operations and data handling;
- `opencv-python` (`cv2`) and `Pillow` for image processing and loading;
- `matplotlib` for plots;
- `scikit-image` for GLCM and LBP feature extraction;
- `PyWavelets` (`pywt`) for wavelet decomposition;
- `scikit-learn` for data splitting, Random Forest and evaluation metrics;
- `tqdm` for progress bars;
- `transformers` for SegFormer.

The notebook installs or imports the required packages in its setup cells. Run the notebook from top to bottom so that variables, transforms, datasets, model definitions and checkpoints are created in the correct order.

### 10.4 Training outputs

The training code saves model artefacts to:

```text
/content/drive/MyDrive/Turbidity_Model/
```

Important files include:

```text
Turbidity_Model/
├── efficientnet_b0_grouped_best.pth
├── efficientnet_b0_grouped_last.pth
├── efficientnet_b0_training_history.csv
└── efficientnet_b0_test_predictions.csv
```

- `efficientnet_b0_grouped_best.pth`: model checkpoint with the best validation loss.
- `efficientnet_b0_grouped_last.pth`: resumable checkpoint containing model, optimizer, scaler and history state.
- `efficientnet_b0_training_history.csv`: epoch-wise training and validation metrics.
- `efficientnet_b0_test_predictions.csv`: true labels, predicted labels, correctness and class probabilities for test images.

The notebook's checkpoint paths and dataset paths may need to be changed if your Google Drive folder structure differs.

## 11. Repository / Report Figures

Suggested figure assets for a project repository or report:

```text
figures/
├── dataset_DIP_montage.png
├── efficientnet_architecture.png
├── model_evaluation.png
└── river_segmentation_example.png
```

Recommended contents:

- `dataset_DIP_montage.png`: representative images from all five classes and selected DIP outputs (CLAHE, denoising and wavelet details).
- `efficientnet_architecture.png`: EfficientNet-B0 architecture diagram, with an appropriate citation if reproduced from another source.
- `model_evaluation.png`: training/validation curves, confusion matrices and ROC curves.
- `river_segmentation_example.png`: original river image, refined water mask and extracted/masked river crop.

Do not include the individual prediction chart as evidence of generalization; it is best presented only as an illustration of the inference workflow.

## 12. Conclusion

The project combines conventional digital image processing with feature-based machine learning and transfer learning to investigate image-based turbidity classification. DIP techniques helped analyse image intensity, colour and texture, while EfficientNet-B0 achieved stronger held-out test performance than the handcrafted Random Forest baseline in the reported experiment. The SegFormer experiment further explored isolating river water from surrounding objects before classification. However, overfitting and the difference between laboratory samples and natural river scenes remain important limitations. Validation using diverse real-river images paired with measured NTU values is the key next step toward a reliable practical system.

## 13. References and Bibliography

1. **An Image-Based Water Turbidity Classification Scheme Using a Convolutional Neural Network.** ResearchGate-hosted paper:  
   https://www.researchgate.net/publication/393953064_An_Image-Based_Water_Turbidity_Classification_Scheme_Using_a_Convolutional_Neural_Network/link/68af62917984e374acebb0f5/download

2. Tan, M. and Le, Q. V. **EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.** Proceedings of the 36th International Conference on Machine Learning (ICML), 2019.  
   https://arxiv.org/abs/1905.11946

3. Xie, E. et al. **SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers.** Advances in Neural Information Processing Systems (NeurIPS), 2021.  
   https://arxiv.org/abs/2105.15203

4. PyTorch documentation. **Torchvision Models — EfficientNet.**  
   https://pytorch.org/vision/stable/models/efficientnet.html

5. Hugging Face documentation. **SegFormer.**  
   https://huggingface.co/docs/transformers/model_doc/segformer

6. OpenCV documentation. **Image Processing (OpenCV-Python Tutorials).**  
   https://docs.opencv.org/

---

## 14. Reproducibility and Reporting Notes

Before submitting or publishing this project:

- Confirm that the Google Colab notebook link is accessible to the intended readers.
- Record the exact image counts in the training, validation and test datasets from the final notebook run.
- Keep the reported test metrics tied to the exact saved checkpoint and split that produced them.
- If the notebook is rerun with a different random seed, split, dataset version or checkpoint, update the reported metrics.
- Preserve the distinction between the held-out laboratory/test-set evaluation and the exploratory natural-river photograph experiment.
- Cite the source of any architecture figure or dataset image that was not created by the project team.
