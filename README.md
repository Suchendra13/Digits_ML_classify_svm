# Handwritten Digits Classification Using Support Vector Machines ( SVM )
 This project demonstrates the process of classifying the handwritten digits ( 0-9 ) using a Support Vector Machine Model. It utilizes the `load_digits` dataset from scikit learn, which is a classic dataset for introductory machine learning tasks.

# Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Exploratory Data Analysis](#exploratory-data-analysis)
4. [Methodology](#methodology)
5. [Results](#results)
6. [Visualisation](#visualisation)
7. [How to Run](#how-to-run)
8. [Libraries Used](#libraries-used)

## 1. Project Overview 
The primary goal of this project is to evaluate and implement Support Vector Machine for multi-class classification. We aim to correctly identify handwritten digits from a dataset of 8x8 pixel images. 

## 2. Dataset
This project uses the `load_digits` dataset available in `sklearn.datasets`.
* **Source** : Scikit Learn's `load_digits` dataset.
* **Description** : This dataset consists of 1,797 8x8 pixel grayscale images of handwritten digits ( 0 to 9 ).
* **Features(X)** :  Each image is represented by a 64-dimensional array (8x8 pixels), where each value represents the pixel intensity. Pixel intensity values range from 0 to 16.
* **Target (y)**: The corresponding digit (0-9) that each image represents.

## 3. Exploratory Data Analysis (EDA)

EDA was performed to understand the structure and characteristics of the `load_digits` dataset.

* **Sample Digit Visualization**: Displays a grid of sample 8x8 images from the dataset, along with their true labels, to provide a visual understanding of the data.
* **Class Distribution**: A count plot shows the distribution of each digit (0-9) in the dataset, indicating if the dataset is balanced across classes. The `load_digits` dataset is generally well-balanced.
* **Pixel Intensity Distribution**: A histogram of all pixel intensity values (0-16) across the entire dataset helps understand the common intensity levels.
* **Correlation Heatmap**: A heatmap of pixel intensities indicates the correlation between different pixels. While less directly interpretable for image data, it can reveal patterns of co-occurrence.
* **2D PCA Visualization**: Principal Component Analysis (PCA) was applied to reduce the 64-dimensional data into 2 principal components. A scatter plot of these components, colored by digit class, helps visualize the separability of the different digit clusters in a lower-dimensional space.

## 4. Methodology

The following steps were performed to build and evaluate the Support Vector Machine model:

1.  **Load Libraries**: All necessary libraries including `pandas`, `numpy`, `matplotlib.pyplot`, `seaborn`, `sklearn.datasets`, `sklearn.model_selection`, `sklearn.preprocessing`, `sklearn.decomposition`, `sklearn.svm`, and `sklearn.metrics` were imported.
2.  **Load Dataset**: The `load_digits` dataset was loaded, separating features (`X`) from the target variable (`y`).
3.  **Data Splitting**: The dataset was split into training and testing sets using `train_test_split`, with 80% for training and 20% for testing. No `stratify` parameter was used in the provided notebook, which is a minor deviation from best practice for classification datasets to ensure class balance in splits, but for `load_digits` it often isn't critical due to its inherent balance.
4.  **Data Scaling**: `StandardScaler` was applied to the training and testing feature sets. This is a crucial preprocessing step for Support Vector Machine, as it helps in faster convergence and better performance by standardizing feature scales. Note: The provided notebook applies `fit_transform` to `X_train` and `fit_transform` to `X_test`. For correct scaling, `transform` should be used on `X_test` after `fit_transform` on `X_train` to avoid data leakage. *This discrepancy is noted for potential improvement.*
5.  **Model Application (Logistic Regression)**: A `SVM` model was initialized.The model was then trained using the scaled training data.
6.  **Prediction**: The trained model made predictions (`predict`) on the scaled test set features.
7.  **Model Evaluation**: The performance of the Support Vector Machine model was assessed using:
    * **Accuracy Score**: The proportion of correctly classified instances.
    * **Classification Report**: Provides precision, recall, and f1-score for each class, as well as macro and weighted averages.
    * **Confusion Matrix**: A heatmap visualization showing the counts of true versus predicted labels for each digit, which helps identify specific misclassifications.
  
## 5. Results

The Support Vector Machine model achieved the following performance on the test set:
* **SVM Accuracy** : `0.9806`

The classification report provides a detailed breakdown:
