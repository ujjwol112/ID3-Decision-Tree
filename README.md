# Decision Tree Classifier for Predictive Modeling

This repository contains Python code for implementing a Decision Tree classifier for predictive modeling tasks. It includes code for both analyzing datasets and making predictions using decision trees. This repository contains a Python script for implementing a decision tree classifier to authenticate banknotes. The dataset used for training and testing the model is provided in the '**BankNote_Authentication.csv**' file.

## Introduction

Decision Trees are a powerful machine learning algorithm used for both classification and regression tasks. They create a model that predicts the value of a target variable by learning simple decision rules inferred from the features. Decision Trees are particularly advantageous for their interpretability, as the resulting model can be visualized as a tree structure, making it easy to understand and interpret the decision-making process.

In this project, Decision Tree classifiers are utilized to analyze datasets and make predictions based on their features. The code encompasses various steps of the machine learning pipeline, including data preprocessing, model training, evaluation, and visualization of the decision tree. By leveraging Decision Trees, insights can be extracted from the data, aiding in understanding the underlying patterns and relationships.

## Dataset

### Banknote Authentication Dataset

The Banknote Authentication dataset contains features extracted from photographic images of genuine and forged banknotes. The Banknote Authentication dataset contains features extracted from photographic images of genuine and forged banknotes. The features include:

- Variance of Wavelet Transformed Image
- Skewness of Wavelet Transformed Image
- Kurtosis of Wavelet Transformed Image
- Entropy of Image
- Class: Whether the banknote is genuine or forged (0 = Genuine, 1 = Forged)

This dataset aims to predict whether a banknote is genuine or forged based on its features. Insights derived from the Banknote Authentication dataset analysis include:

- **Feature Importance**: Identifying the most important features for predicting banknote authenticity and understanding their contribution to the classification task.
- **Decision Tree Visualization**: Visualizing the decision tree constructed by the classifier to understand the decision-making process and identify key decision rules.
- **Model Evaluation**: Evaluating the performance of the Decision Tree classifier using metrics such as accuracy, precision, recall, and F1-score.

## Code Overview

The repository includes the following Python code:

- **Decision Tree Classifier**: This code implements a Decision Tree classifier using the scikit-learn library. It encompasses steps for data preprocessing, model training, evaluation, and visualization of the decision tree.

**Description**:

- Importing Libraries:
  - pandas: For data manipulation and analysis.
  - pylab, matplotlib.pyplot: For plotting graphs and visualizations.
  - seaborn: For enhancing the visualizations.
  - sklearn: For machine learning algorithms and evaluation metrics.
- Loading Data:
  - The script loads the banknote authentication dataset using pandas and separates the features (X) and the target variable (Y).
- Splitting Data:
  - The dataset is split into training and test sets using train_test_split from sklearn.
- Decision Tree Model:
  - A decision tree classifier is instantiated with the criterion set to 'entropy'.
- Model Training:
  - The decision tree model is trained on the training dataset.
- Model Evaluation:
  - The trained model is used to make predictions on the test dataset.
  - Confusion matrix and classification report are computed using confusion_matrix and classification_report from sklearn.
- Visualization:
  - Confusion matrix is visualized as a heatmap using seaborn.
  - Decision tree is visualized using plot_tree from sklearn.

## Dependencies

To run the code, you need the following dependencies:

- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

You can install these dependencies using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage

1. Clone this repository:

```bash
git clone https://github.com/ujjwol112/decision-tree-classifier.git
```

2. Navigate to the repository directory:

```bash
cd decision-tree-classifier
```

3. Ensure that the dataset file `banknote_authentication.csv` is placed in the same directory.


## Results

The code produces the following result:

- **Decision Tree Classifier**: The code trains a Decision Tree classifier on the dataset and evaluates its performance using metrics such as accuracy, precision, recall, and F1-score. The decision tree visualization provides insights into the decision-making process of the classifier.
- Confusion Matrix: Provides insight into the performance of the model in terms of true positives, false positives, true negatives, and false negatives.
- Classification Report: Displays precision, recall, F1-score, and support for each class.
- Decision Tree Visualization: Offers a graphical representation of the decision tree model.
