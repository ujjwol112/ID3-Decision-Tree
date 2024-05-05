# Decision Tree Classifier for Predictive Modeling

This repository contains Python code for implementing a Decision Tree classifier for predictive modeling tasks. It includes code for both analyzing datasets and making predictions using decision trees.

## Introduction

Decision Trees are a powerful machine learning algorithm used for both classification and regression tasks. They create a model that predicts the value of a target variable by learning simple decision rules inferred from the features. Decision Trees are particularly advantageous for their interpretability, as the resulting model can be visualized as a tree structure, making it easy to understand and interpret the decision-making process.

In this project, Decision Tree classifiers are utilized to analyze datasets and make predictions based on their features. The code encompasses various steps of the machine learning pipeline, including data preprocessing, model training, evaluation, and visualization of the decision tree. By leveraging Decision Trees, insights can be extracted from the data, aiding in understanding the underlying patterns and relationships.

## Datasets

### Titanic Dataset

The Titanic dataset contains information about passengers onboard the Titanic, including whether they survived or not. The Titanic dataset contains information about passengers onboard the Titanic, including whether they survived or not. The features include:

PassengerId: Unique identifier for each passenger
Survived: Whether the passenger survived or not (0 = No, 1 = Yes)
Pclass: Ticket class (1st, 2nd, or 3rd)
Name: Passenger's name
Sex: Passenger's gender
Age: Passenger's age
SibSp: Number of siblings/spouses aboard
Parch: Number of parents/children aboard
Ticket: Ticket number
Fare: Passenger's fare
Cabin: Cabin number
Embarked: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

This dataset provides a unique opportunity to analyze the factors influencing survival rates on the Titanic. Insights extracted from the Titanic dataset analysis include:

- **Survival Rate**: Analyzing the survival rate of passengers and identifying factors such as gender, age, and passenger class that influenced survival.
- **Gender Disparity**: Investigating the gender disparity in survival rates and understanding the role of gender in determining survival probability.
- **Passenger Class**: Exploring the impact of passenger class on survival and comparing survival rates across different classes.

### Banknote Authentication Dataset

The Banknote Authentication dataset contains features extracted from photographic images of genuine and forged banknotes. The Banknote Authentication dataset contains features extracted from photographic images of genuine and forged banknotes. The features include:

Variance of Wavelet Transformed Image
Skewness of Wavelet Transformed Image
Kurtosis of Wavelet Transformed Image
Entropy of Image
Class: Whether the banknote is genuine or forged (0 = Genuine, 1 = Forged)

This dataset aims to predict whether a banknote is genuine or forged based on its features. Insights derived from the Banknote Authentication dataset analysis include:

- **Feature Importance**: Identifying the most important features for predicting banknote authenticity and understanding their contribution to the classification task.
- **Decision Tree Visualization**: Visualizing the decision tree constructed by the classifier to understand the decision-making process and identify key decision rules.
- **Model Evaluation**: Evaluating the performance of the Decision Tree classifier using metrics such as accuracy, precision, recall, and F1-score.

## Code Overview

The repository includes the following Python code:

1. **Decision Tree Classifier**: This code implements a Decision Tree classifier using the scikit-learn library. It encompasses steps for data preprocessing, model training, evaluation, and visualization of the decision tree.

2. **Exploratory Data Analysis (EDA)**: This code performs exploratory data analysis on the dataset, including data cleaning, visualization of key features, and analysis of relationships between variables. The EDA process provides insights into the dataset and helps identify patterns and trends that can inform the modeling process.

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

3. Ensure that the dataset files (`titanic.csv` and `banknote_authentication.csv`) are placed in the same directory.


## Results

The code produces the following results:

- **Decision Tree Classifier**: The code trains a Decision Tree classifier on the dataset and evaluates its performance using metrics such as accuracy, precision, recall, and F1-score. The decision tree visualization provides insights into the decision-making process of the classifier.

- **Exploratory Data Analysis (EDA)**: The code provides insights into the dataset through visualizations and statistical analysis, helping understand the relationships between variables and identifying patterns that can inform the modeling process.
