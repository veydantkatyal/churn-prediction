# Customer Churn Prediction using Multilayer Perceptron (MLP)

## Overview
In this project, we explore how MLP, a type of neural network, can be used to predict customer churn based on historical customer data. The model takes various customer features (such as usage patterns, contract type, and tenure) and predicts the probability that a customer will churn.

## Key Features
- **Multilayer Perceptron (MLP) model:** A neural network architecture designed to predict customer churn.
- **Data preprocessing:** Handle missing data, normalization, and feature encoding to prepare the data for the model.
- **Evaluation Metrics:** Metrics such as Accuracy, Precision, Recall, and F1-Score are used to evaluate model performance.
- **Model Training and Tuning:** Train the MLP model and fine-tune hyperparameters for optimal performance.

## Motivation

Customer churn is a critical business issue, especially in competitive industries like telecommunications, retail, and financial services. Identifying which customers are likely to leave helps businesses focus their retention efforts on high-risk customers, potentially saving significant revenue. The goal of this project is to build an accurate model that businesses can use to predict churn and take proactive steps to retain valuable customers.

## Technologies Used

- **Python**
- **Scikit-learn**
- **TensorFlow/Keras** 
- **Pandas** 
- **Matplotlib/Seaborn** 

## Project Workflow

1. **Data Collection:**
   - The dataset used in this project typically consists of historical customer records, including features like customer tenure, usage patterns, billing information, and customer service interactions.

2. **Data Preprocessing:**
   - **Handling missing values**: Missing data is either imputed or removed to avoid skewing the model.
   - **Normalization**: All numerical features are scaled to ensure the model converges efficiently.
   - **Categorical Encoding**: Categorical features like 'gender', 'contract type', and 'payment method' are converted into numerical values using one-hot encoding.

3. **Feature Selection:**
   - Not all features are equally important for predicting churn. Feature selection is performed to retain only the most relevant features, helping the model focus on significant patterns.

4. **Model Building (MLP):**
   - The MLP model is created using **Keras**, with an input layer corresponding to the number of features, one or more hidden layers, and an output layer with a sigmoid activation to predict churn probability.
   - **Loss function**: Binary Crossentropy.
   - **Optimizer**: Adam optimizer is used to minimize the loss function during training.

5. **Model Training:**
   - The model is trained on a portion of the data (e.g., 80%) while the remaining data is used for validation and testing.
   - Early stopping is implemented to prevent overfitting.

6. **Evaluation:**
   - The model is evaluated using metrics such as **Accuracy**, **Precision**, **Recall** and **F1-Score**.
   - The confusion matrix and classification report are analyzed to understand the model’s performance.

7. **Hyperparameter Tuning:**
   - Hyperparameters such as the number of layers, number of neurons, batch size, and learning rate are fine-tuned using techniques like grid search or random search to improve model performance.

## Installation and Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/veydantkatyal/churn-prediction.git
   cd churn-prediction

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt

## Dataset
Download the dataset from Kaggle: [telco customer churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
It Contains features like customer tenure, monthly charges, contract type, payment method, and whether or not a customer has churned.

## Results
After training and tuning, the MLP model achieved the following results:

- Accuracy: 78%
- Precision: 82%
- Recall: 88%
- F1-Score: 85%

Visualizations like the confusion matrix can be found in the results section of the [Colab notebook](https://colab.research.google.com/github/veydantkatyal/churn-prediction/blob/main/churn_prediction.ipynb).
