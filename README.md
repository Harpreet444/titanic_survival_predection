# Titanic Survival Prediction using Decision Tree Classifier

This project predicts the survival of passengers on the Titanic using the Titanic dataset. The dataset is cleaned, preprocessed, and then used to train a Decision Tree Classifier.

## Requirements

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib

## Dataset

The dataset used is the `titanic.csv`, which contains information about passengers on the Titanic.

## Steps

1. **Load Dataset**: Load the Titanic dataset using `pandas`.
2. **Data Preparation**: Prepare the dataset by dropping irrelevant columns and handling missing values.
3. **Label Encoding**: Encode categorical features using `LabelEncoder`.
4. **Model Training**: Split the data into training and testing sets, and train the Decision Tree Classifier.
5. **Evaluation**: Measure the prediction score and plot the confusion matrix.

## Code

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib import pyplot as plt
import joblib

# Load dataset
df = pd.read_csv("D:\\Machine_learning\\Decesion_tree\\titanic.csv")

# Drop irrelevant columns
df = df.drop(["PassengerId", "Name", "SibSp", "Parch", "Ticket", "Cabin", "Embarked"], axis="columns")

# Handle missing values
df['Age'] = df['Age'].fillna(df['Age'].mean())

# Label Encoding
label = LabelEncoder()
label.fit(df['Sex'])
df['Sex'] = label.transform(df['Sex'])

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(df.drop(['Survived'], axis='columns'), df['Survived'], test_size=0.2, random_state=10)

# Train Decision Tree Classifier
model = DecisionTreeClassifier()
model.fit(x_train, y_train)

# Measure prediction score
score = model.score(x_test, y_test)
print(f'Prediction Score: {score}')

# Plot confusion matrix
y_pred = model.predict(x_test)
sns.heatmap(confusion_matrix(y_test, y_pred), cmap='Greens', annot=True, xticklabels=['Not Survive', 'Survive'], yticklabels=['Not Survive', 'Survive'])
plt.title("Confusion Matrix - Decision Tree Classifier")
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
plt.show()
