import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler


## loading the data set from sklearn
iris = load_iris()
x = iris.data ## Contain feature data sepal and petal lenght and widht
y= iris.target ## contains the target labels (soecies of flower 0, 1 and 2)


##Data frame creation using pandas
iris_df = pd.DataFrame(data = x, columns = iris.feature_names)
iris_df["species"] = iris.target

## Visualizing the data set
iris_df.head()

### Ensuring that the dataset is complete
missing_values = iris_df.isnull().sum()
print(f"Missing values:\n{missing_values}")

# Data type verification 
data_types = iris_df.dtypes
print(f"Data types:\n{data_types}")

## generating a statistical summary, to understand the spread and tendency of the features
stats = iris_df.describe()
print(f"Statistical summary:\n{stats}")

## Ploting histograms to visualize the distribuition
iris_df.drop('species', axis=1).hist(bins=10, figsize=(10, 6))
plt.suptitle('Distributions')
plt.show()

# Using boxplot by species to check for outliers
species_names = ['Setosa', 'Versicolor', 'Virginica']

# Create boxplots of features split by species with proper species labels
plt.figure(figsize=(12, 8))
sns.boxplot(data=iris_df, x='species', y='sepal length (cm)', palette='pastel')
plt.title('Boxplot of Sepal Length by Species')
plt.xticks(ticks=[0, 1, 2], labels=species_names)
plt.show()

plt.figure(figsize=(12, 8))
sns.boxplot(data=iris_df, x='species', y='sepal width (cm)', palette='pastel')
plt.title('Boxplot of Sepal Width by Species')
plt.xticks(ticks=[0, 1, 2], labels=species_names) 
plt.show()

plt.figure(figsize=(12, 8))
sns.boxplot(data=iris_df, x='species', y='petal length (cm)', palette='pastel')
plt.title('Boxplot of Petal Length by Species')
plt.xticks(ticks=[0, 1, 2], labels=species_names)
plt.show()

plt.figure(figsize=(12, 8))
sns.boxplot(data=iris_df, x='species', y='petal width (cm)', palette='pastel')
plt.title('Boxplot of Petal Width by Species')
plt.xticks(ticks=[0, 1, 2], labels=species_names)
plt.show()

# Creating a correlation matrix - shown a linear correllation
correlation_matrix = iris_df.drop('species', axis=1).corr()

# Plot heatmap of correlations to vizualize the relationship between the variables
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()


# Pairplot to see the relationship between features
## better visualization of the correlation between the variables
sns.pairplot(iris_df, hue='species', palette='viridis')
plt.suptitle('Variables correlation', y=1.02)
plt.show()

##Data processing - scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)

### Split the data set in to testing and training 30 - 70 - training the model and reduce the risk of overfitting
x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)


### Train Support vector machine classifier (SVM)

clf = SVC(kernel='linear')
clf.fit(x_train, y_train)


###Prediction of the model

y_pred = clf.predict(x_test)


###Model evaluation
## the model will learn to separate the data in to the three species based on the variables 
accuracy = accuracy_score(y_test, y_pred) ## comparing the predicted labels (y_pred) with the true labels(y_test)

print(f"Accuracy:{accuracy * 100:.2f}%")## printing the result 

#Creating the confusion matrix
cm = confusion_matrix(y_test, y_pred) ## cofusion matrix was created to verify the models performance

#Plotting the confusion matrix using a heatmap
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Ground truth')
plt.show()
##the heatmap can help us visualize the accuracy of the model



# Plot three features in 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# 3D features (sepal length, sepal width, and petal length)
colors = ['yellow', 'purple', 'green']
labels = iris.target_names

# data point color coded
for i, color in enumerate(colors):
    mask = y_test == i
    ax.scatter(x_test[mask, 0], x_test[mask, 1], x_test[mask, 2], 
               c=color, label=labels[i], s=100)

# Set axis labels
ax.set_xlabel(iris.feature_names[0])
ax.set_ylabel(iris.feature_names[1])
ax.set_zlabel(iris.feature_names[2])
ax.set_title('Iris morpholofical structures x species')

# Add a legend
ax.legend(loc='best', title="Species", fontsize=10, title_fontsize=12)

plt.show()



