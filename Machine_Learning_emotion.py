# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 14:54:30 2023

@author: lukfa
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import os 

os.chdir('C:/Users/lukfa/OneDrive/Desktop/Project2/data')

df_X = pd.read_csv('X_slow.csv')
df_Y = pd.read_csv('y.csv')

df_Y['y_gender'] = np.where((df_Y['y_actors'] == 3) | (df_Y['y_actors'] == 4) | (df_Y['y_actors'] == 6), 'Male', 'Female')

# Perform PCA
pca = PCA(n_components=13)
X_pca = pca.fit_transform(df_X)

# Get unique emotions
unique = np.unique(df_Y['y_emotions'])

# Assign colors to emotions
colors = ['red', 'blue', 'green', 'yellow','black']#,'grey']  # Add more colors as needed
color_dict = {emotion: color for emotion, color in zip(unique, colors)}

# Plot the first two principal components colored by emotions
plt.figure(figsize=(10, 6), dpi=1200)
for emotion in unique:
    indices = np.where(df_Y['y_emotions'] == emotion)[0]
    plt.scatter(X_pca[indices, 0], X_pca[indices, 1], label=emotion, color=color_dict[emotion])

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA: First Two Components CI')
plt.legend(title='Labels')
plt.grid(axis='y', alpha=0.5)
plt.tight_layout() 
plt.show()


# Split the dataset into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(
    df_X,
    df_Y['y_emotions'],
    stratify = df_Y['y_actors'],
    test_size=0.2,
)

print("X - training set:", X_train.shape)
print("Y - training set:", Y_train.shape)
print("X - test set:", X_test.shape)
print("Y - test set:", Y_test.shape)

clf = RandomForestClassifier(max_depth=3, random_state=42)

clf.fit(X_train, Y_train)

print("Score for Train:", clf.score(X_train, Y_train))
print("Cross-Validation Scores for Training:", cross_val_score(clf, X_train, Y_train, cv=5))

print("Score for Test:", clf.score(X_test, Y_test))

# confusion matrix
randomforprediction = clf.predict(X_test)
confusion_matrixrandom = metrics.confusion_matrix(Y_test, randomforprediction)
class_labels = np.unique(Y_test)
plt.figure(figsize=(10, 8), dpi=1200)
cm_displayrandom = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrixrandom, display_labels=class_labels)
cm_displayrandom.plot(cmap='Blues', xticks_rotation=45)
plt.title('Confusion Matrix', fontsize=14)
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
plt.show()

# ROC Curve
# Binarize the true labels
binarized_labels = label_binarize(Y_test, classes=clf.classes_)

# Compute the predicted probabilities for each class
predicted_probs = clf.predict_proba(X_test)

# Compute the ROC curve and AUC for each class
fpr = {}
tpr = {}
roc_auc = {}
for i, class_label in enumerate(clf.classes_):
    fpr[class_label], tpr[class_label], _ = roc_curve(binarized_labels[:, i], predicted_probs[:, i])
    roc_auc[class_label] = auc(fpr[class_label], tpr[class_label])

color_map = {'anger': 'red', 'happiness': 'yellow', 'sadness': 'blue', 'fear': 'purple', 'disgust': 'green'}

plt.figure(figsize=(10, 6), dpi=1200)
for i, class_label in enumerate(clf.classes_):
    plt.plot(fpr[class_label], tpr[class_label], 
             label=f"Class {class_label} (AUC = {roc_auc[class_label]:.2f})", color=color_map[class_label])

plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for random guessing
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) For Emotions')
plt.legend(loc='lower right')
plt.grid(axis='y', alpha=0.5)
plt.tight_layout()  
plt.show()


feature_importance = clf.feature_importances_

# Sort feature importance in descending order
sorted_indices = np.argsort(feature_importance)[::-1]
sorted_importance = feature_importance[sorted_indices]
sorted_features = X_train.columns[sorted_indices]

# Set the threshold for feature importance
threshold = 0.005

# Filter the feature importances and corresponding feature names based on the threshold
filtered_importance = sorted_importance[sorted_importance > threshold]
filtered_features = sorted_features[sorted_importance > threshold]

# Plot the filtered feature importances
plt.figure(figsize=(10, 6), dpi=1200)
plt.bar(range(len(filtered_importance)), filtered_importance)
plt.xticks(range(len(filtered_importance)), filtered_features, rotation='vertical')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance in Random Forest (Threshold: > 0.05)')
plt.grid(axis='y', alpha=0.5)
plt.tight_layout()
plt.show()
