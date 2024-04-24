# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 18:35:05 2024

@author: shria
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = r'C:\Users\shria\OneDrive\Desktop\IIT-Indore\New_Dataset.xlsx'
df = pd.read_excel(file_path)

# Extract features (X) and target variable (y)
X = df[['Positive_Fault', 'Zero_Fault']]
y = df['Fault_Type']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Base models
base_models = {
    'SVM': SVC(kernel='rbf', C=1.0, gamma='scale'),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(),
    'Naive Bayes': GaussianNB()
}

# Create BaggingClassifier for each base model
bagging_models = {name: BaggingClassifier(model, n_estimators=10, random_state=42) for name, model in base_models.items()}

# Train and evaluate each BaggingClassifier
for name, model in bagging_models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred, zero_division=1)
    conf_matrix = confusion_matrix(y_test, y_pred)
    auc_score = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')
    
    # Print the results
    print(f'\n{name} Model:')
    print(f'Accuracy: {accuracy:.2f}')
    print('AUC Score:', auc_score)
    print('Classification Report:')
    print(classification_rep)
    print('Confusion Matrix:')
    print(conf_matrix)

    # Plot Confusion Matrix with labels
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', cbar=False, xticklabels=df['Fault_Type'].unique(), yticklabels=df['Fault_Type'].unique())
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(f'Confusion Matrix - {name} Model')
    plt.show()
    
    # Plot Scatter plot for Accuracy
    plt.figure()
    plt.scatter(range(len(y_test)), y_test, color='darkgreen', label='True Values')
    plt.scatter(range(len(y_pred)), y_pred, color='darkorange', alpha=0.5, label='Predicted Values')
    plt.xlabel('Sample Index')
    plt.ylabel('Fault Type')
    plt.title(f'Predicted vs True Values - {name} Model')
    plt.legend()
    plt.show()
    
    # Plot Bar plot for Accuracy
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(y_test)), y_test, color='darkgreen', label='True Values')
    plt.bar(range(len(y_pred)), y_pred, color='darkorange', alpha=0.5, label='Predicted Values')
    plt.xlabel('Sample Index')
    plt.ylabel('Fault Type')
    plt.title(f'Predicted vs True Values - {name} Model')
    plt.legend()
    plt.show()

    from sklearn import tree
    import graphviz 

    # Train and evaluate Decision Tree classifier separately
    decision_tree_model = DecisionTreeClassifier()
    decision_tree_model.fit(X_train, y_train)
    y_pred_dt = decision_tree_model.predict(X_test)
    accuracy_dt = accuracy_score(y_test, y_pred_dt)
    classification_rep_dt = classification_report(y_test, y_pred_dt, zero_division=1)
    conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)
    auc_score_dt = roc_auc_score(y_test, decision_tree_model.predict_proba(X_test), multi_class='ovr')

    # Print the results for Decision Tree
    print('\nDecision Tree Model:')
    print(f'Accuracy: {accuracy_dt:.2f}')
    print('AUC Score:', auc_score_dt)
    print('Classification Report:')
    print(classification_rep_dt)
    print('Confusion Matrix:')
    print(conf_matrix_dt)

    # Plot Confusion Matrix for Decision Tree
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_dt, annot=True, cmap='Blues', fmt='g', cbar=False, xticklabels=df['Fault_Type'].unique(), yticklabels=df['Fault_Type'].unique())
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix - Decision Tree Model')
    plt.show()

    # Plot Scatter plot for Accuracy of Decision Tree
    plt.figure()
    plt.scatter(range(len(y_test)), y_test, color='darkgreen', label='True Values')
    plt.scatter(range(len(y_pred_dt)), y_pred_dt, color='darkorange', alpha=0.5, label='Predicted Values')
    plt.xlabel('Sample Index')
    plt.ylabel('Fault Type')
    plt.title('Predicted vs True Values - Decision Tree Model')
    plt.legend()
    plt.show()

    # Visualize the Decision Tree
    dot_data = tree.export_graphviz(decision_tree_model, out_file=None, feature_names=X.columns, class_names=decision_tree_model.classes_, filled=True)
    graph = graphviz.Source(dot_data)
    graph.render("decision_tree_visualization")  # Save the visualization to a file
    graph.render(view=True)  # Display the visualization

