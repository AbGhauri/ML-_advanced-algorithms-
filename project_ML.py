import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split , GridSearchCV ,RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from imblearn.over_sampling import SMOTE 


# loading dataset .
filepath = r"E:\UNI\PAI\Project ML\DryBeanDataset\Dry_Bean_Dataset.xlsx"

# reading the excel 
df = pd.read_excel(filepath)

print(df.head())

print (df.info())

print(df['Class'].value_counts())

# Checking formising data
print (df.isnull().sum())  
# no missing data found 


# features and Target 
X = df.drop(columns=['Class'])
Y = df['Class']

# Standardiztion 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

df_scaled = pd.DataFrame(X_scaled, columns= X.columns)
df_scaled['Class'] = Y 

print(df_scaled.head())

# for Classs Imbalance Smote 

X_train ,X_test ,Y_train , Y_test = train_test_split(X_scaled,Y,test_size=0.3 , random_state=42)

# Smote
smote = SMOTE(sampling_strategy ='auto' ,random_state = 42 )
X_train_smote , Y_train_smote = smote.fit_resample(X_train,Y_train)

# Display class distribution after SMOTE
print("Class distribution after SMOTE:")
print(Y_train_smote.value_counts())


# 1st ALgorithm 
# (Random Forest Classifier )
Rf_model = RandomForestClassifier(n_estimators=100 , random_state=42)
Rf_model.fit(X_train_smote,Y_train_smote)

# predicton
Rf_model_prediction = Rf_model.predict(X_test)

# Evaluation 
print("Confusion Matrix:\n" , confusion_matrix(Y_test,Rf_model_prediction))
print("Classification Report:\n ", classification_report(Y_test,Rf_model_prediction))
print("Accuracy Score:\n", accuracy_score(Y_test,Rf_model_prediction))

# Hpyerparameter tunning 
random_forest = RandomForestClassifier(random_state=42)

param_grid= {
    'n_estimators':[100,200,300],
    'max_depth':[10,20,30],
    'min_samples_split':[2,5,10],
    'min_samples_leaf':[1,4],
    'max_features':['sqrt']
}

# GridCv
Rf_grid_search = GridSearchCV(
    estimator= random_forest,
    param_grid= param_grid,
    scoring='accuracy',
    cv=3,
    verbose=1,
    n_jobs=-1
)

Rf_grid_search.fit(X_train_smote,Y_train_smote)

print("Best Parmeters :\n",Rf_grid_search.best_params_)
print("Best Accuracy:\n",Rf_grid_search.best_score_)

# training the final model on best parameters 
best_Rf_model = Rf_grid_search.best_estimator_

best_Rf_prediction = best_Rf_model.predict(X_test)

# Evalution
print("Confusion_matrix by GridSearchCV:\n",confusion_matrix(Y_test,best_Rf_prediction))
print("Classification Report  by GridSearchCV:\n",classification_report(Y_test,best_Rf_prediction))
print("Accuracy by GridSearchCV:\n",accuracy_score(Y_test,best_Rf_prediction))

# Using the RandomSearchCv 
param_dist = {
    'n_estimators': [int(x) for x in np.linspace(100, 300, num=10)],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 10],
    'min_samples_leaf': [ 2, 4],
    'max_features': ['sqrt']
}

Rf_Random_search = RandomizedSearchCV(
    estimator= random_forest,
    param_distributions= param_dist,
    n_iter=50 ,
    scoring='accuracy',
    cv=3,
    verbose=1,
    n_jobs=-1
)

Rf_Random_search.fit(X_train_smote,Y_train_smote)

print("Best Parmeters :\n",Rf_Random_search.best_params_)
print("Best Accuracy:\n",Rf_Random_search.best_score_)

# training the final model on best parameters 
best_Rf_Random_model = Rf_Random_search.best_estimator_

best_Rf_prediction_Random = best_Rf_Random_model.predict(X_test)

# Evalution
print("Confusion_matrix by RandomSearchCV:\n",confusion_matrix(Y_test,best_Rf_prediction_Random))
print("Classification Report  by RandomSearchCV:\n",classification_report(Y_test,best_Rf_prediction_Random))
print("Accuracy by RandomSearchCV:\n",accuracy_score(Y_test,best_Rf_prediction_Random))

# 2nd Algorithm
# Support Vector Machine (SVM)
# Simple SVM Model
svm_model = SVC(random_state=42)
svm_model.fit(X_train_smote,Y_train_smote)

# Prediction
svm_predictions = svm_model.predict(X_test)

# Evaluation
print("Confusion Matrix:\n", confusion_matrix(Y_test, svm_predictions))
print("Classification Report:\n", classification_report(Y_test, svm_predictions))
print("Accuracy Score:\n", accuracy_score(Y_test, svm_predictions))

# Define the parameter grid for GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto'],
}

# Initialize the SVM model
svm_model = SVC(random_state=42)

# Set up GridSearchCV
svm_grid_search = GridSearchCV(
    estimator=svm_model,
    param_grid=param_grid,
    scoring='accuracy',
    cv=3, 
    verbose=1,
    n_jobs=-1  
)

# Perform the grid search on the training data
svm_grid_search.fit(X_train_smote, Y_train_smote)

# Print the best parameters and the corresponding score
print("Best Parameters:", svm_grid_search.best_params_)
print("Best Accuracy:", svm_grid_search.best_score_)

# Use the best parameters to train the final model
best_svm_model = svm_grid_search.best_estimator_

# Evaluate the model on the test set
svm_predictions = best_svm_model.predict(X_test)

print("Confusion Matrix:", confusion_matrix(Y_test, svm_predictions))
print("Classification Report:", classification_report(Y_test, svm_predictions))
print("Accuracy Score:", accuracy_score(Y_test, svm_predictions))

# RandomizedSearchCV Implementation
random_param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto'],
}

svm_random_search = RandomizedSearchCV(
    estimator=svm_model,
    param_distributions=random_param_grid,
    n_iter=20, 
    scoring='accuracy',
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

# Perform the randomized search on the training data
svm_random_search.fit(X_train_smote, Y_train_smote)

# Print the best parameters and the corresponding score
print("Best Parameters (Random Search):", svm_random_search.best_params_)
print("Best Accuracy (Random Search):", svm_random_search.best_score_)

# Use the best parameters to train the final model
best_random_svm_model = svm_random_search.best_estimator_

# Evaluate the model on the test set
svm_random_predictions = best_random_svm_model.predict(X_test)

print("Confusion Matrix (Random Search):", confusion_matrix(Y_test, svm_random_predictions))
print("Classification Report (Random Search):", classification_report(Y_test, svm_random_predictions))
print("Accuracy Score (Random Search):", accuracy_score(Y_test, svm_random_predictions))


# Plot the confusion matrices for GridSearchCV and RandomizedSearchCV
def plot_confusion_matrices(y_true, y_pred_grid, y_pred_random):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Confusion Matrix for GridSearchCV
    cm_grid = confusion_matrix(y_true, y_pred_grid)
    sns.heatmap(cm_grid, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title('Confusion Matrix - GridSearchCV')
    axes[0].set_xlabel('Predicted Labels')
    axes[0].set_ylabel('True Labels')

    # Confusion Matrix for RandomizedSearchCV
    cm_random = confusion_matrix(y_true, y_pred_random)
    sns.heatmap(cm_random, annot=True, fmt='d', cmap='Greens', ax=axes[1])
    axes[1].set_title('Confusion Matrix - RandomizedSearchCV')
    axes[1].set_xlabel('Predicted Labels')
    axes[1].set_ylabel('True Labels')

    plt.tight_layout()
    plt.show()



# For Random Forest
plot_confusion_matrices(Y_test, best_Rf_prediction, best_Rf_prediction_Random)

# For SVM
plot_confusion_matrices(Y_test, svm_predictions, svm_random_predictions)
