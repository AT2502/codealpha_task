
# Import all required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style

plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

print("IRIS FLOWER CLASSIFICATION PROJECT")
print("=" * 50)
print("Following the complete synopsis step by step")
print("=" * 50)


# PHASE 1: ENVIRONMENT SETUP


print("\n PHASE 1: ENVIRONMENT SETUP")
print("=" * 30)
print("✓ pandas - Data manipulation")
print("✓ numpy - Numerical operations") 
print("✓ matplotlib - Basic plotting")
print("✓ seaborn - Statistical visualization")
print("✓ scikit-learn - Machine learning algorithms")
print("✓ All libraries imported successfully!")

# PHASE 2: DATA LOADING & EXPLORATION


print("\n PHASE 2: DATA LOADING & EXPLORATION")
print("=" * 40)

# Load the iris dataset
iris = load_iris()
print("✓ Iris dataset loaded successfully!")

# Convert to pandas DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print(f"Dataset Shape: {df.shape}")
print(f"Features: {list(iris.feature_names)}")
print(f"Species: {list(iris.target_names)}")

print("\nFirst 5 rows:")
print(df.head())

print(f"\nSpecies distribution:")
print(df['species_name'].value_counts())

print(f"Missing values: {df.isnull().sum().sum()}")

print("\nBasic Statistics:")
print(df.describe().round(2))


# PHASE 3: DATA VISUALIZATION


print("\n PHASE 3: DATA VISUALIZATION")
print("=" * 35)

# Pair Plot

print("Creating Pair Plot...")
plt.figure(figsize=(12, 10))
sns.pairplot(df, hue='species_name', markers=["o", "s", "D"])
plt.suptitle('Iris Dataset - Pair Plot Analysis', y=1.02, fontsize=16)
plt.tight_layout()
plt.show()

# Box Plots

print("Creating Box Plots...")
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

for i, feature in enumerate(features):
    row = i // 2
    col = i % 2
    sns.boxplot(data=df, x='species_name', y=feature, ax=axes[row, col])
    axes[row, col].set_title(f'{feature.title()} by Species')
    axes[row, col].set_xlabel('Species')
    axes[row, col].set_ylabel(feature.title())

plt.suptitle('Feature Distribution by Species', fontsize=16)
plt.tight_layout()
plt.show()

# Correlation Heatmap

print("Creating Correlation Heatmap...")
plt.figure(figsize=(10, 8))
correlation_matrix = df[features].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, fmt='.2f')
plt.title('Feature Correlation Heatmap', fontsize=16)
plt.tight_layout()
plt.show()

# Distribution Plots

print("Creating Distribution Plots...")
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

for i, feature in enumerate(features):
    row = i // 2
    col = i % 2
    for species in df['species_name'].unique():
        species_data = df[df['species_name'] == species][feature]
        axes[row, col].hist(species_data, alpha=0.7, label=species, bins=15)
    
    axes[row, col].set_title(f'{feature.title()} Distribution')
    axes[row, col].set_xlabel(feature.title())
    axes[row, col].set_ylabel('Frequency')
    axes[row, col].legend()

plt.suptitle('Feature Distributions by Species', fontsize=16)
plt.tight_layout()
plt.show()

print("\n Key Insights:")
print("• Setosa flowers clearly different (smaller petals)")
print("• Versicolor and Virginica overlap more")
print("• Petal length and width highly correlated")
print("• Petal measurements more discriminative")


# PHASE 4: DATA PREPROCESSING


print("\n PHASE 4: DATA PREPROCESSING")
print("=" * 35)

# Feature selection

X = df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
y = df['species']

print(f"✓ Features (X): {X.shape}")
print(f"✓ Target (y): {y.shape}")

# Train-test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ Training samples: {X_train.shape[0]}")
print(f"✓ Testing samples: {X_test.shape[0]}")

# Feature scaling

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✓ Feature scaling completed")

# Verify stratification

print("\nSpecies distribution verification:")
print("Training:", pd.Series(y_train).value_counts().sort_index().values)
print("Testing:", pd.Series(y_test).value_counts().sort_index().values)


# PHASE 5: MODEL TRAINING & SELECTION


print("\n PHASE 5: MODEL TRAINING & SELECTION")
print("=" * 42)

# Initialize models and results storage

models = {}
results = {}
target_names = ['setosa', 'versicolor', 'virginica']

print("Training multiple algorithms...")

# Logistic Regression

print("\n1. Logistic Regression...")
log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_train_scaled, y_train)
log_reg_pred = log_reg.predict(X_test_scaled)
log_reg_accuracy = accuracy_score(y_test, log_reg_pred)

models['Logistic Regression'] = log_reg
results['Logistic Regression'] = {
    'accuracy': log_reg_accuracy,
    'predictions': log_reg_pred,
    'uses_scaling': True
}
print(f"   Accuracy: {log_reg_accuracy:.4f} ({log_reg_accuracy*100:.1f}%)")

# Decision Tree

print("\n2. Decision Tree...")
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)

models['Decision Tree'] = dt
results['Decision Tree'] = {
    'accuracy': dt_accuracy,
    'predictions': dt_pred,
    'uses_scaling': False
}
print(f"   Accuracy: {dt_accuracy:.4f} ({dt_accuracy*100:.1f}%)")

# Random Forest

print("\n3. Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)

models['Random Forest'] = rf
results['Random Forest'] = {
    'accuracy': rf_accuracy,
    'predictions': rf_pred,
    'uses_scaling': False
}
print(f"   Accuracy: {rf_accuracy:.4f} ({rf_accuracy*100:.1f}%)")

# Support Vector Machine

print("\n4. Support Vector Machine...")
svm = SVC(random_state=42, probability=True)
svm.fit(X_train_scaled, y_train)
svm_pred = svm.predict(X_test_scaled)
svm_accuracy = accuracy_score(y_test, svm_pred)

models['Support Vector Machine'] = svm
results['Support Vector Machine'] = {
    'accuracy': svm_accuracy,
    'predictions': svm_pred,
    'uses_scaling': True
}
print(f"   Accuracy: {svm_accuracy:.4f} ({svm_accuracy*100:.1f}%)")

# Model comparison

print("\n MODEL COMPARISON:")
print("=" * 30)
performance_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[model]['accuracy'] for model in results.keys()],
    'Accuracy (%)': [results[model]['accuracy']*100 for model in results.keys()]
})

performance_df = performance_df.sort_values('Accuracy', ascending=False)
print(performance_df.to_string(index=False))

# Find best model

best_model_name = performance_df.iloc[0]['Model']
best_accuracy = performance_df.iloc[0]['Accuracy']
best_model = models[best_model_name]
best_predictions = results[best_model_name]['predictions']

print(f"\n Best Model: {best_model_name} ({best_accuracy*100:.1f}%)")

# Visualize comparison

plt.figure(figsize=(12, 6))
bars = plt.bar(performance_df['Model'], performance_df['Accuracy (%)'], 
               color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
plt.title('Model Performance Comparison', fontsize=16, fontweight='bold')
plt.xlabel('Models', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.ylim(0, 100)

for bar, accuracy in zip(bars, performance_df['Accuracy (%)']):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{accuracy:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# PHASE 6: MODEL EVALUATION


print("\n PHASE 6: MODEL EVALUATION")
print("=" * 35)

print(f"Detailed evaluation of: {best_model_name}")

# Classification Report

print("\n1. Classification Report:")
print(classification_report(y_test, best_predictions, target_names=target_names))

# Confusion Matrix

print("\n2. Confusion Matrix:")
cm = confusion_matrix(y_test, best_predictions)
print("   Rows: Actual | Columns: Predicted")
print(cm)

# Visualize confusion matrix

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, yticklabels=target_names)
plt.title(f'Confusion Matrix - {best_model_name}', fontsize=16)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()
plt.show()

# Feature Importance 
if hasattr(best_model, 'feature_importances_'):
    print("\n3. Feature Importance:")
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(feature_importance.to_string(index=False))
    
    # Visualize feature importance
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(feature_importance['Feature'], feature_importance['Importance'])
    plt.title(f'Feature Importance - {best_model_name}', fontsize=16)
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Importance', fontsize=12)
    plt.xticks(rotation=45)
    
    for bar, importance in zip(bars, feature_importance['Importance']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{importance:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

# Comprehensive evaluation for all models

print("\n4. All Models Performance Summary:")
print("=" * 40)

for model_name in models.keys():
    model_predictions = results[model_name]['predictions']
    model_accuracy = results[model_name]['accuracy']
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, model_predictions, average='weighted')
    
    print(f"\n{model_name}:")
    print(f"  Accuracy: {model_accuracy:.4f} ({model_accuracy*100:.1f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")

# PHASE 7: MODEL TESTING & VALIDATION


print("\n PHASE 7: MODEL TESTING & VALIDATION")
print("=" * 42)

# Test with new flower measurements

print("Testing with hypothetical new flowers:")

new_flowers = np.array([
    [5.1, 3.5, 1.4, 0.2],  # Likely Setosa
    [6.2, 2.8, 4.8, 1.8],  # Likely Versicolor  
    [7.2, 3.0, 5.8, 1.6],  # Likely Virginica
    [5.0, 3.0, 1.6, 0.2],  # Edge case
])

descriptions = [
    "Small flower, tiny petals",
    "Medium flower, moderate petals", 
    "Large flower, long petals",
    "Small flower (edge case)"
]

# Make predictions

if results[best_model_name]['uses_scaling']:
    new_flowers_scaled = scaler.transform(new_flowers)
    predictions = best_model.predict(new_flowers_scaled)
    prediction_probs = best_model.predict_proba(new_flowers_scaled)
else:
    predictions = best_model.predict(new_flowers)
    prediction_probs = best_model.predict_proba(new_flowers)

print("\n Prediction Results:")
print("=" * 30)

for i, (flower, description) in enumerate(zip(new_flowers, descriptions)):
    predicted_species = target_names[predictions[i]]
    confidence = np.max(prediction_probs[i]) * 100
    
    print(f"\nFlower {i+1}: {description}")
    print(f"  Measurements: SL={flower[0]:.1f}, SW={flower[1]:.1f}, PL={flower[2]:.1f}, PW={flower[3]:.1f}")
    print(f"  Prediction: {predicted_species.upper()}")
    print(f"  Confidence: {confidence:.1f}%")
    
    # Show probabilities
    
    for j, species in enumerate(target_names):
        prob = prediction_probs[i][j] * 100
        print(f"    {species}: {prob:.1f}%")

# Cross-validation

print("\n Cross-Validation Results:")
print("=" * 30)

for model_name, model in models.items():
    if results[model_name]['uses_scaling']:
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    else:
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    
    print(f"{model_name}:")
    print(f"  CV Scores: {cv_scores}")
    print(f"  Mean: {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")

# Create prediction function

def predict_iris_species(sepal_length, sepal_width, petal_length, petal_width):
   
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    if results[best_model_name]['uses_scaling']:
        input_data = scaler.transform(input_data)
    
    prediction = best_model.predict(input_data)[0]
    probabilities = best_model.predict_proba(input_data)[0]
    
    return target_names[prediction], np.max(probabilities) * 100, probabilities

print("\n Prediction function created: predict_iris_species()")

# Test the prediction function one more time
print(f"\n🔮 Final Test - Predicting a new flower:")
test_prediction, test_confidence, test_probs = predict_iris_species(5.1, 3.5, 1.4, 0.2)
print(f"   Input: SL=5.1, SW=3.5, PL=1.4, PW=0.2")
print(f"   Prediction: {test_prediction.upper()}")
print(f"   Confidence: {test_confidence:.1f}%")
