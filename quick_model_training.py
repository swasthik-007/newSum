import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import warnings
warnings.filterwarnings('ignore')
import time

def print_progress(message):
    """Print progress with timestamp"""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

print("="*60)
print("  QUICK MODEL TRAINING FOR 10K DATASET")
print("="*60)

# Load the large dataset
print_progress("Loading 10k dataset...")
df = pd.read_csv('augmented_dataset_10k.csv')
print_progress(f"Dataset loaded! Shape: {df.shape}")

# Feature engineering (same as before)
print_progress("Engineering features...")

# Basic ratio features
df['theta_beta_ratio'] = df['theta'] / df['beta']
df['alpha_theta_ratio'] = df['alpha'] / df['theta']
df['delta_theta_ratio'] = df['delta'] / df['theta']
df['gamma_beta_ratio'] = df['gamma'] / df['beta']
df['alpha_beta_ratio'] = df['alpha'] / df['beta']
df['delta_alpha_ratio'] = df['delta'] / df['alpha']
df['gamma_alpha_ratio'] = df['gamma'] / df['alpha']

# Power features
df['total_power'] = df['delta'] + df['theta'] + df['alpha'] + df['beta'] + df['gamma']
df['high_freq_power'] = df['beta'] + df['gamma']
df['low_freq_power'] = df['delta'] + df['theta']
df['mid_freq_power'] = df['alpha']

# Relative power features
df['relative_delta'] = df['delta'] / df['total_power']
df['relative_theta'] = df['theta'] / df['total_power']
df['relative_alpha'] = df['alpha'] / df['total_power']
df['relative_beta'] = df['beta'] / df['total_power']
df['relative_gamma'] = df['gamma'] / df['total_power']

# Additional complex features
df['spectral_edge'] = (df['beta'] + df['gamma']) / df['total_power']
df['theta_alpha_beta_ratio'] = df['theta'] / (df['alpha'] + df['beta'])
df['power_variance'] = np.var(df[['delta', 'theta', 'alpha', 'beta', 'gamma']], axis=1)
df['power_skewness'] = ((df['gamma'] - df['delta']) / df['total_power'])
df['power_kurtosis'] = ((df['gamma'] + df['delta'] - 2*df['alpha']) / df['total_power'])

# Dominance features
df['alpha_dominance'] = (df['alpha'] > df[['delta', 'theta', 'beta', 'gamma']].max(axis=1)).astype(int)
df['theta_dominance'] = (df['theta'] > df[['delta', 'alpha', 'beta', 'gamma']].max(axis=1)).astype(int)
df['beta_dominance'] = (df['beta'] > df[['delta', 'theta', 'alpha', 'gamma']].max(axis=1)).astype(int)
df['gamma_dominance'] = (df['gamma'] > df[['delta', 'theta', 'alpha', 'beta']].max(axis=1)).astype(int)

# Advanced ratios
df['high_low_ratio'] = df['high_freq_power'] / df['low_freq_power']
df['complexity_index'] = (df['gamma'] + df['beta']) / (df['delta'] + df['theta'])

# Encode categorical variables
sleep_encoder = LabelEncoder()
disorder_encoder = LabelEncoder()

df['sleep_state_encoded'] = sleep_encoder.fit_transform(df['sleep_state'])
y = disorder_encoder.fit_transform(df['disorder'])

# Feature columns
feature_columns = [
    'delta', 'theta', 'alpha', 'beta', 'gamma', 'sleep_state_encoded',
    'theta_beta_ratio', 'alpha_theta_ratio', 'delta_theta_ratio', 
    'gamma_beta_ratio', 'alpha_beta_ratio', 'delta_alpha_ratio',
    'gamma_alpha_ratio', 'total_power', 'high_freq_power', 'low_freq_power',
    'mid_freq_power', 'relative_delta', 'relative_theta', 'relative_alpha', 
    'relative_beta', 'relative_gamma', 'spectral_edge', 
    'theta_alpha_beta_ratio', 'power_variance', 'power_skewness',
    'power_kurtosis', 'alpha_dominance', 'theta_dominance', 
    'beta_dominance', 'gamma_dominance', 'high_low_ratio', 'complexity_index'
]

X = df[feature_columns]
print_progress(f"Feature matrix shape: {X.shape}")

# Split data
print_progress("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
print_progress("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Feature selection
print_progress("Selecting best features...")
selector = SelectKBest(score_func=f_classif, k=20)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

# Train models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
    'SVM RBF': SVC(C=50, kernel='rbf', gamma='scale', probability=True, random_state=42),
    'XGBoost': xgb.XGBClassifier(n_estimators=150, max_depth=8, learning_rate=0.1, random_state=42, n_jobs=-1)
}

best_accuracy = 0
best_model = None
best_model_name = ""

print("\n" + "="*50)
print("  MODEL TRAINING RESULTS")
print("="*50)

for name, model in models.items():
    print_progress(f"Training {name}...")
    start_time = time.time()
    
    # Train model
    model.fit(X_train_selected, y_train)
    training_time = time.time() - start_time
    
    # Predictions
    y_pred = model.predict(X_test_selected)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Cross validation
    cv_scores = cross_val_score(model, X_train_selected, y_train, cv=3, scoring='accuracy')
    
    print(f"{name} Results:")
    print(f"  Test Accuracy: {accuracy:.4f}")
    print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print(f"  Training Time: {training_time:.2f} seconds")
    print("-" * 50)
    
    # Track best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = name

print(f"\nBest Model: {best_model_name} with {best_accuracy:.4f} accuracy")

# Save the best model and preprocessors
print_progress("Saving models and preprocessors...")
joblib.dump(best_model, 'best_disorder_model.pkl')
joblib.dump(scaler, 'feature_scaler.pkl')
joblib.dump(selector, 'feature_selector.pkl')
joblib.dump(sleep_encoder, 'sleep_encoder.pkl')
joblib.dump(disorder_encoder, 'disorder_encoder.pkl')

# Save detailed classification report
print_progress("Generating detailed classification report...")
y_pred_best = best_model.predict(X_test_selected)
target_names = disorder_encoder.classes_

print("\nDetailed Classification Report:")
print("="*60)
print(classification_report(y_test, y_pred_best, target_names=target_names))

print_progress("Model training complete!")
print(f"Best model saved: {best_model_name}")
print(f"Final accuracy: {best_accuracy:.4f}")
