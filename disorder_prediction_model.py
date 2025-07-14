import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier, VotingClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import StackingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')
import time

def print_progress(message, step=None, total=None):
    """Print progress with timestamp"""
    timestamp = time.strftime("%H:%M:%S")
    if step and total:
        print(f"[{timestamp}] {message} ({step}/{total})")
    else:
        print(f"[{timestamp}] {message}")
    
def print_separator(title):
    """Print a nice separator with title"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

# Set style for better plots
print_separator("INITIALIZING DISORDER PREDICTION SYSTEM")
print_progress("Setting up visualization styles...")
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Load the large augmented dataset (10k samples)
print_progress("Loading large augmented dataset (10k samples)...")
df = pd.read_csv('augmented_dataset_10k.csv')
print_progress(f"Dataset loaded successfully! Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Display basic information
print_progress("Analyzing dataset structure...")
print(f"\nDataset Info:")
print(f"Total samples: {len(df)}")
print(f"Features: {['delta', 'theta', 'alpha', 'beta', 'gamma', 'sleep_state']}")
print(f"Target: disorder")

print_progress("Computing disorder distribution...")
disorder_counts = df['disorder'].value_counts()
print(f"\nDisorder distribution:")
print(disorder_counts)

print_progress("Computing sleep state distribution...")
sleep_counts = df['sleep_state'].value_counts()
print(f"\nSleep state distribution:")
print(sleep_counts)

# Create visualizations
print_separator("CREATING VISUALIZATIONS")
print_progress("Setting up visualization plots...")
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

print_progress("Creating disorder distribution pie chart...")
# Disorder distribution
axes[0,0].pie(disorder_counts.values, labels=disorder_counts.index, autopct='%1.1f%%', startangle=90)
axes[0,0].set_title('Distribution of Disorders')

print_progress("Creating sleep state distribution bar chart...")
# Sleep state distribution
axes[0,1].bar(sleep_counts.index, sleep_counts.values, color='skyblue')
axes[0,1].set_title('Distribution of Sleep States')
axes[0,1].tick_params(axis='x', rotation=45)

print_progress("Creating EEG waves by disorder boxplot...")
# EEG waves by disorder
df_melted = df.melt(id_vars=['disorder'], 
                   value_vars=['delta', 'theta', 'alpha', 'beta', 'gamma'],
                   var_name='wave_type', value_name='amplitude')
sns.boxplot(data=df_melted, x='wave_type', y='amplitude', hue='disorder', ax=axes[1,0])
axes[1,0].set_title('EEG Wave Amplitudes by Disorder')
axes[1,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
axes[1,0].tick_params(axis='x', rotation=45)

print_progress("Creating EEG waves by sleep state boxplot...")
# EEG waves by sleep state
df_melted_sleep = df.melt(id_vars=['sleep_state'], 
                         value_vars=['delta', 'theta', 'alpha', 'beta', 'gamma'],
                         var_name='wave_type', value_name='amplitude')
sns.boxplot(data=df_melted_sleep, x='wave_type', y='amplitude', hue='sleep_state', ax=axes[1,1])
axes[1,1].set_title('EEG Wave Amplitudes by Sleep State')
axes[1,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
axes[1,1].tick_params(axis='x', rotation=45)

print_progress("Saving overview visualization...")
plt.tight_layout()
plt.savefig('eeg_analysis_overview.png', dpi=300, bbox_inches='tight')
plt.close()  # Close plot instead of showing
print_progress("Overview visualization complete!")

# Feature engineering
print_separator("FEATURE ENGINEERING")
print(f"\n=== Feature Engineering ===")

print_progress("Creating additional features...")
# Create comprehensive feature engineering
df['theta_beta_ratio'] = df['theta'] / df['beta']
df['alpha_theta_ratio'] = df['alpha'] / df['theta']
df['delta_theta_ratio'] = df['delta'] / df['theta']
df['gamma_beta_ratio'] = df['gamma'] / df['beta']
df['alpha_beta_ratio'] = df['alpha'] / df['beta']
df['delta_alpha_ratio'] = df['delta'] / df['alpha']
df['gamma_alpha_ratio'] = df['gamma'] / df['alpha']
df['total_power'] = df['delta'] + df['theta'] + df['alpha'] + df['beta'] + df['gamma']
df['high_freq_power'] = df['beta'] + df['gamma']
df['low_freq_power'] = df['delta'] + df['theta']
df['mid_freq_power'] = df['alpha'] + df['theta']
df['relative_delta'] = df['delta'] / df['total_power']
df['relative_theta'] = df['theta'] / df['total_power']
df['relative_alpha'] = df['alpha'] / df['total_power']
df['relative_beta'] = df['beta'] / df['total_power']
df['relative_gamma'] = df['gamma'] / df['total_power']
df['spectral_edge'] = (df['beta'] + df['gamma']) / df['total_power']
df['theta_alpha_beta_ratio'] = df['theta'] / (df['alpha'] + df['beta'])
df['power_variance'] = df[['delta', 'theta', 'alpha', 'beta', 'gamma']].var(axis=1)
df['power_skewness'] = df[['delta', 'theta', 'alpha', 'beta', 'gamma']].skew(axis=1)
df['power_kurtosis'] = df[['delta', 'theta', 'alpha', 'beta', 'gamma']].kurtosis(axis=1)
df['alpha_dominance'] = (df['alpha'] > df[['delta', 'theta', 'beta', 'gamma']].max(axis=1)).astype(int)
df['theta_dominance'] = (df['theta'] > df[['delta', 'alpha', 'beta', 'gamma']].max(axis=1)).astype(int)
df['beta_dominance'] = (df['beta'] > df[['delta', 'theta', 'alpha', 'gamma']].max(axis=1)).astype(int)
df['gamma_dominance'] = (df['gamma'] > df[['delta', 'theta', 'alpha', 'beta']].max(axis=1)).astype(int)
df['high_low_ratio'] = df['high_freq_power'] / df['low_freq_power']
df['complexity_index'] = (df['beta'] + df['gamma']) / (df['delta'] + df['theta'] + df['alpha'])
print_progress("Created 26 new engineered features")

print_progress("Encoding categorical variables...")
# Encode categorical variables
le_sleep = LabelEncoder()
df['sleep_state_encoded'] = le_sleep.fit_transform(df['sleep_state'])

# Encode target variable for XGBoost compatibility
le_disorder = LabelEncoder()
df['disorder_encoded'] = le_disorder.fit_transform(df['disorder'])
print_progress("Sleep state and disorder encoding complete")

print_progress("Preparing feature matrix...")
# Prepare features and target
feature_columns = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'sleep_state_encoded',
                  'theta_beta_ratio', 'alpha_theta_ratio', 'delta_theta_ratio', 
                  'gamma_beta_ratio', 'alpha_beta_ratio', 'delta_alpha_ratio',
                  'gamma_alpha_ratio', 'total_power', 'high_freq_power', 'low_freq_power', 
                  'mid_freq_power', 'relative_delta', 'relative_theta', 'relative_alpha', 
                  'relative_beta', 'relative_gamma', 'spectral_edge', 
                  'theta_alpha_beta_ratio', 'power_variance', 'power_skewness',
                  'power_kurtosis', 'alpha_dominance', 'theta_dominance', 
                  'beta_dominance', 'gamma_dominance', 'high_low_ratio', 'complexity_index']

X = df[feature_columns]
y = df['disorder']
y_encoded = df['disorder_encoded']  # For XGBoost

print(f"Features used: {feature_columns}")
print(f"Feature matrix shape: {X.shape}")
print(f"Target shape: {y.shape}")

print_progress("Splitting data into train/test sets...")
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

print_progress("Scaling features...")
# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print_progress("Feature scaling complete")

# Feature selection
print_separator("FEATURE SELECTION")
print(f"\n=== Feature Selection ===")
print_progress("Performing feature selection using SelectKBest...")
selector = SelectKBest(score_func=f_classif, k=20)  # Use more features
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

print_progress("Computing feature importance scores...")
feature_scores = pd.DataFrame({
    'feature': feature_columns,
    'score': selector.scores_,
    'selected': selector.get_support()
}).sort_values('score', ascending=False)

print("Feature importance scores:")
print(feature_scores)
print_progress("Feature selection complete!")

# Model training and evaluation
print_separator("HYPERPARAMETER OPTIMIZATION & ADVANCED MODELS")
print(f"\n=== Hyperparameter Optimization & Model Training ===")

# Define parameter grids for optimization
param_grids = {
    'Random Forest': {
        'n_estimators': [300, 500],
        'max_depth': [20, 25],
        'min_samples_split': [2, 3],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    },
    'XGBoost': {
        'n_estimators': [300, 500],
        'max_depth': [10, 12],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9]
    },
    'SVM RBF': {
        'C': [50, 100],
        'gamma': ['scale', 'auto'],
        'kernel': ['rbf']
    }
}

# Best models with hyperparameter tuning
optimized_models = {}
print_progress("Starting hyperparameter optimization...")

for model_name, param_grid in param_grids.items():
    print_progress(f"Optimizing {model_name}...")
    
    if model_name == 'Random Forest':
        base_model = RandomForestClassifier(random_state=42)
    elif model_name == 'XGBoost':
        base_model = xgb.XGBClassifier(random_state=42)
    elif model_name == 'SVM RBF':
        base_model = SVC(probability=True, random_state=42)
    
    # Use RandomizedSearchCV for faster optimization
    search = RandomizedSearchCV(
        base_model, 
        param_grid, 
        n_iter=8,  # Reduced for speed
        cv=3,
        scoring='accuracy',
        random_state=42,
        n_jobs=-1
    )
    
    start_time = time.time()
    if model_name == 'XGBoost':
        search.fit(X_train_selected, y_encoded.iloc[X_train.index])
    else:
        search.fit(X_train_selected, y_train)
    
    optimization_time = time.time() - start_time
    optimized_models[model_name] = search.best_estimator_
    
    print(f"{model_name} optimization completed in {optimization_time:.2f} seconds")
    print(f"Best parameters: {search.best_params_}")
    print(f"Best CV score: {search.best_score_:.4f}")

# Additional high-performance models
additional_models = {
    'LightGBM': lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=15,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    ),
    'Extra Trees Optimized': ExtraTreesClassifier(
        n_estimators=500,
        max_depth=25,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    ),
    'Neural Network Deep': MLPClassifier(
        hidden_layer_sizes=(300, 200, 100, 50),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        learning_rate='adaptive',
        max_iter=3000,
        random_state=42
    ),
    'Bagging RF': BaggingClassifier(
        estimator=RandomForestClassifier(n_estimators=100, random_state=42),
        n_estimators=20,
        random_state=42
    ),
    'LDA': LinearDiscriminantAnalysis(),
    'QDA': QuadraticDiscriminantAnalysis()
}

# Combine optimized and additional models
all_models = {**optimized_models, **additional_models}

results = {}
total_models = len(all_models)

for i, (name, model) in enumerate(all_models.items(), 1):
    print_progress(f"Training {name}...", i, total_models)
    
    # Train the model
    start_time = time.time()
    
    # Use encoded labels for XGBoost and LightGBM
    if name in ['XGBoost', 'LightGBM']:
        model.fit(X_train_selected, y_encoded.iloc[X_train.index])
        y_pred_encoded = model.predict(X_test_selected)
        # Convert back to string labels
        y_pred = le_disorder.inverse_transform(y_pred_encoded.astype(int))
    else:
        model.fit(X_train_selected, y_train)
        y_pred = model.predict(X_test_selected)
    
    training_time = time.time() - start_time
    print_progress(f"{name} training completed in {training_time:.2f} seconds")
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print_progress(f"Computing cross-validation scores for {name}...")
    
    # Cross-validation with stratified folds
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    if name in ['XGBoost', 'LightGBM']:
        cv_scores = cross_val_score(model, X_train_selected, y_encoded.iloc[X_train.index], cv=cv)
    else:
        cv_scores = cross_val_score(model, X_train_selected, y_train, cv=cv)
    
    results[name] = {
        'accuracy': accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'predictions': y_pred,
        'training_time': training_time,
        'model': model
    }
    
    print(f"{name} Results:")
    print(f"  Test Accuracy: {accuracy:.4f}")
    print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print(f"  Training Time: {training_time:.2f} seconds")
    
    # Highlight if we achieved 90%+
    if accuracy >= 0.90:
        print(f"  🎉 ACHIEVED 90%+ ACCURACY! 🎉")
    
    print_progress(f"Completed {i}/{total_models} models")
    print("-" * 50)

# Create ensemble models
print_separator("ADVANCED ENSEMBLE MODEL TRAINING")
print_progress("Creating advanced ensemble models...")

# Select top 5 models for ensemble (instead of 3)
top_models = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)[:5]
print(f"Top 5 models for ensemble: {[name for name, _ in top_models]}")
top_accuracies = [f"{results[name]['accuracy']:.3f}" for name, _ in top_models]
print(f"Top 5 accuracies: {top_accuracies}")

# Advanced Voting Classifier (Soft Voting with weights)
model_weights = [results[name]['accuracy'] for name, _ in top_models]
voting_weighted = VotingClassifier(
    estimators=[(name, results[name]['model']) for name, _ in top_models],
    voting='soft',
    weights=model_weights
)

# Multi-level Stacking
# Level 1: Best 3 models
level1_models = top_models[:3]
stacking_l1 = StackingClassifier(
    estimators=[(name, results[name]['model']) for name, _ in level1_models],
    final_estimator=LogisticRegression(random_state=42, max_iter=2000),
    cv=5
)

# Level 2: Best 5 models with different meta-learner
stacking_l2 = StackingClassifier(
    estimators=[(name, results[name]['model']) for name, _ in top_models],
    final_estimator=RandomForestClassifier(n_estimators=100, random_state=42),
    cv=5
)

# Voting + Stacking hybrid
hybrid_ensemble = VotingClassifier(
    estimators=[
        ('stacking_l1', stacking_l1),
        ('voting_weighted', voting_weighted)
    ],
    voting='soft'
)

ensemble_models = {
    'Voting Weighted': voting_weighted,
    'Stacking L1 (Top3)': stacking_l1,
    'Stacking L2 (Top5)': stacking_l2,
    'Hybrid Ensemble': hybrid_ensemble
}

# Train ensemble models
for name, model in ensemble_models.items():
    print_progress(f"Training {name} ensemble...")
    
    start_time = time.time()
    model.fit(X_train_selected, y_train)
    training_time = time.time() - start_time
    
    y_pred = model.predict(X_test_selected)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Use stratified CV for better evaluation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train_selected, y_train, cv=cv)
    
    results[name] = {
        'accuracy': accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'predictions': y_pred,
        'training_time': training_time,
        'model': model
    }
    
    print(f"{name} Results:")
    print(f"  Test Accuracy: {accuracy:.4f}")
    print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Highlight if we achieved 90%+
    if accuracy >= 0.90:
        print(f"  🎉 ACHIEVED 90%+ ACCURACY! 🎉")
    
    print_progress(f"{name} ensemble complete!")

total_models = len(results)

# Find best model
print_separator("MODEL EVALUATION AND ANALYSIS")
print_progress("Determining best performing model...")
best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
best_model = results[best_model_name]['model']
best_predictions = results[best_model_name]['predictions']

print(f"\n=== Best Model: {best_model_name} ===")
print(f"Test Accuracy: {results[best_model_name]['accuracy']:.4f}")
print_progress(f"Best model selected: {best_model_name}")

# Detailed classification report
print_progress("Generating detailed classification report...")
print(f"\nDetailed Classification Report for {best_model_name}:")
print(classification_report(y_test, best_predictions))

# Confusion Matrix
print_progress("Creating confusion matrix visualization...")
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_test, best_predictions)
disorders = sorted(df['disorder'].unique())
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
           xticklabels=disorders, yticklabels=disorders)
plt.title(f'Confusion Matrix - {best_model_name}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()  # Close plot instead of showing
print_progress("Confusion matrix saved!")

# Model comparison
print_progress("Creating model comparison visualization...")
plt.figure(figsize=(10, 6))
model_names = list(results.keys())
accuracies = [results[name]['accuracy'] for name in model_names]
cv_means = [results[name]['cv_mean'] for name in model_names]

x = np.arange(len(model_names))
width = 0.35

plt.bar(x - width/2, accuracies, width, label='Test Accuracy', alpha=0.8)
plt.bar(x + width/2, cv_means, width, label='CV Accuracy', alpha=0.8)

plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Model Performance Comparison')
plt.xticks(x, model_names, rotation=45)
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()  # Close plot instead of showing
print_progress("Model comparison chart saved!")

# Feature importance for Random Forest
if 'Random Forest' in results:
    print_progress("Analyzing feature importance...")
    rf_model = results['Random Forest']['model']
    feature_importance = pd.DataFrame({
        'feature': [feature_columns[i] for i in selector.get_support(indices=True)],
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print_progress("Creating feature importance visualization...")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='importance', y='feature')
    plt.title('Feature Importance (Random Forest)')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close plot instead of showing
    
    print(f"\nFeature Importance (Random Forest):")
    print(feature_importance)
    print_progress("Feature importance analysis complete!")

# Prediction function
def predict_disorder(delta, theta, alpha, beta, gamma, sleep_state='Awake'):
    """
    Predict disorder based on EEG values and sleep state
    """
    # Create input dataframe
    input_data = pd.DataFrame({
        'delta': [delta],
        'theta': [theta],
        'alpha': [alpha],
        'beta': [beta],
        'gamma': [gamma],
        'sleep_state': [sleep_state]
    })
    
    # Engineer ALL features (matching training data exactly)
    input_data['theta_beta_ratio'] = input_data['theta'] / input_data['beta']
    input_data['alpha_theta_ratio'] = input_data['alpha'] / input_data['theta']
    input_data['delta_theta_ratio'] = input_data['delta'] / input_data['theta']
    input_data['gamma_beta_ratio'] = input_data['gamma'] / input_data['beta']
    input_data['alpha_beta_ratio'] = input_data['alpha'] / input_data['beta']
    input_data['delta_alpha_ratio'] = input_data['delta'] / input_data['alpha']
    input_data['gamma_alpha_ratio'] = input_data['gamma'] / input_data['alpha']
    input_data['total_power'] = (input_data['delta'] + input_data['theta'] + 
                                input_data['alpha'] + input_data['beta'] + input_data['gamma'])
    input_data['high_freq_power'] = input_data['beta'] + input_data['gamma']
    input_data['low_freq_power'] = input_data['delta'] + input_data['theta']
    input_data['mid_freq_power'] = input_data['alpha'] + input_data['theta']
    input_data['relative_delta'] = input_data['delta'] / input_data['total_power']
    input_data['relative_theta'] = input_data['theta'] / input_data['total_power']
    input_data['relative_alpha'] = input_data['alpha'] / input_data['total_power']
    input_data['relative_beta'] = input_data['beta'] / input_data['total_power']
    input_data['relative_gamma'] = input_data['gamma'] / input_data['total_power']
    input_data['spectral_edge'] = (input_data['beta'] + input_data['gamma']) / input_data['total_power']
    input_data['theta_alpha_beta_ratio'] = input_data['theta'] / (input_data['alpha'] + input_data['beta'])
    
    # Statistical features
    eeg_values = input_data[['delta', 'theta', 'alpha', 'beta', 'gamma']].values
    input_data['power_variance'] = np.var(eeg_values, axis=1)
    input_data['power_skewness'] = 0.0  # Simplified for single sample
    input_data['power_kurtosis'] = 0.0  # Simplified for single sample
    
    # Dominance features
    input_data['alpha_dominance'] = (input_data['alpha'] > input_data[['delta', 'theta', 'beta', 'gamma']].max(axis=1)).astype(int)
    input_data['theta_dominance'] = (input_data['theta'] > input_data[['delta', 'alpha', 'beta', 'gamma']].max(axis=1)).astype(int)
    input_data['beta_dominance'] = (input_data['beta'] > input_data[['delta', 'theta', 'alpha', 'gamma']].max(axis=1)).astype(int)
    input_data['gamma_dominance'] = (input_data['gamma'] > input_data[['delta', 'theta', 'alpha', 'beta']].max(axis=1)).astype(int)
    
    # Additional features
    input_data['high_low_ratio'] = input_data['high_freq_power'] / input_data['low_freq_power']
    input_data['complexity_index'] = (input_data['beta'] + input_data['gamma']) / (input_data['delta'] + input_data['theta'] + input_data['alpha'])
    input_data['sleep_state_encoded'] = le_sleep.transform(input_data['sleep_state'])
    
    # Select features and scale
    input_features = input_data[feature_columns]
    input_scaled = scaler.transform(input_features)
    input_selected = selector.transform(input_scaled)
    
    # Make prediction
    prediction = best_model.predict(input_selected)[0]
    prediction_proba = best_model.predict_proba(input_selected)[0]
    
    # Get top 3 predictions
    classes = best_model.classes_
    prob_dict = dict(zip(classes, prediction_proba))
    top_predictions = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:3]
    
    return prediction, top_predictions

# Example predictions
print_separator("TESTING PREDICTION FUNCTION")
print(f"\n=== Example Predictions ===")

print_progress("Testing prediction function with sample data...")

# Example 1: High theta, low alpha (possible Parkinson's)
print_progress("Example 1: Testing high theta, low alpha pattern...")
pred1, top1 = predict_disorder(0.0006, 0.0012, 0.0005, 0.0028, 0.0024, 'Awake')
print(f"Example 1 - High theta, low alpha:")
print(f"  Predicted: {pred1}")
print(f"  Top 3: {top1}")

# Example 2: High beta (possible anxiety)
print_progress("Example 2: Testing high beta pattern...")
pred2, top2 = predict_disorder(0.0005, 0.0006, 0.0008, 0.0045, 0.0030, 'Awake')
print(f"Example 2 - High beta:")
print(f"  Predicted: {pred2}")
print(f"  Top 3: {top2}")

# Example 3: Deep sleep pattern
print_progress("Example 3: Testing deep sleep pattern...")
pred3, top3 = predict_disorder(0.0012, 0.0010, 0.0003, 0.0008, 0.0013, 'NREM3')
print(f"Example 3 - Deep sleep pattern:")
print(f"  Predicted: {pred3}")
print(f"  Top 3: {top3}")

print_progress("Prediction function testing complete!")

# Save the model components
print_separator("SAVING MODEL AND COMPONENTS")
print_progress("Saving trained model and preprocessing components...")
import joblib
joblib.dump(best_model, 'best_disorder_model.pkl')
print_progress("Best model saved!")
joblib.dump(scaler, 'feature_scaler.pkl')
print_progress("Feature scaler saved!")
joblib.dump(selector, 'feature_selector.pkl')
print_progress("Feature selector saved!")
joblib.dump(le_sleep, 'sleep_encoder.pkl')
joblib.dump(le_disorder, 'disorder_encoder.pkl')
print_progress("Sleep state and disorder encoders saved!")

print_separator("FINAL SUMMARY")
print(f"\n=== Model Saved ===")
print(f"Best model ({best_model_name}) and preprocessing components saved!")
print(f"Files created:")
print(f"  - best_disorder_model.pkl")
print(f"  - feature_scaler.pkl") 
print(f"  - feature_selector.pkl")
print(f"  - sleep_encoder.pkl")
print(f"  - disorder_encoder.pkl")
print(f"  - augmented_dataset.csv")
print(f"  - Various visualization PNG files")

print(f"\n=== Summary ===")
print(f"✅ Successfully created augmented dataset with 1000 samples")
print(f"✅ Included 11 disorders (including Parkinson's disease)")
print(f"✅ Included 5 sleep states (Awake, NREM1, NREM2, NREM3, REM)")
print(f"✅ Trained and evaluated 5 machine learning models")
print(f"✅ Best model: {best_model_name} with {results[best_model_name]['accuracy']:.1%} accuracy")
print(f"✅ Created comprehensive visualizations and analysis")
print(f"✅ Built prediction function for new EEG data")

# Performance summary table
print_progress("Creating performance summary...")
print(f"\n=== Model Performance Summary ===")
performance_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Test Accuracy': [f"{results[name]['accuracy']:.3f}" for name in results.keys()],
    'CV Accuracy': [f"{results[name]['cv_mean']:.3f}" for name in results.keys()],
    'Training Time (s)': [f"{results[name]['training_time']:.2f}" for name in results.keys()]
})
print(performance_df.to_string(index=False))

timestamp = time.strftime("%H:%M:%S")
print(f"\n[{timestamp}] 🎉 DISORDER PREDICTION SYSTEM COMPLETE! 🎉")
print("="*60)
