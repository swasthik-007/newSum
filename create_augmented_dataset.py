import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Load the original dataset (it appears to be space-separated)
df_original = pd.read_csv('dataset.csv', sep='\s+')
print(f"Original dataset shape: {df_original.shape}")
print(f"Columns: {df_original.columns.tolist()}")
print(f"First few rows:")
print(df_original.head())

# Define the disorders including Parkinson's disease
disorders = [
    'Healthy Control',
    'Anxiety Disorder', 
    'Depression',
    'Bipolar Disorder',
    'ADHD',
    'Schizophrenia',
    'PTSD',
    'OCD',
    'Borderline Personality Disorder',
    'Eating Disorder',
    'Parkinson\'s Disease'
]

# Define sleep states
sleep_states = ['Awake', 'NREM1', 'NREM2', 'NREM3', 'REM']

def generate_eeg_pattern(disorder, sleep_state, base_row=None):
    """
    Generate EEG patterns based on disorder and sleep state
    """
    if base_row is not None:
        # Use base values from existing data
        delta_base = base_row['delta']
        theta_base = base_row['theta'] 
        alpha_base = base_row['alpha']
        beta_base = base_row['beta']
        gamma_base = base_row['gamma']
    else:
        # Default healthy awake values
        delta_base = 0.0006
        theta_base = 0.0006
        alpha_base = 0.0008
        beta_base = 0.0029
        gamma_base = 0.0026
    
    # Adjust for sleep state first
    if sleep_state == 'Awake':
        sleep_multipliers = {'delta': 1.0, 'theta': 1.0, 'alpha': 1.0, 'beta': 1.0, 'gamma': 1.0}
    elif sleep_state == 'NREM1':  # Light sleep
        sleep_multipliers = {'delta': 1.2, 'theta': 1.3, 'alpha': 0.8, 'beta': 0.7, 'gamma': 0.8}
    elif sleep_state == 'NREM2':  # Deeper sleep
        sleep_multipliers = {'delta': 1.5, 'theta': 1.4, 'alpha': 0.6, 'beta': 0.5, 'gamma': 0.7}
    elif sleep_state == 'NREM3':  # Deep sleep
        sleep_multipliers = {'delta': 2.0, 'theta': 1.6, 'alpha': 0.4, 'beta': 0.3, 'gamma': 0.5}
    elif sleep_state == 'REM':    # REM sleep
        sleep_multipliers = {'delta': 0.8, 'theta': 1.8, 'alpha': 0.9, 'beta': 1.2, 'gamma': 1.1}
    
    # Apply sleep state modifications
    delta = delta_base * sleep_multipliers['delta']
    theta = theta_base * sleep_multipliers['theta']
    alpha = alpha_base * sleep_multipliers['alpha']
    beta = beta_base * sleep_multipliers['beta']
    gamma = gamma_base * sleep_multipliers['gamma']
    
    # Now adjust for disorders
    if disorder == 'Healthy Control':
        # Keep base values with slight random variation
        pass
    elif disorder == 'Anxiety Disorder':
        beta *= np.random.uniform(1.3, 1.8)  # Increased beta
        gamma *= np.random.uniform(1.1, 1.4)
        alpha *= np.random.uniform(0.8, 0.9)
    elif disorder == 'Depression':
        alpha *= np.random.uniform(1.2, 1.6)  # Increased alpha
        theta *= np.random.uniform(1.1, 1.4)
        gamma *= np.random.uniform(0.7, 0.9)  # Decreased gamma
        beta *= np.random.uniform(0.8, 0.9)
    elif disorder == 'Bipolar Disorder':
        # Variable patterns depending on episode
        if np.random.random() < 0.5:  # Manic episode
            beta *= np.random.uniform(1.4, 1.9)
            gamma *= np.random.uniform(1.2, 1.6)
            theta *= np.random.uniform(0.7, 0.9)
        else:  # Depressive episode
            alpha *= np.random.uniform(1.3, 1.7)
            delta *= np.random.uniform(1.2, 1.5)
            gamma *= np.random.uniform(0.6, 0.8)
    elif disorder == 'ADHD':
        theta *= np.random.uniform(1.4, 1.8)  # Increased theta
        beta *= np.random.uniform(0.7, 0.9)   # Decreased beta (high theta/beta ratio)
        alpha *= np.random.uniform(0.8, 0.9)
    elif disorder == 'Schizophrenia':
        gamma *= np.random.uniform(0.5, 0.8)  # Significantly decreased gamma
        beta *= np.random.uniform(1.1, 1.4)
        theta *= np.random.uniform(1.2, 1.5)
        delta *= np.random.uniform(1.1, 1.3)
    elif disorder == 'PTSD':
        beta *= np.random.uniform(1.3, 1.7)   # Hyperarousal
        alpha *= np.random.uniform(0.7, 0.9)  # Decreased alpha
        theta *= np.random.uniform(1.1, 1.4)
        gamma *= np.random.uniform(0.8, 1.0)
    elif disorder == 'OCD':
        beta *= np.random.uniform(1.2, 1.6)   # Increased beta
        gamma *= np.random.uniform(1.1, 1.4)  # Increased gamma
        alpha *= np.random.uniform(0.8, 0.9)
        theta *= np.random.uniform(1.0, 1.2)
    elif disorder == 'Borderline Personality Disorder':
        # Emotional dysregulation patterns
        beta *= np.random.uniform(1.2, 1.7)
        alpha *= np.random.uniform(0.7, 1.2)  # Variable
        gamma *= np.random.uniform(0.8, 1.3)  # Variable
        theta *= np.random.uniform(1.0, 1.3)
    elif disorder == 'Eating Disorder':
        alpha *= np.random.uniform(1.1, 1.5)  # Often comorbid with anxiety/depression
        beta *= np.random.uniform(1.1, 1.4)
        gamma *= np.random.uniform(0.8, 1.0)
        theta *= np.random.uniform(1.0, 1.2)
    elif disorder == 'Parkinson\'s Disease':
        theta *= np.random.uniform(1.4, 1.9)  # Increased theta
        alpha *= np.random.uniform(0.6, 0.8)  # Decreased alpha
        beta *= np.random.uniform(0.8, 1.1)   # Slightly decreased beta
        delta *= np.random.uniform(1.1, 1.4)  # Increased delta
        gamma *= np.random.uniform(0.7, 0.9)  # Decreased gamma
    
    # Add random noise to make it more realistic
    noise_factor = 0.05
    delta *= np.random.uniform(1-noise_factor, 1+noise_factor)
    theta *= np.random.uniform(1-noise_factor, 1+noise_factor)
    alpha *= np.random.uniform(1-noise_factor, 1+noise_factor)
    beta *= np.random.uniform(1-noise_factor, 1+noise_factor)
    gamma *= np.random.uniform(1-noise_factor, 1+noise_factor)
    
    return {
        'delta': round(delta, 6),
        'theta': round(theta, 6),
        'alpha': round(alpha, 6),
        'beta': round(beta, 6),
        'gamma': round(gamma, 6)
    }

# Create augmented dataset
augmented_data = []

# First, include all original data as "Healthy Control" and "Awake"
for idx, row in df_original.iterrows():
    if idx == 0:  # Skip header row
        continue
    
    augmented_data.append({
        'PersonID': len(augmented_data) + 1,
        'delta': row['delta'],
        'theta': row['theta'],
        'alpha': row['alpha'],
        'beta': row['beta'],
        'gamma': row['gamma'],
        'sleep_state': 'Awake',
        'disorder': 'Healthy Control'
    })

# Generate additional data to reach 1000 samples
target_samples = 1000
current_samples = len(augmented_data)

samples_per_disorder = (target_samples - current_samples) // len(disorders)
remaining_samples = (target_samples - current_samples) % len(disorders)

for disorder_idx, disorder in enumerate(disorders):
    # Determine how many samples for this disorder
    if disorder_idx < remaining_samples:
        n_samples = samples_per_disorder + 1
    else:
        n_samples = samples_per_disorder
    
    for _ in range(n_samples):
        # Randomly select sleep state (more awake samples for realistic distribution)
        sleep_state_probs = [0.6, 0.1, 0.1, 0.1, 0.1]  # More awake samples
        sleep_state = np.random.choice(sleep_states, p=sleep_state_probs)
        
        # Use random base row from original data
        base_row = df_original.iloc[np.random.randint(1, len(df_original))]
        
        # Generate EEG pattern
        eeg_data = generate_eeg_pattern(disorder, sleep_state, base_row)
        
        augmented_data.append({
            'PersonID': len(augmented_data) + 1,
            'delta': eeg_data['delta'],
            'theta': eeg_data['theta'],
            'alpha': eeg_data['alpha'],
            'beta': eeg_data['beta'],
            'gamma': eeg_data['gamma'],
            'sleep_state': sleep_state,
            'disorder': disorder
        })

# Create DataFrame
df_augmented = pd.DataFrame(augmented_data)

# Shuffle the data
df_augmented = df_augmented.sample(frac=1, random_state=42).reset_index(drop=True)
df_augmented['PersonID'] = range(1, len(df_augmented) + 1)

# Save the augmented dataset
df_augmented.to_csv('augmented_dataset.csv', index=False)

print(f"Augmented dataset created with {len(df_augmented)} samples")
print(f"\nDataset shape: {df_augmented.shape}")
print(f"Columns: {list(df_augmented.columns)}")

print(f"\nDisorder distribution:")
print(df_augmented['disorder'].value_counts().sort_index())

print(f"\nSleep state distribution:")
print(df_augmented['sleep_state'].value_counts().sort_index())

print(f"\nFirst 10 rows of augmented dataset:")
print(df_augmented.head(10))

print(f"\nDataset statistics:")
print(df_augmented[['delta', 'theta', 'alpha', 'beta', 'gamma']].describe())
