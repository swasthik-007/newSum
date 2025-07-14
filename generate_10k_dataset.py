import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import time

def print_progress(message, step=None, total=None):
    """Print progress with timestamp"""
    timestamp = time.strftime("%H:%M:%S")
    if step and total:
        print(f"[{timestamp}] {message} ({step}/{total})")
    else:
        print(f"[{timestamp}] {message}")

# Set random seed for reproducibility
np.random.seed(42)

print_progress("Starting 10,000 sample EEG dataset generation...")

# Define the 10 psychological disorders + Parkinson's disease
disorders = [
    'Depression', 'Anxiety', 'ADHD', 'Bipolar Disorder', 'Schizophrenia',
    'PTSD', 'OCD', 'Autism Spectrum Disorder', 'Eating Disorder', 
    'Sleep Disorder', 'Parkinson Disease'
]

# Define sleep states
sleep_states = ['Awake', 'NREM1', 'NREM2', 'NREM3', 'REM']

# Define realistic EEG patterns for each disorder (based on clinical research)
disorder_patterns = {
    'Depression': {
        'delta': (0.0003, 0.0008),   # Reduced delta activity
        'theta': (0.0008, 0.0015),   # Increased theta
        'alpha': (0.0004, 0.0009),   # Reduced alpha
        'beta': (0.0020, 0.0040),    # Normal to high beta
        'gamma': (0.0015, 0.0030)    # Normal gamma
    },
    'Anxiety': {
        'delta': (0.0004, 0.0009),
        'theta': (0.0007, 0.0013),   # Elevated theta
        'alpha': (0.0005, 0.0010),   # Reduced alpha
        'beta': (0.0035, 0.0055),    # High beta activity
        'gamma': (0.0020, 0.0035)    # Elevated gamma
    },
    'ADHD': {
        'delta': (0.0005, 0.0012),
        'theta': (0.0012, 0.0025),   # High theta (hallmark of ADHD)
        'alpha': (0.0006, 0.0012),
        'beta': (0.0015, 0.0030),    # Lower beta
        'gamma': (0.0010, 0.0025)
    },
    'Bipolar Disorder': {
        'delta': (0.0004, 0.0010),
        'theta': (0.0009, 0.0018),
        'alpha': (0.0007, 0.0014),
        'beta': (0.0025, 0.0045),    # Variable beta
        'gamma': (0.0018, 0.0035)
    },
    'Schizophrenia': {
        'delta': (0.0006, 0.0013),   # Increased delta
        'theta': (0.0010, 0.0020),
        'alpha': (0.0005, 0.0011),   # Reduced alpha
        'beta': (0.0020, 0.0040),
        'gamma': (0.0008, 0.0020)    # Reduced gamma synchrony
    },
    'PTSD': {
        'delta': (0.0004, 0.0009),
        'theta': (0.0011, 0.0022),   # Elevated theta
        'alpha': (0.0006, 0.0012),
        'beta': (0.0030, 0.0050),    # High beta
        'gamma': (0.0015, 0.0030)
    },
    'OCD': {
        'delta': (0.0005, 0.0011),
        'theta': (0.0008, 0.0016),
        'alpha': (0.0007, 0.0013),
        'beta': (0.0028, 0.0048),    # Elevated beta
        'gamma': (0.0018, 0.0033)
    },
    'Autism Spectrum Disorder': {
        'delta': (0.0006, 0.0014),
        'theta': (0.0009, 0.0018),
        'alpha': (0.0008, 0.0015),
        'beta': (0.0022, 0.0042),
        'gamma': (0.0012, 0.0028)    # Altered gamma
    },
    'Eating Disorder': {
        'delta': (0.0005, 0.0012),
        'theta': (0.0010, 0.0019),
        'alpha': (0.0006, 0.0013),
        'beta': (0.0025, 0.0045),
        'gamma': (0.0016, 0.0031)
    },
    'Sleep Disorder': {
        'delta': (0.0008, 0.0018),   # Disrupted delta
        'theta': (0.0012, 0.0024),   # Irregular theta
        'alpha': (0.0005, 0.0011),   # Reduced alpha
        'beta': (0.0018, 0.0038),
        'gamma': (0.0013, 0.0028)
    },
    'Parkinson Disease': {
        'delta': (0.0007, 0.0015),   # Increased delta
        'theta': (0.0014, 0.0026),   # High theta activity
        'alpha': (0.0004, 0.0009),   # Reduced alpha
        'beta': (0.0016, 0.0032),    # Reduced beta
        'gamma': (0.0010, 0.0022)    # Reduced gamma
    }
}

# Sleep state modifiers (how sleep states affect EEG patterns)
sleep_modifiers = {
    'Awake': {'delta': 1.0, 'theta': 1.0, 'alpha': 1.0, 'beta': 1.0, 'gamma': 1.0},
    'NREM1': {'delta': 1.2, 'theta': 1.3, 'alpha': 0.8, 'beta': 0.7, 'gamma': 0.6},
    'NREM2': {'delta': 1.5, 'theta': 1.1, 'alpha': 0.6, 'beta': 0.5, 'gamma': 0.4},
    'NREM3': {'delta': 2.0, 'theta': 0.8, 'alpha': 0.4, 'beta': 0.3, 'gamma': 0.2},
    'REM': {'delta': 0.8, 'theta': 1.4, 'alpha': 0.9, 'beta': 1.2, 'gamma': 1.1}
}

data = []
print_progress("Generating 10,000 EEG samples...")

for i in range(10000):
    if i % 1000 == 0:
        print_progress(f"Generated {i} samples", i, 10000)
    
    # Randomly select disorder and sleep state
    disorder = np.random.choice(disorders)
    sleep_state = np.random.choice(sleep_states)
    
    # Get base EEG pattern for the disorder
    pattern = disorder_patterns[disorder]
    
    # Generate base EEG values with some noise
    delta_base = np.random.uniform(pattern['delta'][0], pattern['delta'][1])
    theta_base = np.random.uniform(pattern['theta'][0], pattern['theta'][1])
    alpha_base = np.random.uniform(pattern['alpha'][0], pattern['alpha'][1])
    beta_base = np.random.uniform(pattern['beta'][0], pattern['beta'][1])
    gamma_base = np.random.uniform(pattern['gamma'][0], pattern['gamma'][1])
    
    # Apply sleep state modifiers
    sleep_mod = sleep_modifiers[sleep_state]
    delta = delta_base * sleep_mod['delta'] * np.random.normal(1.0, 0.1)
    theta = theta_base * sleep_mod['theta'] * np.random.normal(1.0, 0.1)
    alpha = alpha_base * sleep_mod['alpha'] * np.random.normal(1.0, 0.1)
    beta = beta_base * sleep_mod['beta'] * np.random.normal(1.0, 0.1)
    gamma = gamma_base * sleep_mod['gamma'] * np.random.normal(1.0, 0.1)
    
    # Ensure positive values
    delta = max(0.0001, delta)
    theta = max(0.0001, theta)
    alpha = max(0.0001, alpha)
    beta = max(0.0001, beta)
    gamma = max(0.0001, gamma)
    
    # Add individual variability
    individual_noise = np.random.normal(1.0, 0.05)
    delta *= individual_noise
    theta *= individual_noise
    alpha *= individual_noise
    beta *= individual_noise
    gamma *= individual_noise
    
    data.append({
        'delta': round(delta, 6),
        'theta': round(theta, 6),
        'alpha': round(alpha, 6),
        'beta': round(beta, 6),
        'gamma': round(gamma, 6),
        'sleep_state': sleep_state,
        'disorder': disorder
    })

print_progress("Creating DataFrame...")
df = pd.DataFrame(data)

print_progress("Dataset generation complete!")
print(f"Dataset shape: {df.shape}")
print(f"Disorders: {sorted(df['disorder'].unique())}")
print(f"Sleep states: {sorted(df['sleep_state'].unique())}")

# Display distribution
print_progress("Computing disorder distribution...")
disorder_counts = df['disorder'].value_counts()
print("\nDisorder distribution:")
for disorder, count in disorder_counts.items():
    print(f"  {disorder}: {count} ({count/len(df)*100:.1f}%)")

print_progress("Computing sleep state distribution...")
sleep_counts = df['sleep_state'].value_counts()
print("\nSleep state distribution:")
for sleep_state, count in sleep_counts.items():
    print(f"  {sleep_state}: {count} ({count/len(df)*100:.1f}%)")

print_progress("Saving dataset...")
df.to_csv('augmented_dataset_10k.csv', index=False)
print_progress("Dataset saved as 'augmented_dataset_10k.csv'")

print_progress("🎉 10,000 sample EEG dataset generation complete! 🎉")
