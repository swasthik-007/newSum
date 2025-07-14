import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def generate_large_eeg_dataset(n_samples=10000):
    """Generate a large, realistic EEG dataset with 10,000 samples"""
    
    # Define the 11 disorders (10 psychological + 1 neurological)
    disorders = [
        'Depression', 'Anxiety', 'ADHD', 'Bipolar Disorder', 'Schizophrenia',
        'PTSD', 'OCD', 'Borderline Personality Disorder', 'Autism Spectrum Disorder',
        'Eating Disorder', 'Parkinson\'s Disease'
    ]
    
    # Define sleep states
    sleep_states = ['Awake', 'NREM1', 'NREM2', 'NREM3', 'REM']
    
    # Create realistic EEG pattern templates for each disorder
    eeg_patterns = {
        'Depression': {
            'delta': (0.0005, 0.0015), 'theta': (0.001, 0.003), 'alpha': (0.0003, 0.001),
            'beta': (0.001, 0.003), 'gamma': (0.0005, 0.002), 'sleep_bias': ['Awake', 'NREM1', 'NREM2']
        },
        'Anxiety': {
            'delta': (0.0003, 0.001), 'theta': (0.0008, 0.0025), 'alpha': (0.0002, 0.0008),
            'beta': (0.003, 0.006), 'gamma': (0.002, 0.004), 'sleep_bias': ['Awake', 'NREM1']
        },
        'ADHD': {
            'delta': (0.0004, 0.0012), 'theta': (0.002, 0.005), 'alpha': (0.0005, 0.0015),
            'beta': (0.001, 0.0025), 'gamma': (0.001, 0.003), 'sleep_bias': ['Awake', 'NREM1', 'REM']
        },
        'Bipolar Disorder': {
            'delta': (0.0006, 0.0018), 'theta': (0.0015, 0.0035), 'alpha': (0.0008, 0.0020),
            'beta': (0.0025, 0.0045), 'gamma': (0.0015, 0.0035), 'sleep_bias': ['Awake', 'NREM1', 'NREM2', 'REM']
        },
        'Schizophrenia': {
            'delta': (0.0008, 0.0025), 'theta': (0.0012, 0.0030), 'alpha': (0.0003, 0.0012),
            'beta': (0.002, 0.004), 'gamma': (0.0008, 0.0025), 'sleep_bias': ['Awake', 'NREM1', 'NREM3']
        },
        'PTSD': {
            'delta': (0.0004, 0.0014), 'theta': (0.0008, 0.0025), 'alpha': (0.0005, 0.0015),
            'beta': (0.003, 0.005), 'gamma': (0.002, 0.004), 'sleep_bias': ['Awake', 'NREM1', 'REM']
        },
        'OCD': {
            'delta': (0.0005, 0.0015), 'theta': (0.0007, 0.0020), 'alpha': (0.0006, 0.0018),
            'beta': (0.003, 0.005), 'gamma': (0.002, 0.004), 'sleep_bias': ['Awake', 'NREM1', 'NREM2']
        },
        'Borderline Personality Disorder': {
            'delta': (0.0005, 0.0016), 'theta': (0.0008, 0.0025), 'alpha': (0.0006, 0.0018),
            'beta': (0.0025, 0.0045), 'gamma': (0.0015, 0.0035), 'sleep_bias': ['Awake', 'NREM1', 'NREM2']
        },
        'Autism Spectrum Disorder': {
            'delta': (0.0006, 0.0020), 'theta': (0.0010, 0.0028), 'alpha': (0.0004, 0.0015),
            'beta': (0.0020, 0.0040), 'gamma': (0.003, 0.005), 'sleep_bias': ['Awake', 'NREM1', 'NREM3', 'REM']
        },
        'Eating Disorder': {
            'delta': (0.0005, 0.0015), 'theta': (0.0006, 0.0020), 'alpha': (0.0008, 0.0025),
            'beta': (0.002, 0.004), 'gamma': (0.001, 0.003), 'sleep_bias': ['Awake', 'NREM1', 'NREM2']
        },
        'Parkinson\'s Disease': {
            'delta': (0.0010, 0.0030), 'theta': (0.0015, 0.0040), 'alpha': (0.0004, 0.0015),
            'beta': (0.0015, 0.0035), 'gamma': (0.0008, 0.0025), 'sleep_bias': ['Awake', 'NREM1', 'NREM2', 'NREM3']
        }
    }
    
    # Generate samples
    data = []
    samples_per_disorder = n_samples // len(disorders)
    
    for disorder in disorders:
        pattern = eeg_patterns[disorder]
        
        for i in range(samples_per_disorder):
            # Generate EEG values with some realistic noise and correlations
            base_noise = np.random.normal(0, 0.0001)
            
            # Generate correlated EEG waves
            delta = np.random.uniform(*pattern['delta']) + base_noise
            theta = np.random.uniform(*pattern['theta']) + base_noise * 0.8
            alpha = np.random.uniform(*pattern['alpha']) + base_noise * 0.6
            beta = np.random.uniform(*pattern['beta']) + base_noise * 0.4
            gamma = np.random.uniform(*pattern['gamma']) + base_noise * 0.3
            
            # Add realistic constraints (no negative values)
            delta = max(0.0001, delta)
            theta = max(0.0001, theta)
            alpha = max(0.0001, alpha)
            beta = max(0.0001, beta)
            gamma = max(0.0001, gamma)
            
            # Sleep state with bias towards certain states for each disorder
            if np.random.random() < 0.7:  # 70% chance of biased sleep state
                sleep_state = np.random.choice(pattern['sleep_bias'])
            else:  # 30% chance of any sleep state
                sleep_state = np.random.choice(sleep_states)
            
            # Adjust EEG values based on sleep state
            if sleep_state == 'NREM3':  # Deep sleep - higher delta
                delta *= np.random.uniform(1.5, 3.0)
                beta *= np.random.uniform(0.3, 0.7)
                gamma *= np.random.uniform(0.2, 0.6)
            elif sleep_state == 'REM':  # REM sleep - higher theta and gamma
                theta *= np.random.uniform(1.3, 2.2)
                gamma *= np.random.uniform(1.2, 2.0)
                delta *= np.random.uniform(0.4, 0.8)
            elif sleep_state == 'Awake':  # Awake - higher beta and gamma
                beta *= np.random.uniform(1.2, 2.0)
                gamma *= np.random.uniform(1.1, 1.8)
                delta *= np.random.uniform(0.5, 0.9)
            
            data.append({
                'PersonID': len(data) + 1,
                'delta': round(delta, 6),
                'theta': round(theta, 6),
                'alpha': round(alpha, 6),
                'beta': round(beta, 6),
                'gamma': round(gamma, 6),
                'sleep_state': sleep_state,
                'disorder': disorder
            })
    
    # Add remaining samples to reach exactly n_samples
    remaining = n_samples - len(data)
    for i in range(remaining):
        disorder = np.random.choice(disorders)
        pattern = eeg_patterns[disorder]
        
        base_noise = np.random.normal(0, 0.0001)
        delta = max(0.0001, np.random.uniform(*pattern['delta']) + base_noise)
        theta = max(0.0001, np.random.uniform(*pattern['theta']) + base_noise * 0.8)
        alpha = max(0.0001, np.random.uniform(*pattern['alpha']) + base_noise * 0.6)
        beta = max(0.0001, np.random.uniform(*pattern['beta']) + base_noise * 0.4)
        gamma = max(0.0001, np.random.uniform(*pattern['gamma']) + base_noise * 0.3)
        
        sleep_state = np.random.choice(sleep_states)
        
        data.append({
            'PersonID': len(data) + 1,
            'delta': round(delta, 6),
            'theta': round(theta, 6),
            'alpha': round(alpha, 6),
            'beta': round(beta, 6),
            'gamma': round(gamma, 6),
            'sleep_state': sleep_state,
            'disorder': disorder
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df['PersonID'] = range(1, len(df) + 1)
    
    return df

def main():
    print("Generating large EEG dataset with 10,000 samples...")
    
    # Generate the dataset
    df = generate_large_eeg_dataset(10000)
    
    # Save the dataset
    df.to_csv('augmented_dataset_10k.csv', index=False)
    
    print(f"Dataset generated successfully!")
    print(f"Shape: {df.shape}")
    print(f"Disorders: {df['disorder'].value_counts()}")
    print(f"Sleep states: {df['sleep_state'].value_counts()}")
    
    # Basic statistics
    print("\nDataset Statistics:")
    print(f"Total samples: {len(df)}")
    print(f"Unique disorders: {df['disorder'].nunique()}")
    print(f"Unique sleep states: {df['sleep_state'].nunique()}")
    
    # Show sample data
    print("\nSample data:")
    print(df.head(10))
    
    print(f"\nDataset saved as 'augmented_dataset_10k.csv'")

if __name__ == "__main__":
    main()
