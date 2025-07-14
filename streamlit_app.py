import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="🧠 EEG Disorder Prediction System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .prediction-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .success-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .sidebar-content {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load model components
@st.cache_resource
def load_model_components():
    try:
        model = joblib.load('best_disorder_model.pkl')
        scaler = joblib.load('feature_scaler.pkl')
        selector = joblib.load('feature_selector.pkl')
        sleep_encoder = joblib.load('sleep_encoder.pkl')
        disorder_encoder = joblib.load('disorder_encoder.pkl')
        return model, scaler, selector, sleep_encoder, disorder_encoder
    except FileNotFoundError:
        st.error("❌ Model files not found! Please run the training script first.")
        st.stop()

# Load dataset for analysis
@st.cache_data
def load_dataset():
    try:
        return pd.read_csv('augmented_dataset.csv')
    except FileNotFoundError:
        st.error("❌ Dataset not found! Please generate the dataset first.")
        st.stop()

# Feature engineering function
def engineer_features(delta, theta, alpha, beta, gamma, sleep_state, sleep_encoder):
    """Engineer all features for prediction"""
    input_data = pd.DataFrame({
        'delta': [delta],
        'theta': [theta],
        'alpha': [alpha],
        'beta': [beta],
        'gamma': [gamma],
        'sleep_state': [sleep_state]
    })
    
    # Basic ratios
    input_data['theta_beta_ratio'] = input_data['theta'] / input_data['beta']
    input_data['alpha_theta_ratio'] = input_data['alpha'] / input_data['theta']
    input_data['delta_theta_ratio'] = input_data['delta'] / input_data['theta']
    input_data['gamma_beta_ratio'] = input_data['gamma'] / input_data['beta']
    input_data['alpha_beta_ratio'] = input_data['alpha'] / input_data['beta']
    input_data['delta_alpha_ratio'] = input_data['delta'] / input_data['alpha']
    input_data['gamma_alpha_ratio'] = input_data['gamma'] / input_data['alpha']
    
    # Power features
    input_data['total_power'] = (input_data['delta'] + input_data['theta'] + 
                                input_data['alpha'] + input_data['beta'] + input_data['gamma'])
    input_data['high_freq_power'] = input_data['beta'] + input_data['gamma']
    input_data['low_freq_power'] = input_data['delta'] + input_data['theta']
    input_data['mid_freq_power'] = input_data['alpha'] + input_data['theta']
    
    # Relative powers
    input_data['relative_delta'] = input_data['delta'] / input_data['total_power']
    input_data['relative_theta'] = input_data['theta'] / input_data['total_power']
    input_data['relative_alpha'] = input_data['alpha'] / input_data['total_power']
    input_data['relative_beta'] = input_data['beta'] / input_data['total_power']
    input_data['relative_gamma'] = input_data['gamma'] / input_data['total_power']
    
    # Complex features
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
    input_data['sleep_state_encoded'] = sleep_encoder.transform(input_data['sleep_state'])
    
    return input_data

# Prediction function
def predict_disorder(delta, theta, alpha, beta, gamma, sleep_state, model, scaler, selector, sleep_encoder):
    """Predict disorder and return probabilities"""
    # Feature columns (must match training)
    feature_columns = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'sleep_state_encoded',
                      'theta_beta_ratio', 'alpha_theta_ratio', 'delta_theta_ratio', 
                      'gamma_beta_ratio', 'alpha_beta_ratio', 'delta_alpha_ratio',
                      'gamma_alpha_ratio', 'total_power', 'high_freq_power', 'low_freq_power', 
                      'mid_freq_power', 'relative_delta', 'relative_theta', 'relative_alpha', 
                      'relative_beta', 'relative_gamma', 'spectral_edge', 
                      'theta_alpha_beta_ratio', 'power_variance', 'power_skewness',
                      'power_kurtosis', 'alpha_dominance', 'theta_dominance', 
                      'beta_dominance', 'gamma_dominance', 'high_low_ratio', 'complexity_index']
    
    # Engineer features
    input_data = engineer_features(delta, theta, alpha, beta, gamma, sleep_state, sleep_encoder)
    
    # Select features and scale
    input_features = input_data[feature_columns]
    input_scaled = scaler.transform(input_features)
    input_selected = selector.transform(input_scaled)
    
    # Make prediction
    prediction = model.predict(input_selected)[0]
    prediction_proba = model.predict_proba(input_selected)[0]
    
    # Get all predictions with probabilities
    classes = model.classes_
    prob_dict = dict(zip(classes, prediction_proba))
    sorted_predictions = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
    
    return prediction, sorted_predictions

# Load components
model, scaler, selector, sleep_encoder, disorder_encoder = load_model_components()
df = load_dataset()

# Main title
st.markdown('<h1 class="main-header">🧠 EEG Disorder Prediction System</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Advanced AI-powered neurological disorder detection using EEG brainwave analysis</p>', unsafe_allow_html=True)

# Sidebar for input parameters
st.sidebar.markdown("## 📊 EEG Input Parameters")
st.sidebar.markdown("Adjust the EEG brainwave values and sleep state:")

with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    
    # EEG wave inputs with realistic ranges
    st.subheader("🌊 Brainwave Amplitudes")
    delta = st.slider("Delta Waves (0.5-4 Hz)", 0.0001, 0.002, 0.0008, 0.0001, 
                     help="Deep sleep, unconscious processes")
    theta = st.slider("Theta Waves (4-8 Hz)", 0.0001, 0.002, 0.0007, 0.0001, 
                     help="Light sleep, creativity, meditation")
    alpha = st.slider("Alpha Waves (8-13 Hz)", 0.0001, 0.02, 0.001, 0.0001, 
                     help="Relaxed awareness, closed eyes")
    beta = st.slider("Beta Waves (13-30 Hz)", 0.001, 0.01, 0.003, 0.0001, 
                    help="Active thinking, concentration")
    gamma = st.slider("Gamma Waves (30+ Hz)", 0.001, 0.01, 0.0025, 0.0001, 
                     help="Higher cognitive functions")
    
    st.subheader("😴 Sleep State")
    sleep_state = st.selectbox("Select Sleep State", 
                              ['Awake', 'NREM1', 'NREM2', 'NREM3', 'REM'],
                              help="Current sleep stage")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # EEG Visualization
    st.markdown('<h2 class="sub-header">📈 EEG Wave Analysis</h2>', unsafe_allow_html=True)
    
    # Create EEG wave chart
    wave_data = {
        'Wave Type': ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'],
        'Amplitude': [delta, theta, alpha, beta, gamma],
        'Frequency': ['0.5-4 Hz', '4-8 Hz', '8-13 Hz', '13-30 Hz', '30+ Hz'],
        'Color': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    }
    
    fig = px.bar(wave_data, x='Wave Type', y='Amplitude', 
                title='Current EEG Brainwave Pattern',
                color='Wave Type',
                color_discrete_sequence=wave_data['Color'])
    
    fig.update_layout(
        height=400,
        showlegend=False,
        title_x=0.5,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Power distribution pie chart
    total_power = delta + theta + alpha + beta + gamma
    power_percentages = [
        (delta/total_power)*100,
        (theta/total_power)*100,
        (alpha/total_power)*100,
        (beta/total_power)*100,
        (gamma/total_power)*100
    ]
    
    fig_pie = px.pie(values=power_percentages, 
                    names=['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'],
                    title='EEG Power Distribution',
                    color_discrete_sequence=wave_data['Color'])
    
    fig_pie.update_layout(height=350, title_x=0.5)
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    # Prediction section
    st.markdown('<h2 class="sub-header">🎯 Disorder Prediction</h2>', unsafe_allow_html=True)
    
    if st.button("🔍 Analyze EEG Pattern", type="primary", use_container_width=True):
        with st.spinner("Analyzing brainwave patterns..."):
            try:
                prediction, all_predictions = predict_disorder(
                    delta, theta, alpha, beta, gamma, sleep_state,
                    model, scaler, selector, sleep_encoder
                )
                
                # Main prediction
                confidence = all_predictions[0][1] * 100
                
                st.markdown(f'''
                <div class="prediction-card">
                    <h3>🎯 Primary Prediction</h3>
                    <h2>{prediction}</h2>
                    <p>Confidence: {confidence:.1f}%</p>
                </div>
                ''', unsafe_allow_html=True)
                
                # Top 3 predictions
                st.markdown("### 📊 Top Predictions:")
                for i, (disorder, prob) in enumerate(all_predictions[:3]):
                    confidence_pct = prob * 100
                    color = "success" if i == 0 else "info" if i == 1 else "warning"
                    
                    st.markdown(f'''
                    <div class="success-card">
                        <strong>#{i+1}: {disorder}</strong><br>
                        <small>{confidence_pct:.1f}% confidence</small>
                    </div>
                    ''', unsafe_allow_html=True)
                
                # Probability chart
                disorders = [pred[0] for pred in all_predictions]
                probabilities = [pred[1] * 100 for pred in all_predictions]
                
                fig_prob = px.bar(
                    x=probabilities, 
                    y=disorders,
                    orientation='h',
                    title='Prediction Probabilities',
                    labels={'x': 'Probability (%)', 'y': 'Disorder'},
                    color=probabilities,
                    color_continuous_scale='Viridis'
                )
                
                fig_prob.update_layout(
                    height=500,
                    title_x=0.5,
                    showlegend=False
                )
                
                st.plotly_chart(fig_prob, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")

# Additional features in tabs
st.markdown("---")
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dataset Overview", "🧠 Model Performance", "📖 Disorder Information", "🔬 Feature Analysis"])

with tab1:
    st.markdown("### 📈 Dataset Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f'''
        <div class="metric-card">
            <h3>{len(df)}</h3>
            <p>Total Samples</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'''
        <div class="metric-card">
            <h3>{len(df['disorder'].unique())}</h3>
            <p>Disorders</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'''
        <div class="metric-card">
            <h3>{len(df['sleep_state'].unique())}</h3>
            <p>Sleep States</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        st.markdown(f'''
        <div class="metric-card">
            <h3>85.5%</h3>
            <p>Model Accuracy</p>
        </div>
        ''', unsafe_allow_html=True)
    
    # Disorder distribution
    disorder_counts = df['disorder'].value_counts()
    fig_dist = px.bar(x=disorder_counts.index, y=disorder_counts.values,
                     title="Disorder Distribution in Dataset",
                     labels={'x': 'Disorder', 'y': 'Count'})
    fig_dist.update_layout(height=400, title_x=0.5)
    st.plotly_chart(fig_dist, use_container_width=True)

with tab2:
    st.markdown("### 🏆 Model Performance Metrics")
    
    # Model performance data (from your results)
    model_performance = {
        'Model': ['SVM RBF', 'Voting Hard', 'Voting Soft', 'XGBoost', 'Extra Trees'],
        'Accuracy': [85.5, 84.5, 84.0, 82.5, 82.0],
        'Type': ['Single', 'Ensemble', 'Ensemble', 'Boosting', 'Ensemble']
    }
    
    fig_perf = px.bar(model_performance, x='Model', y='Accuracy',
                     title='Model Performance Comparison',
                     color='Type',
                     text='Accuracy')
    
    fig_perf.update_traces(texttemplate='%{text}%', textposition='outside')
    fig_perf.update_layout(height=400, title_x=0.5)
    st.plotly_chart(fig_perf, use_container_width=True)
    
    st.success("🎉 **Best Model: SVM RBF with 85.5% accuracy**")

with tab3:
    st.markdown("### 🧠 Neurological Disorders Information")
    
    disorders_info = {
        "ADHD": {
            "description": "Attention-Deficit/Hyperactivity Disorder - characterized by difficulty focusing, hyperactivity, and impulsiveness.",
            "eeg_pattern": "Increased theta/beta ratio, especially in frontal regions."
        },
        "Anxiety Disorder": {
            "description": "Persistent excessive worry and fear that interferes with daily activities.",
            "eeg_pattern": "Increased beta activity, particularly in frontal and central regions."
        },
        "Depression": {
            "description": "Persistent feelings of sadness, hopelessness, and loss of interest in activities.",
            "eeg_pattern": "Increased alpha activity, reduced gamma waves."
        },
        "Parkinson's Disease": {
            "description": "Progressive nervous system disorder affecting movement and coordination.",
            "eeg_pattern": "Increased theta waves, decreased alpha activity."
        },
        "Schizophrenia": {
            "description": "Chronic brain disorder affecting thinking, perception, emotions, and behavior.",
            "eeg_pattern": "Reduced gamma oscillations, abnormal synchronization."
        }
    }
    
    selected_disorder = st.selectbox("Select a disorder to learn more:", list(disorders_info.keys()))
    
    if selected_disorder:
        info = disorders_info[selected_disorder]
        st.markdown(f"**{selected_disorder}**")
        st.write(f"📝 **Description:** {info['description']}")
        st.write(f"🧠 **EEG Pattern:** {info['eeg_pattern']}")

with tab4:
    st.markdown("### 🔬 Feature Importance Analysis")
    
    # Feature importance data (from your model)
    feature_importance = {
        'Feature': ['Alpha/Beta Ratio', 'Relative Gamma', 'Gamma/Beta Ratio', 
                   'Relative Alpha', 'Delta/Theta Ratio', 'Mid Freq Power'],
        'Importance': [10.4, 9.7, 7.9, 7.9, 7.1, 6.9]
    }
    
    fig_feat = px.bar(feature_importance, x='Importance', y='Feature',
                     orientation='h',
                     title='Top Feature Importance',
                     color='Importance',
                     color_continuous_scale='Viridis')
    
    fig_feat.update_layout(height=400, title_x=0.5)
    st.plotly_chart(fig_feat, use_container_width=True)
    
    st.info("💡 **Tip:** Alpha/Beta ratio is the most important feature for disorder prediction!")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🧠 <strong>EEG Disorder Prediction System</strong></p>
    <p>Advanced AI-powered neurological analysis • Built with Streamlit & Scikit-learn</p>
    <p><small>⚠️ This system is for research purposes only. Always consult medical professionals for diagnosis.</small></p>
</div>
""", unsafe_allow_html=True)
