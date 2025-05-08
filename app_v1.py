import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #2563EB;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #EFF6FF;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: bold;
        color: #1E40AF;
    }
    .metric-label {
        font-size: 1.2rem;
        color: #475569;
    }
    .insight-text {
        font-size: 1.1rem;
        line-height: 1.6;
        color: #334155;
    }
    .highlight {
        background-color: #DBEAFE;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-weight: 500;
    }
    
    /* New styles for model architecture */
    .model-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    .model-component {
        margin-bottom: 1rem;
    }
    .model-title {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1E40AF;
        margin-bottom: 0.5rem;
    }
    .model-list {
        margin-left: 1.5rem;
    }
    .model-item {
        margin-bottom: 0.3rem;
    }
    
    /* New styles for confusion matrix */
    .confusion-matrix {
        width: 100%;
        max-width: 600px;
        margin: 0 auto;
        border-collapse: collapse;
    }
    .matrix-header {
        background-color: #E0E7FF;
        font-weight: bold;
        text-align: center;
        padding: 0.8rem;
    }
    .matrix-label {
        background-color: #E0E7FF;
        font-weight: bold;
        padding: 0.8rem;
    }
    .matrix-true-negative {
        background-color: #DBEAFE;
        text-align: center;
        padding: 0.8rem;
        font-weight: bold;
    }
    .matrix-false-positive {
        background-color: #FEE2E2;
        text-align: center;
        padding: 0.8rem;
        font-weight: bold;
    }
    .matrix-false-negative {
        background-color: #FEE2E2;
        text-align: center;
        padding: 0.8rem;
        font-weight: bold;
    }
    .matrix-true-positive {
        background-color: #DBEAFE;
        text-align: center;
        padding: 0.8rem;
        font-weight: bold;
    }
    .subtext {
        font-size: 0.8rem;
        color: #4B5563;
        text-align: center;
        display: block;
        margin-top: 0.3rem;
    }
    
    /* Feature importance styles */
    .feature-bar {
        height: 2rem;
        background-color: #3B82F6;
        border-radius: 4px;
        margin-bottom: 0.5rem;
    }
    .feature-label {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.2rem;
    }
    .feature-name {
        font-weight: 500;
    }
    .feature-value {
        color: #4B5563;
    }
</style>
""", unsafe_allow_html=True)
    
    # Model Performance page
    elif page == "Model Performance":
        # Header
        st.markdown('<h1 class="main-header">Fraud Detection Model Performance</h1>', unsafe_allow_html=True)
        
        # Model overview
        st.markdown("## 🧠 Fraud Detection Model Architecture")
        st.markdown("""
        **Model Used:** `XGBoost Classifier`  
        A powerful gradient boosting algorithm that works well with imbalanced datasets like fraud detection.
        ---
        ### 🔍 Input Feature Categories
        - **Transaction characteristics**: amount, time, hour of day, day of week
        - **Account information**: account age, credit utilization, days since last address change
        - **Merchant information**: merchant category, historical fraud rates, frequency
        - **Security features**: CVV match, card present, expiration key match
        - **Behavioral patterns**: average spend per account, transaction count, ratio to account average
        ---
        ### ⚙️ Preprocessing Pipeline
        - Standardization of numeric features using `StandardScaler`
        - One-hot encoding of categorical features (e.g., merchantCategoryCode, posEntryMode)
        - Handling missing values with median/imputation
        - Class imbalance addressed with scaled class weights in XGBoost
        - Model threshold optimized based on F1-score
        - Performance measured using **ROC AUC**, **Precision-Recall AUC**, and **confusion matrix**
        ---
        ### 📌 Why XGBoost?
        XGBoost is known for:
        - High performance on tabular data
        - Robustness against overfitting
        - Support for handling skewed class distributions using `scale_pos_weight`
        - Interpretability through feature importance
        """)
        
        # Model performance metrics
        st.markdown('<h2 class="sub-header">Performance Metrics</h2>', unsafe_allow_html=True)
        
        # Updated metrics based on your provided data
        metrics = {
            'accuracy': 0.87,
            'precision': 0.14,
            'recall': 0.23,
            'f1': 0.17,
            'auc': 0.8198
        }
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{metrics['accuracy']:.2%}</div>
                <div class="metric-label">Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{metrics['precision']:.2%}</div>
                <div class="metric-label">Precision</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{metrics['recall']:.2%}</div>
                <div class="metric-label">Recall</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{metrics['auc']:.2%}</div>
                <div class="metric-label">AUC-ROC</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Additional metrics
        st.markdown("""
        <div style="background-color: #F0F9FF; padding: 15px; border-radius: 8px; margin-top: 20px;">
            <h3 style="color: #1E40AF; font-size: 1.2rem; margin-bottom: 10px;">Additional Performance Metrics</h3>
            <ul style="margin-bottom: 0;">
                <li><strong>PR AUC:</strong> 0.0991</li>
                <li><strong>Cross-Validation Mean ROC AUC:</strong> 0.8190</li>
                <li><strong>F1-score (fraud):</strong> 0.17</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Confusion matrix
        st.markdown('<h2 class="sub-header">Confusion Matrix</h2>', unsafe_allow_html=True)
        
        # Create HTML confusion matrix with updated values from your image
        st.markdown("""
        <table class="confusion-matrix">
            <tr>
                <td></td>
                <td></td>
                <td colspan="2" class="matrix-header">Predicted</td>
            </tr>
            <tr>
                <td></td>
                <td></td>
                <td class="matrix-header">Legitimate (0)</td>
                <td class="matrix-header">Fraud (1)</td>
            </tr>
            <tr>
                <td rowspan="2" style="writing-mode: vertical-rl; transform: rotate(180deg); text-align: center; font-weight: bold; background-color: #E0E7FF; padding: 0.8rem;">Actual</td>
                <td class="matrix-label">Legitimate (0)</td>
                <td class="matrix-true-negative">135,569<span class="subtext">True Negative</span></td>
                <td class="matrix-false-positive">19,221<span class="subtext">False Positive</span></td>
            </tr>
            <tr>
                <td class="matrix-label">Fraud (1)</td>
                <td class="matrix-false-negative">1,452<span class="subtext">False Negative</span></td>
                <td class="matrix-true-positive">"1,031"<span class="subtext">True Positive</span></td>
            </tr>
        </table>
        <div style="text-align: center; margin-top: 10px; font-style: italic; color: #4B5563;">SMOTE Model Confusion Matrix</div>
        """, unsafe_allow_html=True)
        
        # Add explanation of the matrix
        st.markdown("""
        <div style="background-color: #F0F9FF; padding: 15px; border-radius: 8px; margin-top: 20px;">
            <h3 style="color: #1E40AF; font-size: 1.2rem; margin-bottom: 10px;">Confusion Matrix Analysis</h3>
            <p>This confusion matrix shows the performance of our XGBoost classifier with SMOTE balancing:</p>
            <ul>
                <li><strong>True Negatives (135,569):</strong> Correctly identified legitimate transactions</li>
                <li><strong>False Positives (19,221):</strong> Legitimate transactions incorrectly flagged as fraud</li>
                <li><strong>False Negatives (1,452):</strong> Fraudulent transactions missed by the model</li>
                <li><strong>True Positives ("1,031"):</strong> Correctly identified fraudulent transactions</li>
            </ul>
            <p>The model successfully captures 23% of fraud cases (recall) but has a relatively high false positive rate, resulting in 14% precision on fraud cases.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature importance - UPDATED with your data
        st.markdown('<h2 class="sub-header">Top Predictive Features</h2>', unsafe_allow_html=True)
        
        # Create visual feature importance with corrected features and values
        features = [
            ("posEntryMode_05", 0.182827),
            ("transactionAmount", 0.112089),
            ("mcc_fraud_rate", 0.111687),
            ("cardPresent", 0.046556),
            ("merchantCategoryCode_entertainment", 0.042012),
            ("account_avg_amount", 0.034769),
            ("account_txn_count", 0.034097),
            ("merchantCategoryCode_fastfood", 0.033396),
            ("posEntryMode_09", 0.032947),
            ("merchant_txn_count", 0.029333)
        ]

        # Normalize the importance values for visualization (0-100 scale)
        max_importance = max([imp for _, imp in features])
        for feature, importance in features:
            # Calculate percentage for width (0-100%)
            percentage = (importance / max_importance) * 100
            
            st.markdown(f"""
            <div class="feature-label">
                <span class="feature-name">{feature}</span>
                <span class="feature-value">""" + f"{importance:.4f}" + """</span>
            </div>
            <div class="feature-bar" style="width: {percentage}%;"></div>
            """, unsafe_allow_html=True)
            
        # Add model architecture explanation with SMOTE
        st.markdown("""
        <div style="background-color: #F0F9FF; padding: 15px; border-radius: 8px; margin-top: 20px;">
            <h3 style="color: #1E40AF; font-size: 1.2rem; margin-bottom: 10px;">XGBoost with SMOTE Balancing</h3>
            <p>Our model uses <strong>XGBoost (Extreme Gradient Boosting)</strong> with <strong>SMOTE (Synthetic Minority Over-sampling Technique)</strong> to handle the class imbalance in fraud detection:</p>
            <ul>
                <li><strong>SMOTE:</strong> Creates synthetic samples of the minority class (fraud) to balance the dataset</li>
                <li><strong>Threshold Tuning:</strong> Optimized decision threshold to balance precision and recall</li>
                <li><strong>Feature Engineering:</strong> Created domain-specific features like merchant category risk scores</li>
                <li><strong>Cross-Validation:</strong> 5-fold cross-validation with Mean ROC AUC of 0.8190 (0.0045)</li>
            </ul>
            <p>While the model achieves good AUC (0.8198), the precision-recall tradeoff remains challenging due to the inherent imbalance in fraud detection problems.</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()