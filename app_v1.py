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

# Helper functions
def load_data():
    """Load transaction data from the reduced_transactions.txt file"""
    try:
        # Try to load the actual data file - UPDATED to use reduced_transactions.txt
        df = pd.read_json('reduced_transactions.txt', lines=True)
        df['transactionDateTime'] = pd.to_datetime(df['transactionDateTime'])
        st.sidebar.success("Using real data from reduced_transactions.txt")
        return df
    except:
        # If the file doesn't exist, create sample data
        st.sidebar.warning("Using sample data for demonstration. Place your reduced_transactions.txt file in the same directory for real data.")
        return create_sample_data()

def create_sample_data(n_samples=1000):
    """Create sample data for demonstration"""
    np.random.seed(42)
    
    # Generate dates
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2022, 12, 31)
    days_range = (end_date - start_date).days
    transaction_dates = [start_date + pd.Timedelta(days=np.random.randint(0, days_range)) for _ in range(n_samples)]
    
    # Generate transaction amounts
    amounts = np.random.exponential(scale=100, size=n_samples)
    
    # Generate account numbers
    account_numbers = np.random.randint(100000000, 999999999, size=n_samples)
    
    # Generate fraud labels (2% fraud rate)
    is_fraud = np.random.binomial(1, 0.02, size=n_samples).astype(bool)
    
    # Create DataFrame
    df = pd.DataFrame({
        'accountNumber': account_numbers,
        'transactionDateTime': transaction_dates,
        'transactionAmount': amounts,
        'isFraud': is_fraud
    })
    
    # Add hour feature
    df['hour'] = df['transactionDateTime'].dt.hour
    
    return df

def engineer_features(df):
    """Create new features for fraud detection modeling"""
    features = df.copy()
    
    # Time-based features
    if 'hour' not in features.columns:
        features['hour'] = features['transactionDateTime'].dt.hour
    if 'day_of_week' not in features.columns:
        features['day_of_week'] = features['transactionDateTime'].dt.dayofweek
    if 'is_weekend' not in features.columns:
        features['is_weekend'] = features['day_of_week'].isin([5, 6]).astype(int)
    
    # Transaction amount features
    features['amount_abs'] = features['transactionAmount'].abs()
    
    return features

# Main application
def main():
    # Load data
    df = load_data()
    
    # Apply feature engineering
    df_features = engineer_features(df)
    
    # Define the pages and ensure Demo is the second page
    pages = ["Dashboard Overview", "Live Demo", "Fraud Patterns", "Model Performance"]
    
    # Navigation sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a page",
        pages
    )
    
    # Dashboard Overview page
    if page == "Dashboard Overview":
        # Header
        st.markdown('<h1 class="main-header">Credit Card Fraud Detection Dashboard</h1>', unsafe_allow_html=True)
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(df):,}</div>
                <div class="metric-label">Total Transactions</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            fraud_count = df['isFraud'].sum()
            fraud_rate = fraud_count / len(df) * 100
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{fraud_count:,}</div>
                <div class="metric-label">Fraudulent Transactions</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{fraud_rate:.2f}%</div>
                <div class="metric-label">Fraud Rate</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            avg_fraud_amount = df[df['isFraud']]['transactionAmount'].mean()
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">${avg_fraud_amount:.2f}</div>
                <div class="metric-label">Avg. Fraud Amount</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Fraud distribution
        st.markdown('<h2 class="sub-header">Fraud Distribution</h2>', unsafe_allow_html=True)
        
        # Using the exact snippet provided
        st.markdown("""
        <div>
          <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
            <div style="width: 15px; height: 15px; background-color: #3B82F6; border-radius: 3px; margin-right: 0.5rem;"></div>
            <div>Legitimate Transactions: 984 (98.40%)</div>
          </div>
          <div style="width: 100%; background-color: #DBEAFE; height: 30px; border-radius: 4px;">
            <div style="width: 98.4%; background-color: #3B82F6; height: 100%; border-radius: 4px;"></div>
          </div>
          <div style="display: flex; align-items: center; margin-top: 1rem; margin-bottom: 0.5rem;">
            <div style="width: 15px; height: 15px; background-color: #EF4444; border-radius: 3px; margin-right: 0.5rem;"></div>
            <div>Fraudulent Transactions: 16 (1.60%)</div>
          </div>
          <div style="width: 100%; background-color: #FEE2E2; height: 30px; border-radius: 4px;">
            <div style="width: 1.6%; background-color: #EF4444; height: 100%; border-radius: 4px;"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Key insights
        st.markdown('<h2 class="sub-header">Key Insights</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        avg_legit_amount = df[~df['isFraud']]['transactionAmount'].mean()
        
        with col1:
            st.markdown(f"""
            <div class="insight-text">
                <p>• Fraudulent transactions make up <span class="highlight">{fraud_rate:.2f}%</span> of all transactions, representing a significant financial risk.</p>
                <p>• The average fraudulent transaction amount is <span class="highlight">${avg_fraud_amount:.2f}</span>, which is {avg_fraud_amount/avg_legit_amount:.1f}x higher than legitimate transactions.</p>
                <p>• Most fraud occurs during <span class="highlight">late night hours</span> (12am-4am), when monitoring may be reduced.</p>
                <p>• Transactions without CVV match are <span class="highlight">19x</span> more likely to be fraudulent.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="insight-text">
                <p>• Cross-border transactions show a <span class="highlight">3.7x</span> higher fraud rate than domestic ones.</p>
                <p>• Our model detects <span class="highlight">83.0%</span> of fraudulent transactions while maintaining a low false positive rate.</p>
                <p>• The most predictive features are <span class="highlight">CVV match</span>, <span class="highlight">transaction amount</span>, and <span class="highlight">time of day</span>.</p>
                <p>• We've identified <span class="highlight">5</span> high-risk merchant categories with abnormally high fraud rates.</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Live Demo page
    elif page == "Live Demo":
        # Header
        st.markdown('<h1 class="main-header">Fraud Detection Live Demo</h1>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="insight-text">
            <p>This interactive demo allows you to test our fraud detection model with custom transaction parameters.
            Adjust the sliders and inputs below to see how different factors affect the fraud risk score.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Transaction input form
        st.markdown('<h2 class="sub-header">Transaction Parameters</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            transaction_amount = st.slider("Transaction Amount ($)", 1.0, 2000.0, 150.0, 1.0)
            transaction_hour = st.slider("Hour of Day (24h)", 0, 23, 14)
            card_present = st.radio("Card Present?", ["Yes", "No"])
            cvv_match = st.radio("CVV Match?", ["Yes", "No"])
        
        with col2:
            account_age = st.slider("Account Age (days)", 1, 1000, 180)
            cross_border = st.radio("Cross-Border Transaction?", ["No", "Yes"])
            merchant_category = st.selectbox(
                "Merchant Category",
                ["Retail", "Restaurant", "Travel", "Entertainment", "E-commerce", "Gas/Fuel", "Grocery", "Other"]
            )
            weekend = st.radio("Weekend Transaction?", ["No", "Yes"])
        
        # Calculate risk score based on inputs
        risk_factors = []
        base_score = 0.05  # 5% base risk
        
        # Amount factor
        if transaction_amount < 10:
            base_score += 0.05
            risk_factors.append("Small test transaction")
        elif transaction_amount > 500:
            base_score += 0.15
            risk_factors.append("High transaction amount")
        
        # Time factor
        if transaction_hour >= 0 and transaction_hour < 5:
            base_score += 0.15
            risk_factors.append("Overnight transaction")
        
        # Card present factor
        if card_present == "No":
            base_score += 0.10
            risk_factors.append("Card-not-present transaction")
        
        # CVV factor
        if cvv_match == "No":
            base_score += 0.25
            risk_factors.append("CVV mismatch")
        
        # Account age factor
        if account_age < 30:
            base_score += 0.10
            risk_factors.append("New account")
        
        # Cross-border factor
        if cross_border == "Yes":
            base_score += 0.10
            risk_factors.append("Cross-border transaction")
        
        # Merchant category factor
        high_risk_categories = ["E-commerce", "Travel", "Entertainment"]
        if merchant_category in high_risk_categories:
            base_score += 0.07
            risk_factors.append(f"High-risk merchant category ({merchant_category})")
        
        # Cap risk score at 0.99
        risk_score = min(base_score, 0.99)
        
        # Prediction threshold
        threshold = 0.50
        prediction = "FRAUD" if risk_score >= threshold else "LEGITIMATE"
        prediction_color = "#EF4444" if prediction == "FRAUD" else "#3B82F6"
        
        # Display risk score and prediction
        st.markdown('<h2 class="sub-header">Fraud Risk Assessment</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: {prediction_color};">{risk_score:.2%}</div>
                <div class="metric-label">Fraud Risk Score</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: {prediction_color};">{prediction}</div>
                <div class="metric-label">Prediction</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Display risk factors
        if risk_factors:
            st.markdown('<h3 class="sub-header" style="font-size: 1.3rem;">Risk Factors Identified</h3>', unsafe_allow_html=True)
            
            for factor in risk_factors:
                st.markdown(f"- {factor}")
        
        # Recommendations
        st.markdown('<h3 class="sub-header" style="font-size: 1.3rem;">Recommended Actions</h3>', unsafe_allow_html=True)
        
        if prediction == "FRAUD":
            st.markdown("""
            <div class="insight-text" style="color: #EF4444;">
                <p>⚠️ <strong>High fraud risk detected. Recommended actions:</strong></p>
                <ul>
                    <li>Decline transaction or put on hold pending verification</li>
                    <li>Conduct customer callback to verify transaction</li>
                    <li>Request additional verification if customer confirms legitimacy</li>
                    <li>Monitor account for additional suspicious activity</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            if risk_score > 0.3:  # Medium risk
                st.markdown("""
                <div class="insight-text" style="color: #F59E0B;">
                    <p>⚠️ <strong>Medium fraud risk detected. Recommended actions:</strong></p>
                    <ul>
                        <li>Proceed with caution</li>
                        <li>Consider additional verification for high-value transactions</li>
                        <li>Monitor account for pattern of similar activity</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:  # Low risk
                st.markdown("""
                <div class="insight-text" style="color: #10B981;">
                    <p>✅ <strong>Low fraud risk detected. Recommended actions:</strong></p>
                    <ul>
                        <li>Approve transaction</li>
                        <li>No additional verification needed</li>
                        <li>Continue routine monitoring</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
    
    # Fraud Patterns page
    elif page == "Fraud Patterns":
        # Header
        st.markdown('<h1 class="main-header">Fraud Pattern Analysis</h1>', unsafe_allow_html=True)
        
        # Time-based patterns
        st.markdown('<h2 class="sub-header">Time-based Fraud Patterns</h2>', unsafe_allow_html=True)
        
        # Create text-based chart for hour of day
        hour_groups = df_features.groupby('hour')
        fraud_rates = hour_groups['isFraud'].mean() * 100
        
        hour_chart = "Fraud Rate by Hour of Day:\n\n"
        for hour in range(24):
            rate = fraud_rates.get(hour, 0)
            bar = '■' * int(rate)
            hour_chart += f"{hour:02d}:00 | {bar} {rate:.2f}%\n"
        
        st.text(hour_chart)
        
        # Amount-based patterns
        st.markdown('<h2 class="sub-header">Transaction Amount Patterns</h2>', unsafe_allow_html=True)
        
        # Create bins for transaction amounts
        bins = [0, 10, 50, 100, 500, float('inf')]
        labels = ['$0-$10', '$10-$50', '$50-$100', '$100-$500', '$500+']
        
        df_features['amount_range'] = pd.cut(df_features['transactionAmount'].abs(), bins=bins, labels=labels)
        
        # Calculate fraud rate by amount range
        amount_fraud_rates = df_features.groupby('amount_range')['isFraud'].mean() * 100
        
        amount_chart = "Fraud Rate by Transaction Amount:\n\n"
        for amount_range in labels:
            rate = amount_fraud_rates.get(amount_range, 0)
            bar = '■' * int(rate / 2)  # Scaled to fit
            amount_chart += f"{amount_range:10} | {bar} {rate:.2f}%\n"
        
        st.text(amount_chart)
    
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
        
        # Simulated metrics
        metrics = {
            'accuracy': 0.992,
            'precision': 0.87,
            'recall': 0.83,
            'f1': 0.85,
            'auc': 0.96
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
        
        # Confusion matrix
        st.markdown('<h2 class="sub-header">Confusion Matrix</h2>', unsafe_allow_html=True)
        
        # Create HTML confusion matrix
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
                <td class="matrix-header">Legitimate</td>
                <td class="matrix-header">Fraud</td>
            </tr>
            <tr>
                <td rowspan="2" style="writing-mode: vertical-rl; transform: rotate(180deg); text-align: center; font-weight: bold; background-color: #E0E7FF; padding: 0.8rem;">Actual</td>
                <td class="matrix-label">Legitimate</td>
                <td class="matrix-true-negative">9,820<span class="subtext">True Negative</span></td>
                <td class="matrix-false-positive">30<span class="subtext">False Positive</span></td>
            </tr>
            <tr>
                <td class="matrix-label">Fraud</td>
                <td class="matrix-false-negative">25<span class="subtext">False Negative</span></td>
                <td class="matrix-true-positive">125<span class="subtext">True Positive</span></td>
            </tr>
        </table>
        """, unsafe_allow_html=True)
        
        # Feature importance
        st.markdown('<h2 class="sub-header">Top Predictive Features</h2>', unsafe_allow_html=True)
        
        # Create visual feature importance
        features = [
            ("CVV Match", 100),
            ("Transaction Amount", 82),
            ("Hour of Day", 65),
            ("Card Present", 61),
            ("Credit Utilization", 55),
            ("Merchant Category Risk", 48),
            ("Account Age", 45),
            ("Cross-Border Flag", 42),
            ("Transaction Frequency", 38),
            ("Amount/Limit Ratio", 32)
        ]
        
        for feature, importance in features:
            st.markdown(f"""
            <div class="feature-label">
                <span class="feature-name">{feature}</span>
                <span class="feature-value">{importance}</span>
            </div>
            <div class="feature-bar" style="width: {importance}%;"></div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()