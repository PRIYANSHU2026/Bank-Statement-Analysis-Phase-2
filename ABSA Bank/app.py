import streamlit as st
import pdfplumber
import re
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import hashlib
import json

# Advanced ML and NLP imports
try:
    import xgboost as xgb
    from sklearn.ensemble import RandomForestClassifier, IsolationForest
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    import nltk
    from textblob import TextBlob
except ImportError as e:
    st.warning(f"Some advanced features may not be available. Missing: {str(e)}")

# Configuration
st.set_page_config(
    page_title='Enhanced Bank Statement Analysis',
    page_icon='🏦',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Initialize session state
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = None
if 'transactions_df' not in st.session_state:
    st.session_state.transactions_df = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

class DatabaseManager:
    """Handles all database operations for user profiles and financial data"""

    def __init__(self):
        self.db_path = 'financial_analysis.db'
        self.init_database()

    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # User profiles table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                financial_goals TEXT,
                risk_tolerance TEXT,
                monthly_income REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Financial data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS financial_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                date TEXT NOT NULL,
                description TEXT,
                amount REAL,
                category TEXT,
                balance REAL,
                analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES user_profiles (user_id)
            )
        ''')

        # Recommendations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                recommendation_type TEXT,
                title TEXT,
                description TEXT,
                priority INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES user_profiles (user_id)
            )
        ''')

        conn.commit()
        conn.close()

    def create_user(self, user_id, name, email, password, financial_goals="", risk_tolerance="moderate", monthly_income=0):
        """Create a new user profile"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        password_hash = hashlib.sha256(password.encode()).hexdigest()

        try:
            cursor.execute('''
                INSERT INTO user_profiles (user_id, name, email, password_hash, financial_goals, risk_tolerance, monthly_income)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (user_id, name, email, password_hash, financial_goals, risk_tolerance, monthly_income))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
        finally:
            conn.close()

    def authenticate_user(self, email, password):
        """Authenticate user login"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        password_hash = hashlib.sha256(password.encode()).hexdigest()
        cursor.execute('SELECT user_id, name FROM user_profiles WHERE email = ? AND password_hash = ?',
                      (email, password_hash))
        result = cursor.fetchone()
        conn.close()

        return result

    def get_user_profile(self, user_id):
        """Get user profile data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM user_profiles WHERE user_id = ?', (user_id,))
        result = cursor.fetchone()
        conn.close()

        if result:
            columns = ['id', 'user_id', 'name', 'email', 'password_hash', 'financial_goals', 'risk_tolerance', 'monthly_income', 'created_at']
            return dict(zip(columns, result))
        return None

class PersonalizationEngine:
    """Advanced personalization and recommendation system"""

    def __init__(self, user_profile=None):
        self.user_profile = user_profile
        self.scaler = StandardScaler()

    def analyze_spending_patterns(self, df):
        """Analyze user spending patterns using advanced ML"""
        if df.empty:
            return {}

        # Calculate advanced metrics
        spending_by_category = df.groupby('Category')['Amount'].agg(['sum', 'mean', 'count', 'std'])

        # Time-based analysis
        df['Date'] = pd.to_datetime(df['Date'])
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['Month'] = df['Date'].dt.month
        df['IsWeekend'] = df['DayOfWeek'].isin([5, 6])

        patterns = {
            'spending_by_category': spending_by_category.to_dict(),
            'weekend_vs_weekday': {
                'weekend_avg': df[df['IsWeekend']]['Amount'].mean(),
                'weekday_avg': df[~df['IsWeekend']]['Amount'].mean()
            },
            'monthly_trends': df.groupby('Month')['Amount'].mean().to_dict(),
            'transaction_frequency': len(df) / max(1, (df['Date'].max() - df['Date'].min()).days),
            'balance_trend': df['Balance'].iloc[-1] - df['Balance'].iloc[0] if len(df) > 1 else 0
        }

        return patterns

    def generate_personalized_recommendations(self, df, patterns):
        """Generate personalized financial recommendations"""
        recommendations = []

        if df.empty:
            return recommendations

        # Budget recommendations
        total_spending = abs(df[df['Amount'] < 0]['Amount'].sum())

        if self.user_profile and self.user_profile.get('monthly_income', 0) > 0:
            spending_ratio = total_spending / self.user_profile['monthly_income']

            if spending_ratio > 0.8:
                recommendations.append({
                    'type': 'budget',
                    'priority': 'high',
                    'title': 'Reduce Monthly Spending',
                    'description': f'Your spending is {spending_ratio:.1%} of your income. Consider reducing expenses by 20%.'
                })
            elif spending_ratio < 0.5:
                recommendations.append({
                    'type': 'savings',
                    'priority': 'medium',
                    'title': 'Increase Savings',
                    'description': f'Great job! You\'re only spending {spending_ratio:.1%} of your income. Consider increasing your savings rate.'
                })

        # Category-specific recommendations
        for category, data in patterns.get('spending_by_category', {}).items():
            if data.get('sum', 0) < 0 and abs(data['sum']) > 1000:  # Significant spending categories
                recommendations.append({
                    'type': 'category',
                    'priority': 'medium',
                    'title': f'Optimize {category} Spending',
                    'description': f'You spent R{abs(data["sum"]):.2f} on {category}. Consider reviewing these expenses.'
                })

        # Investment recommendations
        if patterns.get('balance_trend', 0) > 5000:
            recommendations.append({
                'type': 'investment',
                'priority': 'medium',
                'title': 'Consider Investment Options',
                'description': 'Your balance is growing steadily. Consider investing excess funds for better returns.'
            })

        return recommendations

    def calculate_financial_health_score(self, df, patterns):
        """Calculate comprehensive financial health score"""
        if df.empty:
            return 0

        score_components = {
            'income_stability': 0,
            'spending_control': 0,
            'savings_rate': 0,
            'transaction_diversity': 0,
            'balance_growth': 0
        }

        # Income stability (based on regular credits)
        credits = df[df['Amount'] > 0]
        if len(credits) > 0:
            credit_std = credits['Amount'].std()
            credit_mean = credits['Amount'].mean()
            stability = max(0, 1 - (credit_std / credit_mean if credit_mean > 0 else 1))
            score_components['income_stability'] = min(stability * 25, 25)

        # Spending control (consistent spending patterns)
        debits = df[df['Amount'] < 0]
        if len(debits) > 0:
            spending_consistency = 1 - (debits['Amount'].std() / abs(debits['Amount'].mean()) if debits['Amount'].mean() != 0 else 1)
            score_components['spending_control'] = max(0, spending_consistency * 20)

        # Savings rate
        if self.user_profile and self.user_profile.get('monthly_income', 0) > 0:
            total_spending = abs(df[df['Amount'] < 0]['Amount'].sum())
            savings_rate = 1 - (total_spending / self.user_profile['monthly_income'])
            score_components['savings_rate'] = max(0, min(savings_rate * 25, 25))

        # Transaction diversity
        unique_categories = df['Category'].nunique()
        diversity_score = min(unique_categories / 8 * 15, 15)  # Max 15 points for 8+ categories
        score_components['transaction_diversity'] = diversity_score

        # Balance growth
        balance_trend = patterns.get('balance_trend', 0)
        if balance_trend > 0:
            score_components['balance_growth'] = min(balance_trend / 5000 * 15, 15)

        total_score = sum(score_components.values())
        return min(100, max(0, total_score)), score_components

class AdvancedAnalytics:
    """Advanced analytics and ML models for financial analysis"""

    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()

    def detect_anomalies(self, df):
        """Detect unusual transactions using Isolation Forest"""
        if len(df) < 10:
            return df

        # Prepare features for anomaly detection
        features = df[['Amount']].copy()
        features['DayOfWeek'] = pd.to_datetime(df['Date']).dt.dayofweek
        features['IsWeekend'] = features['DayOfWeek'].isin([5, 6]).astype(int)

        # Scale features
        features_scaled = self.scaler.fit_transform(features)

        # Detect anomalies
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomalies = iso_forest.fit_predict(features_scaled)

        df_copy = df.copy()
        df_copy['IsAnomaly'] = anomalies == -1

        return df_copy

    def predict_future_spending(self, df, days_ahead=30):
        """Predict future spending patterns"""
        if len(df) < 10:
            return None

        # Prepare time series data
        df_daily = df.groupby(pd.to_datetime(df['Date']).dt.date)['Amount'].sum().reset_index()
        df_daily['Date'] = pd.to_datetime(df_daily['Date'])
        df_daily = df_daily.sort_values('Date')

        # Simple linear trend prediction
        x = np.arange(len(df_daily))
        y = df_daily['Amount'].values

        coeffs = np.polyfit(x, y, 1)

        # Predict future values
        future_x = np.arange(len(df_daily), len(df_daily) + days_ahead)
        future_predictions = np.polyval(coeffs, future_x)

        future_dates = [df_daily['Date'].max() + timedelta(days=i+1) for i in range(days_ahead)]

        return pd.DataFrame({
            'Date': future_dates,
            'Predicted_Amount': future_predictions
        })

    def enhanced_loan_prediction(self, df):
        """Enhanced loan eligibility prediction using XGBoost"""
        try:
            # Load training data
            training_data = pd.read_csv('absa.csv')

            # Prepare features
            total_credits = df[df['Amount'] > 0]['Amount'].sum()
            total_debits = abs(df[df['Amount'] < 0]['Amount'].sum())
            num_transactions = len(df)

            # Advanced features
            avg_transaction_amount = df['Amount'].mean()
            transaction_variability = df['Amount'].std()
            balance_trend = df['Balance'].iloc[-1] - df['Balance'].iloc[0] if len(df) > 1 else 0

            # Additional features
            credit_frequency = len(df[df['Amount'] > 0]) / max(1, len(df))
            max_single_debit = abs(df[df['Amount'] < 0]['Amount'].min()) if len(df[df['Amount'] < 0]) > 0 else 0
            balance_volatility = df['Balance'].std()

            # Prepare feature vector
            features = pd.DataFrame({
                'total_credits': [total_credits],
                'total_debits': [total_debits],
                'num_transactions': [num_transactions],
                'avg_transaction_amount': [avg_transaction_amount],
                'transaction_variability': [transaction_variability],
                'balance_trend': [balance_trend],
                'credit_frequency': [credit_frequency],
                'max_single_debit': [max_single_debit],
                'balance_volatility': [balance_volatility]
            })

            # Train XGBoost model if available
            if 'xgb' in globals():
                X_train = training_data[['total_credits', 'total_debits', 'num_transactions',
                                       'avg_transaction_amount', 'transaction_variability', 'balance_trend']]
                y_train = training_data['Eligibility (y)']

                # Add missing features with defaults
                for col in features.columns:
                    if col not in X_train.columns:
                        X_train[col] = 0

                model = xgb.XGBClassifier(random_state=42)
                model.fit(X_train[features.columns], y_train)

                prediction = model.predict(features)[0]
                prediction_proba = model.predict_proba(features)[0]

                return {
                    'eligible': bool(prediction),
                    'confidence': float(max(prediction_proba)),
                    'model_type': 'XGBoost'
                }
            else:
                # Fallback to Random Forest
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                X_train = training_data[['total_credits', 'total_debits', 'num_transactions',
                                       'avg_transaction_amount', 'transaction_variability', 'balance_trend']]
                y_train = training_data['Eligibility (y)']
                model.fit(X_train, y_train)

                features_basic = features[['total_credits', 'total_debits', 'num_transactions',
                                         'avg_transaction_amount', 'transaction_variability', 'balance_trend']]
                prediction = model.predict(features_basic)[0]
                prediction_proba = model.predict_proba(features_basic)[0]

                return {
                    'eligible': bool(prediction),
                    'confidence': float(max(prediction_proba)),
                    'model_type': 'Random Forest'
                }

        except Exception as e:
            st.error(f"Error in loan prediction: {str(e)}")
            return {'eligible': False, 'confidence': 0.0, 'model_type': 'Error'}

def parse_pdf_enhanced(file):
    """Enhanced PDF parsing with better text extraction"""
    try:
        with pdfplumber.open(file) as pdf:
            text = ''
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + '\n'
        return text
    except Exception as e:
        st.error(f"Error parsing PDF: {str(e)}")
        return ""

def process_text_to_df_enhanced(text):
    """Enhanced text processing with better pattern recognition"""
    transactions = []

    # Multiple transaction patterns for different formats
    patterns = [
        # Pattern 1: Standard ABSA format
        re.compile(r'(\d{4}-\d{2}-\d{2})\s+(.+?)\s+(-?R?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s+(-?R?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'),
        # Pattern 2: Alternative date format
        re.compile(r'(\d{2}/\d{2}/\d{4})\s+(.+?)\s+(-?R?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s+(-?R?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'),
        # Pattern 3: No currency symbol
        re.compile(r'(\d{4}-\d{2}-\d{2})\s+(.+?)\s+(-?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s+(-?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)')
    ]

    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue

        for pattern in patterns:
            match = pattern.search(line)
            if match:
                try:
                    date_str, description, amount_str, balance_str = match.groups()

                    # Convert date format if needed
                    if '/' in date_str:
                        date_obj = datetime.strptime(date_str, '%d/%m/%Y')
                        date_str = date_obj.strftime('%Y-%m-%d')

                    # Clean and convert amounts
                    amount = float(re.sub(r'[R,\s]', '', amount_str))
                    balance = float(re.sub(r'[R,\s]', '', balance_str))

                    transactions.append([date_str, description.strip(), amount, balance])
                    break
                except (ValueError, AttributeError):
                    continue

    if not transactions:
        return pd.DataFrame(columns=['Date', 'Description', 'Amount', 'Balance'])

    df = pd.DataFrame(transactions, columns=['Date', 'Description', 'Amount', 'Balance'])
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    return df

def categorize_expense_enhanced(description):
    """Enhanced expense categorization using NLP"""
    description_lower = description.lower()

    # Enhanced categorization with more sophisticated rules
    category_keywords = {
        'Salary/Income': ['salary', 'wage', 'income', 'payroll', 'refund'],
        'Groceries': ['grocery', 'supermarket', 'food', 'spar', 'checkers', 'woolworths'],
        'Transport': ['fuel', 'petrol', 'uber', 'taxi', 'transport', 'car payment'],
        'Utilities': ['electricity', 'water', 'municipal', 'rates', 'internet', 'phone'],
        'Entertainment': ['restaurant', 'movie', 'entertainment', 'netflix', 'spotify'],
        'Healthcare': ['medical', 'doctor', 'hospital', 'pharmacy', 'health'],
        'Shopping': ['retail', 'clothing', 'amazon', 'takealot', 'mall'],
        'Investments': ['investment', 'shares', 'unit trust', 'retirement'],
        'Insurance': ['insurance', 'medical aid', 'life cover', 'short term'],
        'POS Purchases': ['cashsend mobile', 'pos purchase'],
        'Payments': ['immediate trf', 'digital payment', 'payment'],
        'Credits': ['acb credit', 'immediate trf cr', 'credit'],
        'Bank Charges': ['fees', 'charge', 'commission'],
        'Cash Transactions': ['atm', 'cash deposit', 'withdrawal'],
        'Cellular': ['airtime', 'data', 'vodacom', 'mtn', 'cell c'],
        'Interest': ['interest'],
        'Failed Transactions': ['unsuccessful', 'declined', 'failed']
    }

    # Check for specific keywords
    for category, keywords in category_keywords.items():
        if any(keyword in description_lower for keyword in keywords):
            return category

    # Try NLP-based categorization if available
    try:
        if 'TextBlob' in globals():
            blob = TextBlob(description)
            # Simple sentiment and keyword analysis could be added here
            pass
    except:
        pass

    return 'Others'

def create_advanced_visualizations(df, patterns, recommendations):
    """Create advanced interactive visualizations"""

    # 1. Financial Health Dashboard
    col1, col2, col3, col4 = st.columns(4)

    total_income = df[df['Amount'] > 0]['Amount'].sum()
    total_expenses = abs(df[df['Amount'] < 0]['Amount'].sum())
    net_flow = total_income - total_expenses
    transaction_count = len(df)

    with col1:
        st.metric("Total Income", f"R{total_income:,.2f}", delta=None)
    with col2:
        st.metric("Total Expenses", f"R{total_expenses:,.2f}", delta=None)
    with col3:
        st.metric("Net Cash Flow", f"R{net_flow:,.2f}",
                 delta=f"{'Positive' if net_flow > 0 else 'Negative'}")
    with col4:
        st.metric("Transactions", f"{transaction_count}", delta=None)

    # 2. Enhanced spending analysis
    fig_treemap = px.treemap(
        df.groupby('Category')['Amount'].sum().abs().reset_index(),
        path=['Category'],
        values='Amount',
        title='Spending Distribution by Category (Treemap)'
    )
    st.plotly_chart(fig_treemap, use_container_width=True)

    # 3. Time series analysis with predictions
    daily_spending = df.groupby(df['Date'].dt.date)['Amount'].sum().reset_index()
    daily_spending['Date'] = pd.to_datetime(daily_spending['Date'])

    fig_timeseries = go.Figure()
    fig_timeseries.add_trace(go.Scatter(
        x=daily_spending['Date'],
        y=daily_spending['Amount'],
        mode='lines+markers',
        name='Actual Spending',
        line=dict(color='blue')
    ))

    # Add trend line
    x_numeric = np.arange(len(daily_spending))
    z = np.polyfit(x_numeric, daily_spending['Amount'], 1)
    p = np.poly1d(z)
    fig_timeseries.add_trace(go.Scatter(
        x=daily_spending['Date'],
        y=p(x_numeric),
        mode='lines',
        name='Trend',
        line=dict(color='red', dash='dash')
    ))

    fig_timeseries.update_layout(title='Daily Spending Trend with Projection')
    st.plotly_chart(fig_timeseries, use_container_width=True)

    # 4. Category-wise monthly analysis
    df['Month'] = df['Date'].dt.to_period('M')
    monthly_category = df.groupby(['Month', 'Category'])['Amount'].sum().abs().reset_index()
    monthly_category['Month'] = monthly_category['Month'].astype(str)

    fig_monthly = px.bar(
        monthly_category,
        x='Month',
        y='Amount',
        color='Category',
        title='Monthly Spending by Category'
    )
    st.plotly_chart(fig_monthly, use_container_width=True)

def main():
    """Main application function"""

    # Initialize database
    db = DatabaseManager()

    # Sidebar for user authentication
    with st.sidebar:
        st.title("🏦 Financial Analysis")

        if st.session_state.user_profile is None:
            tab1, tab2 = st.tabs(["Login", "Sign Up"])

            with tab1:
                st.subheader("Login")
                email = st.text_input("Email", key="login_email")
                password = st.text_input("Password", type="password", key="login_password")

                if st.button("Login", key="login_btn"):
                    user_data = db.authenticate_user(email, password)
                    if user_data:
                        user_id, name = user_data
                        st.session_state.user_profile = db.get_user_profile(user_id)
                        st.success(f"Welcome back, {name}!")
                        st.rerun()
                    else:
                        st.error("Invalid credentials")

            with tab2:
                st.subheader("Create Account")
                new_name = st.text_input("Full Name", key="signup_name")
                new_email = st.text_input("Email", key="signup_email")
                new_password = st.text_input("Password", type="password", key="signup_password")
                monthly_income = st.number_input("Monthly Income (R)", min_value=0.0, key="signup_income")
                risk_tolerance = st.selectbox("Risk Tolerance", ["conservative", "moderate", "aggressive"], key="signup_risk")
                financial_goals = st.text_area("Financial Goals", key="signup_goals")

                if st.button("Create Account", key="signup_btn"):
                    if new_name and new_email and new_password:
                        user_id = hashlib.md5(new_email.encode()).hexdigest()[:8]
                        if db.create_user(user_id, new_name, new_email, new_password, financial_goals, risk_tolerance, monthly_income):
                            st.success("Account created successfully! Please login.")
                        else:
                            st.error("Email already exists")
                    else:
                        st.error("Please fill all required fields")
                        st.rerun()

        else:
            st.success(f"Welcome, {st.session_state.user_profile['name']}!")
            if st.button("Logout"):
                st.session_state.user_profile = None
                st.session_state.transactions_df = None
                st.session_state.analysis_complete = False
                st.experimental_rerun()

    # Main content area
    if st.session_state.user_profile is None:
        st.markdown("""
        # 🏦 Enhanced Bank Statement Analysis

        ### Welcome to the next generation of financial analysis!

        **Key Features:**
        - 🤖 **AI-Powered Insights**: Advanced machine learning for personalized recommendations
        - 📊 **Comprehensive Analytics**: Deep dive into your spending patterns
        - 🎯 **Goal Tracking**: Set and monitor your financial objectives
        - 🔮 **Predictive Analysis**: Forecast future spending trends
        - 🛡️ **Anomaly Detection**: Identify unusual transactions
        - 💡 **Smart Recommendations**: Personalized financial advice

        **Please login or create an account to get started.**
        """)

        return

    # Main analysis interface
    st.title(f"🏦 Financial Analysis Dashboard - {st.session_state.user_profile['name']}")

    # File upload section
    st.markdown("### 📄 Upload Your Bank Statement")
    uploaded_file = st.file_uploader(
        "Choose a PDF bank statement",
        type="pdf",
        help="Upload your ABSA bank statement in PDF format for analysis"
    )

    if uploaded_file is not None:
        try:
            # Parse PDF
            with st.spinner("🔍 Parsing bank statement..."):
                text = parse_pdf_enhanced(uploaded_file)
                df = process_text_to_df_enhanced(text)

            if df.empty:
                st.warning("⚠️ No transactions found in the uploaded statement. Please check the file format.")
                return

            # Store in session state
            st.session_state.transactions_df = df

            # Enhance data with categories
            df['Category'] = df['Description'].apply(categorize_expense_enhanced)

            # Initialize analytics engines
            personalization = PersonalizationEngine(st.session_state.user_profile)
            analytics = AdvancedAnalytics()

            # Perform analysis
            with st.spinner("🧠 Analyzing your financial data..."):
                patterns = personalization.analyze_spending_patterns(df)
                recommendations = personalization.generate_personalized_recommendations(df, patterns)
                health_score, score_components = personalization.calculate_financial_health_score(df, patterns)
                loan_prediction = analytics.enhanced_loan_prediction(df)
                df_with_anomalies = analytics.detect_anomalies(df)

            st.session_state.analysis_complete = True

            # Display results in tabs
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "📊 Overview", "💡 Recommendations", "🏥 Health Score",
                "🔍 Detailed Analysis", "🚨 Anomalies", "💰 Loan Eligibility"
            ])

            with tab1:
                st.markdown("### 📈 Financial Overview")
                create_advanced_visualizations(df, patterns, recommendations)

                # Transaction table with enhanced features
                st.markdown("### 📋 Transaction History")
                st.dataframe(
                    df[['Date', 'Description', 'Category', 'Amount', 'Balance']],
                    use_container_width=True
                )

            with tab2:
                st.markdown("### 💡 Personalized Recommendations")

                if recommendations:
                    for i, rec in enumerate(recommendations):
                        priority_color = {
                            'high': '🔴',
                            'medium': '🟡',
                            'low': '🟢'
                        }.get(rec['priority'], '⚪')

                        st.markdown(f"""
                        **{priority_color} {rec['title']}**

                        {rec['description']}

                        *Category: {rec['type'].title()} | Priority: {rec['priority'].title()}*
                        """)
                        st.divider()
                else:
                    st.info("💫 Great job! Your financial habits look healthy. Keep up the good work!")

            with tab3:
                st.markdown("### 🏥 Financial Health Score")

                # Display overall score
                col1, col2 = st.columns([1, 2])

                with col1:
                    # Create gauge chart for health score
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = health_score,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Financial Health Score"},
                        delta = {'reference': 75},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 25], 'color': "lightgray"},
                                {'range': [25, 50], 'color': "gray"},
                                {'range': [50, 75], 'color': "lightgreen"},
                                {'range': [75, 100], 'color': "green"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        }
                    ))
                    fig_gauge.update_layout(height=300)
                    st.plotly_chart(fig_gauge, use_container_width=True)

                with col2:
                    st.markdown("#### Score Breakdown")
                    for component, score in score_components.items():
                        component_name = component.replace('_', ' ').title()
                        st.progress(score/25, text=f"{component_name}: {score:.1f}/25")

                # Health recommendations
                if health_score < 60:
                    st.warning("⚠️ Your financial health needs attention. Consider the recommendations above.")
                elif health_score < 80:
                    st.info("💡 Good financial health! A few improvements could boost your score.")
                else:
                    st.success("🎉 Excellent financial health! You're doing great!")

            with tab4:
                st.markdown("### 🔍 Detailed Financial Analysis")

                # Advanced metrics
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### Spending Patterns")
                    weekend_avg = patterns.get('weekend_vs_weekday', {}).get('weekend_avg', 0)
                    weekday_avg = patterns.get('weekend_vs_weekday', {}).get('weekday_avg', 0)

                    st.write(f"Weekend Average: R{weekend_avg:.2f}")
                    st.write(f"Weekday Average: R{weekday_avg:.2f}")
                    st.write(f"Transaction Frequency: {patterns.get('transaction_frequency', 0):.2f} per day")

                with col2:
                    st.markdown("#### Monthly Trends")
                    monthly_trends = patterns.get('monthly_trends', {})
                    for month, avg_spending in monthly_trends.items():
                        month_name = pd.to_datetime(f"2023-{month:02d}-01").strftime("%B")
                        st.write(f"{month_name}: R{avg_spending:.2f}")

                # Category analysis
                st.markdown("#### Category Analysis")
                category_data = []
                for category, data in patterns.get('spending_by_category', {}).items():
                    if isinstance(data, dict):
                        category_data.append({
                            'Category': category,
                            'Total': data.get('sum', 0),
                            'Average': data.get('mean', 0),
                            'Count': data.get('count', 0),
                            'Std Dev': data.get('std', 0)
                        })

                if category_data:
                    category_df = pd.DataFrame(category_data)
                    st.dataframe(category_df, use_container_width=True)

            with tab5:
                st.markdown("### 🚨 Anomaly Detection")

                anomalies = df_with_anomalies[df_with_anomalies['IsAnomaly']]

                if not anomalies.empty:
                    st.warning(f"⚠️ Found {len(anomalies)} unusual transactions:")
                    st.dataframe(
                        anomalies[['Date', 'Description', 'Amount', 'Category']],
                        use_container_width=True
                    )

                    # Visualization
                    fig_anomaly = px.scatter(
                        df_with_anomalies,
                        x='Date',
                        y='Amount',
                        color='IsAnomaly',
                        title='Transaction Anomalies',
                        color_discrete_map={True: 'red', False: 'blue'}
                    )
                    st.plotly_chart(fig_anomaly, use_container_width=True)
                else:
                    st.success("✅ No unusual transactions detected. Your spending patterns look normal!")

            with tab6:
                st.markdown("### 💰 Loan Eligibility Assessment")

                # Display loan prediction results
                if loan_prediction['eligible']:
                    st.success(f"✅ **Congratulations!** You are eligible for a loan.")
                else:
                    st.error(f"❌ **Unfortunately,** you are not currently eligible for a loan.")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Confidence Score", f"{loan_prediction['confidence']:.1%}")
                with col2:
                    st.metric("Model Used", loan_prediction['model_type'])

                # Detailed loan analysis
                st.markdown("#### Loan Assessment Factors")

                total_credits = df[df['Amount'] > 0]['Amount'].sum()
                total_debits = abs(df[df['Amount'] < 0]['Amount'].sum())
                debt_to_income = total_debits / total_credits if total_credits > 0 else float('inf')

                factors = {
                    "Total Income": f"R{total_credits:,.2f}",
                    "Total Expenses": f"R{total_debits:,.2f}",
                    "Debt-to-Income Ratio": f"{debt_to_income:.2%}",
                    "Net Cash Flow": f"R{total_credits - total_debits:,.2f}",
                    "Transaction Count": str(len(df)),
                    "Account Balance Trend": f"R{patterns.get('balance_trend', 0):,.2f}"
                }

                for factor, value in factors.items():
                    st.write(f"**{factor}:** {value}")

                # Improvement suggestions for loan eligibility
                if not loan_prediction['eligible']:
                    st.markdown("#### 💡 How to Improve Your Loan Eligibility")
                    st.markdown("""
                    - **Increase Income**: Look for ways to boost your monthly income
                    - **Reduce Expenses**: Cut down on non-essential spending
                    - **Build Savings**: Maintain a higher account balance
                    - **Regular Transactions**: Show consistent financial activity
                    - **Improve Cash Flow**: Ensure more money comes in than goes out
                    """)

        except Exception as e:
            st.error(f"❌ An error occurred while processing your statement: {str(e)}")
            st.info("Please ensure your PDF is a valid ABSA bank statement and try again.")

if __name__ == "__main__":
    main()

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }

    .recommendation-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #28a745;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }

    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)
