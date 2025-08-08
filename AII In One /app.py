 
import streamlit as st
import pdfplumber
import re
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import hashlib
import tempfile
import PyPDF2
import os
import webbrowser
import threading
import uuid
import subprocess
import time
import requests
from http.server import HTTPServer, SimpleHTTPRequestHandler
from dateutil.relativedelta import relativedelta

# Mistral API Configuration
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "bPj0wARXs5dk2L1ipFOdoqHMmQnXuMNv")

# Advanced ML and NLP imports
try:
    import xgboost as xgb
    from sklearn.ensemble import RandomForestClassifier, IsolationForest
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.cluster import KMeans
    import nltk
    from textblob import TextBlob
except ImportError as e:
    st.warning(f"Some advanced features may not be available. Missing: {str(e)}")

# Configuration
st.set_page_config(
    page_title='Universal Bank Statement Analyzer',
    page_icon='🏦',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Custom CSS for improved UI
st.markdown('''
<style>
    /* Main theme colors */
    :root {
        --primary-color: #4F8BF9;
        --secondary-color: #1E88E5;
        --background-color: #FAFAFA;
        --text-color: #212121;
        --light-text-color: #757575;
        --accent-color: #FF4081;
        --success-color: #4CAF50;
        --warning-color: #FFC107;
        --error-color: #F44336;
    }
    
    /* Main container styling */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    /* Card-like elements */
    div[data-testid="stExpander"] {
        border: 1px solid #E0E0E0;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    
    /* Headings */
    h1, h2, h3 {
        color: var(--primary-color);
        font-weight: 600;
    }
    
    /* Metrics styling */
    div[data-testid="stMetric"] {
        background-color: white;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 20px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
        background-color: #F5F7FA;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 4px 4px 0 0;
        padding: 8px 16px;
        font-weight: 500;
    }
    
    /* Dataframe styling */
    .stDataFrame {        
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    /* File uploader */
    .stFileUploader > div > div {
        border-radius: 8px;
        border: 2px dashed #E0E0E0;
        padding: 1.5rem;
    }
    
    /* Progress bar */
    div[data-testid="stProgressBar"] > div {
        border-radius: 10px;
        height: 10px;
    }
    
    /* Alerts */
    div[data-testid="stAlert"] {
        border-radius: 8px;
    }
</style>
''', unsafe_allow_html=True)

# Initialize session state
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = None
if 'transactions_df' not in st.session_state:
    st.session_state.transactions_df = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'detected_bank' not in st.session_state:
    st.session_state.detected_bank = None
if 'mistral_insights' not in st.session_state:
    st.session_state.mistral_insights = None
if 'ai_assistant_active' not in st.session_state:
    st.session_state.ai_assistant_active = False
if 'budgets' not in st.session_state:
    st.session_state.budgets = {}
if 'budget_forecast' not in st.session_state:
    st.session_state.budget_forecast = None


class DatabaseManager:
    """Handles all database operations for user profiles and financial data"""

    def __init__(self):
        self.db_path = 'financial_analysis.db'
        self.init_database()

    def init_database(self):
        """Initialize database tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Test database integrity first
            cursor.execute("PRAGMA integrity_check;")
            integrity_result = cursor.fetchone()

            if integrity_result[0] != "ok":
                conn.close()
                raise sqlite3.DatabaseError("Database integrity check failed")

        except sqlite3.DatabaseError as e:
            # Handle corrupted database by removing it and creating fresh
            import os
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
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
                bank_name TEXT,
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
                bank_name TEXT,
                recommendation_type TEXT,
                title TEXT,
                description TEXT,
                priority INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES user_profiles (user_id)
            )
        ''')

        # Budgets table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS budgets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                category TEXT NOT NULL,
                budget_amount REAL NOT NULL,
                month_year TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES user_profiles (user_id)
            )
        ''')

        conn.commit()
        conn.close()

    def create_user(self, user_id, name, email, password, financial_goals="", risk_tolerance="moderate",
                    monthly_income=0):
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
            columns = ['id', 'user_id', 'name', 'email', 'password_hash', 'financial_goals',
                       'risk_tolerance', 'monthly_income', 'created_at']
            return dict(zip(columns, result))
        return None

    # ADDED THE MISSING BUDGET METHODS
    def save_budgets(self, user_id, budgets, month_year):
        """Save budgets to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Delete existing budgets for this month
        cursor.execute('DELETE FROM budgets WHERE user_id = ? AND month_year = ?', (user_id, month_year))

        # Insert new budgets
        for category, amount in budgets.items():
            cursor.execute('''
                INSERT INTO budgets (user_id, category, budget_amount, month_year)
                VALUES (?, ?, ?, ?)
            ''', (user_id, category, amount, month_year))

        conn.commit()
        conn.close()
        return True

    def get_budgets(self, user_id, month_year):
        """Get budgets for a specific month"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT category, budget_amount FROM budgets WHERE user_id = ? AND month_year = ?',
                       (user_id, month_year))
        results = cursor.fetchall()
        conn.close()

        return {row[0]: row[1] for row in results} if results else {}


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
        df['Hour'] = df['Date'].dt.hour  # For hourly analysis

        patterns = {
            'spending_by_category': spending_by_category.to_dict(),
            'weekend_vs_weekday': {
                'weekend_avg': df[df['IsWeekend']]['Amount'].mean(),
                'weekday_avg': df[~df['IsWeekend']]['Amount'].mean()
            },
            'monthly_trends': df.groupby('Month')['Amount'].mean().to_dict(),
            'transaction_frequency': len(df) / max(1, (df['Date'].max() - df['Date'].min()).days),
            'balance_trend': df['Balance'].iloc[-1] - df['Balance'].iloc[0] if len(df) > 1 else 0,
            'hourly_trends': df.groupby('Hour')['Amount'].mean().to_dict()
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
            spending_consistency = 1 - (
                debits['Amount'].std() / abs(debits['Amount'].mean()) if debits['Amount'].mean() != 0 else 1)
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
            # Return original dataframe with IsAnomaly column set to False
            df_copy = df.copy()
            df_copy['IsAnomaly'] = False
            return df_copy

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

        future_dates = [df_daily['Date'].max() + timedelta(days=i + 1) for i in range(days_ahead)]

        return pd.DataFrame({
            'Date': future_dates,
            'Predicted_Amount': future_predictions
        })

    def enhanced_loan_prediction(self, df, bank_name):
        """Enhanced loan eligibility prediction with bank-specific models"""
        try:
            # Dispatch to bank-specific methods
            if bank_name == 'ABSA':
                return self.enhanced_loan_prediction_absa(df)
            elif bank_name == 'Nedbank':
                return self.enhanced_loan_prediction_nedbank(df)
            elif bank_name == 'FNB':
                return self.enhanced_loan_prediction_fnb(df)
            elif bank_name == 'Standard Bank':
                return self.enhanced_loan_prediction_standardbank(df)
            else:
                return self.enhanced_loan_prediction_default(df, bank_name)

        except Exception as e:
            st.error(f"Error in loan prediction: {str(e)}")
            return {'eligible': False, 'confidence': 0.0, 'model_type': 'Error'}

    def enhanced_loan_prediction_absa(self, df):
        """Enhanced loan eligibility prediction for ABSA Bank"""
        try:
            # Load training data
            training_data = pd.read_csv('bank_data/absa.csv')

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

            # Ensure all required columns exist in training data
            for col in features.columns:
                if col not in training_data.columns:
                    training_data[col] = 0.0

            # Check for eligibility column with possible names
            eligibility_col = None
            for col in ['Eligibility', 'Eligibility (y)', 'Loan_Eligibility']:
                if col in training_data.columns:
                    eligibility_col = col
                    break

            if not eligibility_col:
                st.error("Missing eligibility column in ABSA dataset")
                return {'eligible': False, 'confidence': 0.0, 'model_type': 'Error'}

            # Handle string labels
            le = LabelEncoder()
            y_train = le.fit_transform(training_data[eligibility_col])

            # Train XGBoost model if available
            if 'xgb' in globals():
                X_train = training_data[features.columns]

                if len(X_train) == 0 or len(y_train) == 0:
                    raise ValueError("Training data is empty")

                model = xgb.XGBClassifier(random_state=42)
                model.fit(X_train, y_train)

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
                X_train = training_data[features.columns]

                if len(X_train) == 0 or len(y_train) == 0:
                    raise ValueError("Training data is empty")

                model.fit(X_train, y_train)

                prediction = model.predict(features)[0]
                prediction_proba = model.predict_proba(features)[0]

                return {
                    'eligible': bool(prediction),
                    'confidence': float(max(prediction_proba)),
                    'model_type': 'Random Forest'
                }

        except Exception as e:
            st.error(f"Error in ABSA loan prediction: {str(e)}")
            return {'eligible': False, 'confidence': 0.0, 'model_type': 'Error'}

    def enhanced_loan_prediction_nedbank(self, df):
        """Enhanced loan eligibility prediction for Nedbank"""
        try:
            # Load training data
            training_data = pd.read_csv('bank_data/nedbank.csv')

            # Prepare features from the transaction data
            total_credits = df[df['Amount'] > 0]['Amount'].sum()
            total_debits = abs(df[df['Amount'] < 0]['Amount'].sum())
            num_transactions = len(df)
            avg_balance = df['Balance'].mean()
            closing_balance = df['Balance'].iloc[-1] if len(df) > 0 else 0

            # Prepare feature vector
            features = pd.DataFrame({
                'Closing Balance': [closing_balance],
                'Total Credit': [total_credits],
                'Average Balance': [avg_balance],
                'Number of Transactions': [num_transactions],
                'Number of Debits': [len(df[df['Amount'] < 0])],
                'Number of Credits': [len(df[df['Amount'] > 0])]
            })

            # Ensure all required columns exist in training data
            for col in features.columns:
                if col not in training_data.columns:
                    training_data[col] = 0.0

            # Check for eligibility column with possible names
            eligibility_col = None
            for col in ['Eligibility', 'Eligibility (y)', 'Loan_Eligibility']:
                if col in training_data.columns:
                    eligibility_col = col
                    break

            if not eligibility_col:
                st.error("Missing eligibility column in Nedbank dataset")
                return {'eligible': False, 'confidence': 0.0, 'model_type': 'Error'}

            # Handle string labels
            le = LabelEncoder()
            y_train = le.fit_transform(training_data[eligibility_col])

            # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            X_train = training_data[features.columns]

            if len(X_train) == 0 or len(y_train) == 0:
                raise ValueError("Training data is empty")

            model.fit(X_train, y_train)

            prediction = model.predict(features)[0]
            prediction_proba = model.predict_proba(features)[0]

            return {
                'eligible': bool(prediction),
                'confidence': float(max(prediction_proba)),
                'model_type': 'Random Forest'
            }

        except Exception as e:
            st.error(f"Error in Nedbank loan prediction: {str(e)}")
            return {'eligible': False, 'confidence': 0.0, 'model_type': 'Error'}

    def enhanced_loan_prediction_fnb(self, df):
        """Enhanced loan eligibility prediction for FNB"""
        try:
            # Load training data
            training_data = pd.read_csv('bank_data/fnb.csv')

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

            # Ensure all required columns exist in training data
            for col in features.columns:
                if col not in training_data.columns:
                    training_data[col] = 0.0

            # Check for eligibility column with possible names
            eligibility_col = None
            for col in ['Eligibility', 'Eligibility (y)', 'Loan_Eligibility']:
                if col in training_data.columns:
                    eligibility_col = col
                    break

            if not eligibility_col:
                st.error("Missing eligibility column in FNB dataset")
                return {'eligible': False, 'confidence': 0.0, 'model_type': 'Error'}

            # Handle string labels
            le = LabelEncoder()
            y_train = le.fit_transform(training_data[eligibility_col])

            # Train XGBoost model if available
            if 'xgb' in globals():
                X_train = training_data[features.columns]

                if len(X_train) == 0 or len(y_train) == 0:
                    raise ValueError("Training data is empty")

                model = xgb.XGBClassifier(random_state=42)
                model.fit(X_train, y_train)

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
                X_train = training_data[features.columns]

                if len(X_train) == 0 or len(y_train) == 0:
                    raise ValueError("Training data is empty")

                model.fit(X_train, y_train)

                prediction = model.predict(features)[0]
                prediction_proba = model.predict_proba(features)[0]

                return {
                    'eligible': bool(prediction),
                    'confidence': float(max(prediction_proba)),
                    'model_type': 'Random Forest'
                }

        except Exception as e:
            st.error(f"Error in FNB loan prediction: {str(e)}")
            return {'eligible': False, 'confidence': 0.0, 'model_type': 'Error'}

    def enhanced_loan_prediction_standardbank(self, df):
        """Enhanced loan eligibility prediction for Standard Bank"""
        try:
            # Load training data
            training_data = pd.read_csv('bank_data/standardbank.csv')

            # Prepare features from the transaction data
            total_credits = df[df['Amount'] > 0]['Amount'].sum()
            total_debits = abs(df[df['Amount'] < 0]['Amount'].sum())
            num_transactions = len(df)
            avg_balance = df['Balance'].mean()
            closing_balance = df['Balance'].iloc[-1] if len(df) > 0 else 0

            # Prepare feature vector
            features = pd.DataFrame({
                'Total_Credits': [total_credits],
                'Total_Debits': [total_debits],
                'Average_Balance': [avg_balance],
                'Num_Transactions': [num_transactions]
            })

            # Ensure all required columns exist in training data
            for col in features.columns:
                if col not in training_data.columns:
                    training_data[col] = 0.0

            # Check for eligibility column with possible names
            eligibility_col = None
            for col in ['Eligibility', 'Eligibility (y)', 'Loan_Eligibility']:
                if col in training_data.columns:
                    eligibility_col = col
                    break

            if not eligibility_col:
                st.error("Missing eligibility column in Standard Bank dataset")
                return {'eligible': False, 'confidence': 0.0, 'model_type': 'Error'}

            # Handle string labels
            le = LabelEncoder()
            y_train = le.fit_transform(training_data[eligibility_col])

            # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            X_train = training_data[features.columns]

            if len(X_train) == 0 or len(y_train) == 0:
                raise ValueError("Training data is empty")

            model.fit(X_train, y_train)

            prediction = model.predict(features)[0]
            prediction_proba = model.predict_proba(features)[0]

            return {
                'eligible': bool(prediction),
                'confidence': float(max(prediction_proba)),
                'model_type': 'Random Forest'
            }

        except Exception as e:
            st.error(f"Error in Standard Bank loan prediction: {str(e)}")
            return {'eligible': False, 'confidence': 0.0, 'model_type': 'Error'}

    def enhanced_loan_prediction_default(self, df, bank_name):
        """Default loan eligibility prediction for other banks"""
        try:
            # Load training data
            training_data = pd.read_csv('bank_data/default.csv')

            # Prepare features from the transaction data
            total_credits = df[df['Amount'] > 0]['Amount'].sum()
            total_debits = abs(df[df['Amount'] < 0]['Amount'].sum())
            num_transactions = len(df)
            avg_balance = df['Balance'].mean()
            closing_balance = df['Balance'].iloc[-1] if len(df) > 0 else 0

            # Prepare feature vector
            features = pd.DataFrame({
                'Total_Credits': [total_credits],
                'Total_Debits': [total_debits],
                'Average_Balance': [avg_balance],
                'Num_Transactions': [num_transactions],
                'Closing_Balance': [closing_balance]
            })

            # Ensure all required columns exist in training data
            for col in features.columns:
                if col not in training_data.columns:
                    training_data[col] = 0.0

            # Check for eligibility column with possible names
            eligibility_col = None
            for col in ['Eligibility', 'Eligibility (y)', 'Loan_Eligibility']:
                if col in training_data.columns:
                    eligibility_col = col
                    break

            if not eligibility_col:
                st.error("Missing eligibility column in default dataset")
                return {'eligible': False, 'confidence': 0.0, 'model_type': 'Error'}

            # Handle string labels
            le = LabelEncoder()
            y_train = le.fit_transform(training_data[eligibility_col])

            # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            X_train = training_data[features.columns]

            if len(X_train) == 0 or len(y_train) == 0:
                raise ValueError("Training data is empty")

            model.fit(X_train, y_train)

            prediction = model.predict(features)[0]
            prediction_proba = model.predict_proba(features)[0]

            return {
                'eligible': bool(prediction),
                'confidence': float(max(prediction_proba)),
                'model_type': 'Random Forest'
            }

        except Exception as e:
            st.error(f"Error in default loan prediction: {str(e)}")
            return {'eligible': False, 'confidence': 0.0, 'model_type': 'Error'}


def extract_text_from_pdf(file):
    """Extract text from PDF file"""
    try:
        if isinstance(file, str):
            with open(file, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ''
                for page in reader.pages:
                    text += page.extract_text()
        else:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page in reader.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""


def identify_bank_from_text(text):
    """Identify bank from statement text"""
    text_lower = text.lower()
    bank_keywords = {
        'FNB': ['fnb', 'first national bank'],
        'Standard Bank': ['standard bank'],
        'Nedbank': ['nedbank'],
        'ABSA': ['absa'],
        'Capitec Bank': ['capitec']
    }

    for bank, keywords in bank_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            return bank
    return None


def extract_bank_statement_metadata(text, bank_name):
    """Extract metadata from bank statement based on bank"""
    if bank_name == 'FNB':
        # Extract account holder name
        name_match = re.search(r"(MR|MRS)\s+([A-Z\s]+)", text)
        account_holder_name = name_match.group(0) if name_match else "Name not found"

        # Extract closing balance
        closing_balance_match = re.search(r"Closing Balance\s+([\d,]+\.?\d*)", text)
        closing_balance = float(closing_balance_match.group(1).replace(',', '')) if closing_balance_match else 0.0

        return account_holder_name, closing_balance

    elif bank_name == 'Standard Bank':
        name_match = re.search(r"Account Holder:\s*(.+?)\n", text)
        account_holder_name = name_match.group(1).strip() if name_match else "Name not found"

        acc_match = re.search(r"Account Number:\s*(\d+)", text)
        account_number = acc_match.group(1) if acc_match else "Not found"

        period_match = re.search(r"Statement Period:\s*(.+?)\n", text)
        statement_period = period_match.group(1).strip() if period_match else "Not specified"

        balance_match = re.search(r"Closing Balance:\s*(-?R?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)", text)
        closing_balance = float(balance_match.group(1).replace(',', '').replace('R', '')) if balance_match else 0.0

        return account_holder_name, account_number, statement_period, closing_balance

    elif bank_name == 'Nedbank':
        name_match = re.search(r"Account Holder:\s*(.+?)\n", text)
        account_holder_name = name_match.group(1).strip() if name_match else "Name not found"

        acc_match = re.search(r"Account Number:\s*(\d+)", text)
        account_number = acc_match.group(1) if acc_match else "Not found"

        period_match = re.search(r"Statement Period:\s*(.+?)\n", text)
        statement_period = period_match.group(1).strip() if period_match else "Not specified"

        balance_match = re.search(r"Closing Balance:\s*(-?R?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)", text)
        closing_balance = float(balance_match.group(1).replace(',', '').replace('R', '')) if balance_match else 0.0

        return account_holder_name, account_number, statement_period, closing_balance

    elif bank_name == 'ABSA':
        name_match = re.search(r"Account Holder:\s*(.+?)\n", text)
        account_holder_name = name_match.group(1).strip() if name_match else "Name not found"

        acc_match = re.search(r"Account Number:\s*(\d+)", text)
        account_number = acc_match.group(1) if acc_match else "Not found"

        period_match = re.search(r"Statement Period:\s*(.+?)\n", text)
        statement_period = period_match.group(1).strip() if period_match else "Not specified"

        balance_match = re.search(r"Closing Balance:\s*(-?R?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)", text)
        closing_balance = float(balance_match.group(1).replace(',', '').replace('R', '')) if balance_match else 0.0

        return account_holder_name, account_number, statement_period, closing_balance

    else:  # Default for other banks
        name_match = re.search(r"Account Holder:\s*(.+?)\n", text)
        account_holder_name = name_match.group(1).strip() if name_match else "Name not found"

        balance_match = re.search(r"Closing Balance:\s*(-?R?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)", text)
        closing_balance = float(balance_match.group(1).replace(',', '').replace('R', '')) if balance_match else 0.0

        return account_holder_name, closing_balance


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


def process_text_to_df_enhanced(text, bank_name):
    """Process text to DataFrame based on bank format"""
    transactions = []

    if bank_name == 'FNB':
        transaction_pattern = re.compile(
            r'(\d{2} \w{3})\s+'  # Date (e.g., "02 Apr")
            r'(.+?)\s+'  # Description
            r'(-?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)(Cr|Dr)?\s*'  # Amount
            r'(-?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)(Cr|Dr)?'  # Balance
        )

        for line in text.split('\n'):
            line = line.strip()
            if not line or 'Transactions in RAND' in line or 'Date Description' in line:
                continue

            match = transaction_pattern.search(line)
            if match:
                try:
                    groups = match.groups()
                    date_str, description, amount_str, cr_dr1, balance_str, _ = groups

                    # Convert date to standard format
                    current_year = datetime.now().year
                    date_obj = datetime.strptime(f"{date_str} {current_year}", "%d %b %Y")
                    date_str = date_obj.strftime("%Y-%m-%d")

                    # Clean and convert amounts
                    amount = float(amount_str.replace(',', ''))
                    if cr_dr1 == 'Cr':
                        amount = -amount  # Credits are negative in our system

                    balance = float(balance_str.replace(',', ''))

                    transactions.append([date_str, description.strip(), amount, balance])
                except (ValueError, AttributeError) as e:
                    st.warning(f"Error parsing FNB transaction: {str(e)}")
                    continue

    elif bank_name == 'Standard Bank':
        transaction_pattern = re.compile(
            r'(\d{2} \w{3} \d{2})\s+'  # Date (e.g., "02 Apr 22")
            r'(.+?)\s+'  # Description
            r'(-?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s+'  # Amount
            r'(-?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'  # Balance
        )

        for line in text.split('\n'):
            line = line.strip()
            if not line or 'Date Description' in line or 'Transactions in RAND' in line:
                continue

            match = transaction_pattern.search(line)
            if match:
                try:
                    date_str, description, amount_str, balance_str = match.groups()

                    # Convert date to standard format
                    date_obj = datetime.strptime(date_str, '%d %b %y')
                    date_str = date_obj.strftime('%Y-%m-%d')

                    # Clean and convert amounts
                    amount = float(amount_str.replace(',', ''))
                    balance = float(balance_str.replace(',', ''))

                    transactions.append([date_str, description.strip(), amount, balance])
                except (ValueError, AttributeError) as e:
                    st.warning(f"Error parsing Standard Bank transaction: {str(e)}")
                    continue

    elif bank_name == 'Nedbank':
        transaction_pattern = re.compile(
            r'(\d{2}/\d{2}/\d{4})\s+'  # Date (e.g., "02/04/2022")
            r'(.+?)\s+'  # Description
            r'(-?R?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s+'  # Amount
            r'(-?R?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'  # Balance
        )

        for line in text.split('\n'):
            line = line.strip()
            if not line or 'Date Description' in line or 'Transactions in RAND' in line:
                continue

            match = transaction_pattern.search(line)
            if match:
                try:
                    date_str, description, amount_str, balance_str = match.groups()

                    # Convert date to standard format (FIXED: use dayfirst format)
                    date_obj = datetime.strptime(date_str, '%d/%m/%Y')
                    date_str = date_obj.strftime('%Y-%m-%d')

                    # Clean and convert amounts
                    amount = float(amount_str.replace(',', '').replace('R', '').replace(' ', ''))
                    balance = float(balance_str.replace(',', '').replace('R', '').replace(' ', ''))

                    transactions.append([date_str, description.strip(), amount, balance])
                except (ValueError, AttributeError) as e:
                    st.warning(f"Error parsing Nedbank transaction: {str(e)}")
                    continue

    else:  # Default pattern for other banks
        transaction_pattern = re.compile(
            r'(\d{4}-\d{2}-\d{2})\s+'  # Date
            r'(.+?)\s+'  # Description
            r'(-?R?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s+'  # Amount
            r'(-?R?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'  # Balance
        )

        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue

            match = transaction_pattern.search(line)
            if match:
                try:
                    date_str, description, amount_str, balance_str = match.groups()

                    # Clean and convert amounts
                    amount = float(amount_str.replace(',', '').replace('R', '').replace(' ', ''))
                    balance = float(balance_str.replace(',', '').replace('R', '').replace(' ', ''))

                    transactions.append([date_str, description.strip(), amount, balance])
                except (ValueError, AttributeError) as e:
                    st.warning(f"Error parsing transaction: {str(e)}")
                    continue

    if not transactions:
        return pd.DataFrame(columns=['Date', 'Description', 'Amount', 'Balance'])

    df = pd.DataFrame(transactions, columns=['Date', 'Description', 'Amount', 'Balance'])

    # Convert to datetime with error handling
    try:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        # Drop rows with invalid dates
        df = df.dropna(subset=['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
    except Exception as e:
        st.error(f"Error converting dates: {str(e)}")
        return pd.DataFrame()

    df['Bank'] = bank_name  # Add bank name column

    return df


def categorize_expense_enhanced(description, bank_name):
    """Enhanced expense categorization with bank-specific rules"""
    description_lower = description.lower()

    # Common categories across all banks
    common_categories = {
        'Salary/Income': ['salary', 'wage', 'income', 'payroll', 'refund'],
        'Groceries': ['grocery', 'supermarket', 'food', 'spar', 'checkers', 'woolworths'],
        'Transport': ['fuel', 'petrol', 'uber', 'taxi', 'transport', 'car payment'],
        'Utilities': ['electricity', 'water', 'municipal', 'rates', 'internet', 'phone'],
        'Entertainment': ['restaurant', 'movie', 'entertainment', 'netflix', 'spotify'],
        'Healthcare': ['medical', 'doctor', 'hospital', 'pharmacy', 'health'],
        'Shopping': ['retail', 'clothing', 'amazon', 'takealot', 'mall'],
        'Investments': ['investment', 'shares', 'unit trust', 'retirement'],
        'Insurance': ['insurance', 'medical aid', 'life cover', 'short term'],
        'Bank Charges': ['fees', 'charge', 'service', 'cost', 'monthly account fee'],
        'Cash Transactions': ['atm', 'cash', 'withdrawal', 'deposit'],
        'Cellular': ['airtime', 'data', 'vodacom', 'mtn', 'cell c'],
        'Interest': ['interest'],
        'Failed Transactions': ['unsuccessful', 'declined', 'failed']
    }

    # Bank-specific categories
    bank_specific = {
        'FNB': {
            'POS Purchases': ['pos purchase', 'card purchase', 'debit order'],
            'Payments': ['payment to', 'fnb app rtc pmt', 'internet pmt', 'debit order']
        },
        'Standard Bank': {
            'Payments': ['payment', 'transfer', 'debit order', 'immediate trf', 'digital payment']
        },
        'Nedbank': {
            'POS Purchases': ['pos purchase', 'card purchase', 'debit order', 'cashsend'],
            'Payments': ['payment to', 'immediate trf', 'digital payment', 'pmt to'],
            'Loans': ['loan payment', 'nedloan', 'personal loan']
        },
        'ABSA': {
            'POS Purchases': ['cashsend mobile', 'pos purchase'],
            'Payments': ['immediate trf', 'digital payment', 'payment']
        }
    }

    # Combine common and bank-specific categories
    category_keywords = {**common_categories, **(bank_specific.get(bank_name, {}))}

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

    return 'Other'


def get_mistral_insights(df, user_profile, bank_name):
    """Get financial insights from Mistral API"""
    if df.empty:
        return "Not enough data to generate insights"

    # Prepare data summary for the AI
    total_income = df[df['Amount'] > 0]['Amount'].sum()
    total_expenses = abs(df[df['Amount'] < 0]['Amount'].sum())
    net_flow = total_income - total_expenses

    # Top spending categories
    spending_by_category = df[df['Amount'] < 0].groupby('Category')['Amount'].sum().abs().nlargest(5)
    top_categories = "\n".join([f"- {cat}: R{amt:.2f}" for cat, amt in spending_by_category.items()])

    # Time period
    start_date = df['Date'].min().strftime('%Y-%m-%d')
    end_date = df['Date'].max().strftime('%Y-%m-%d')

    # User profile info
    name = user_profile.get('name', 'User')
    income = user_profile.get('monthly_income', 0)
    goals = user_profile.get('financial_goals', 'Not specified')

    prompt = f"""
    You are a financial advisor analyzing a bank statement for {name} from {bank_name}.
    The statement covers the period from {start_date} to {end_date}.

    Financial Summary:
    - Total Income: R{total_income:.2f}
    - Total Expenses: R{total_expenses:.2f}
    - Net Cash Flow: R{net_flow:.2f}
    - Monthly Income: R{income:.2f}
    - Financial Goals: {goals}

    Top Spending Categories:
    {top_categories}

    Provide detailed financial insights and recommendations in markdown format. Include:
    1. Key observations about spending patterns
    2. Comparison of income vs expenses
    3. Specific recommendations for each spending category
    4. Suggestions to achieve financial goals
    5. Potential savings opportunities
    6. Investment advice based on cash flow

    Structure your response with clear headings and bullet points. Use emojis to make it engaging.
    """

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {MISTRAL_API_KEY}"
    }

    payload = {
        "model": "mistral-large-latest",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 1500
    }

    try:
        response = requests.post(MISTRAL_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data['choices'][0]['message']['content']
    except Exception as e:
        st.error(f"Error getting insights from Mistral API: {str(e)}")
        return "Could not generate insights at this time. Please try again later."


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
        title='Spending Distribution by Category (Treemap)',
        color_discrete_sequence=px.colors.sequential.RdBu_r
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

    # 4. Income vs Expenses Comparison
    st.markdown("### 💰 Income vs Expenses")
    col1, col2 = st.columns(2)

    with col1:
        # Income sources
        income_df = df[df['Amount'] > 0].groupby('Category')['Amount'].sum().reset_index()
        fig_income = px.pie(
            income_df,
            values='Amount',
            names='Category',
            title='Income Sources Distribution',
            hole=0.3
        )
        fig_income.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_income, use_container_width=True)

    with col2:
        # Expense breakdown
        expense_df = df[df['Amount'] < 0].copy()
        expense_df['Amount'] = expense_df['Amount'].abs()
        expense_df = expense_df.groupby('Category')['Amount'].sum().reset_index()
        fig_expense = px.pie(
            expense_df,
            values='Amount',
            names='Category',
            title='Expense Breakdown',
            hole=0.3
        )
        fig_expense.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_expense, use_container_width=True)

    # 5. Cumulative Cash Flow
    st.markdown("### 📈 Cumulative Cash Flow")
    df_sorted = df.sort_values('Date')
    df_sorted['Cumulative'] = df_sorted['Amount'].cumsum()

    fig_cumulative = px.area(
        df_sorted,
        x='Date',
        y='Cumulative',
        title='Cumulative Cash Flow Over Time',
        labels={'Cumulative': 'Balance (R)'}
    )
    fig_cumulative.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig_cumulative, use_container_width=True)

    # 6. Spending by Day of Week and Time
    st.markdown("### 📅 Spending Patterns by Time")
    col1, col2 = st.columns(2)

    with col1:
        # Day of week analysis
        df['DayOfWeek'] = df['Date'].dt.day_name()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_spending = df[df['Amount'] < 0].groupby('DayOfWeek')['Amount'].sum().abs().reindex(day_order).reset_index()

        fig_day = px.bar(
            day_spending,
            x='DayOfWeek',
            y='Amount',
            title='Total Spending by Day of Week',
            labels={'Amount': 'Total Spending (R)'},
            color='Amount',
            color_continuous_scale='Bluered'
        )
        st.plotly_chart(fig_day, use_container_width=True)

    with col2:
        # Hourly spending (if data available)
        if 'Hour' in df.columns:
            hour_spending = df[df['Amount'] < 0].groupby('Hour')['Amount'].sum().abs().reset_index()

            fig_hour = px.bar(
                hour_spending,
                x='Hour',
                y='Amount',
                title='Total Spending by Hour of Day',
                labels={'Amount': 'Total Spending (R)'},
                color='Amount',
                color_continuous_scale='Tealgrn'
            )
            st.plotly_chart(fig_hour, use_container_width=True)
        else:
            st.info("Hour data not available for spending patterns")

    # 7. Financial Health Radar Chart
    st.markdown("### 📊 Financial Health Overview")
    if 'score_components' in patterns:
        health_metrics = patterns['score_components']
        categories = list(health_metrics.keys())
        values = list(health_metrics.values())

        # Close the radar chart
        categories.append(categories[0])
        values.append(values[0])

        fig_radar = go.Figure(
            data=go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='Financial Health'
            )
        )

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 25]  # Max 25 per component
                )),
            showlegend=False,
            title='Financial Health Score Breakdown'
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # 8. Interactive Category Drill-Down
    st.markdown("### 🔍 Category Spending Analysis")
    df_expenses = df[df['Amount'] < 0].copy()
    df_expenses['Amount'] = df_expenses['Amount'].abs()

    # Select category to drill down
    categories = df_expenses['Category'].unique().tolist()
    selected_category = st.selectbox("Select Category to Analyze", categories)

    if selected_category:
        category_df = df_expenses[df_expenses['Category'] == selected_category]

        # Top merchants in category
        top_merchants = category_df.groupby('Description')['Amount'].sum().nlargest(5).reset_index()

        col1, col2 = st.columns(2)
        with col1:
            fig_merchants = px.bar(
                top_merchants,
                x='Description',
                y='Amount',
                title=f'Top Merchants in {selected_category}',
                labels={'Amount': 'Total Spending (R)'},
                color='Amount',
                color_continuous_scale='Magenta'
            )
            st.plotly_chart(fig_merchants, use_container_width=True)

        with col2:
            # Monthly trend for category
            category_df['Month'] = category_df['Date'].dt.to_period('M')
            monthly_spending = category_df.groupby('Month')['Amount'].sum().reset_index()
            monthly_spending['Month'] = monthly_spending['Month'].astype(str)

            fig_monthly = px.line(
                monthly_spending,
                x='Month',
                y='Amount',
                title=f'Monthly Spending Trend for {selected_category}',
                markers=True
            )
            st.plotly_chart(fig_monthly, use_container_width=True)

    # 9. Balance Projection
    st.markdown("### 🔮 Balance Projection")
    balance_df = df[['Date', 'Balance']].copy()
    balance_df = balance_df.sort_values('Date')

    # Simple projection
    last_date = balance_df['Date'].max()
    future_dates = pd.date_range(
        start=last_date + timedelta(days=1),
        periods=30,
        freq='D'
    )

    # Simple linear projection
    balance_dates = balance_df['Date'].values.astype(np.int64) // 10 ** 9
    balance_values = balance_df['Balance'].values

    if len(balance_dates) > 1:
        balance_coeffs = np.polyfit(balance_dates, balance_values, 1)
        balance_model = np.poly1d(balance_coeffs)

        future_timestamps = future_dates.values.astype(np.int64) // 10 ** 9
        future_balances = balance_model(future_timestamps)

        # Create projection dataframe
        projection_df = pd.DataFrame({
            'Date': future_dates,
            'Balance': future_balances,
            'Type': 'Projection'
        })

        # Combine with historical data
        history_df = pd.DataFrame({
            'Date': balance_df['Date'],
            'Balance': balance_df['Balance'],
            'Type': 'Historical'
        })

        combined_df = pd.concat([history_df, projection_df])

        fig_projection = px.line(
            combined_df,
            x='Date',
            y='Balance',
            color='Type',
            title='Account Balance Projection',
            line_dash='Type'
        )
        fig_projection.add_vline(x=last_date, line_dash="dash", line_color="red")
        st.plotly_chart(fig_projection, use_container_width=True)
    else:
        st.info("Not enough data for balance projection")


def financial_budget_tracker():
    """Financial Budget Tracker with advanced analytics"""
    st.markdown("## 💰 Financial Budget Tracker")

    if st.session_state.transactions_df is None or st.session_state.transactions_df.empty:
        st.warning("Please upload and analyze a bank statement first")
        return

    df = st.session_state.transactions_df.copy()
    user_id = st.session_state.user_profile['user_id']
    db = DatabaseManager()

    # Prepare date range
    min_date = df['Date'].min().to_pydatetime()
    max_date = df['Date'].max().to_pydatetime()

    # Month selection
    month_options = pd.date_range(min_date, max_date, freq='MS').strftime("%Y-%m").tolist()
    if not month_options:
        st.warning("No valid months found in transaction data")
        return

    selected_month = st.selectbox("Select Month for Budget Tracking",
                                  month_options,
                                  index=len(month_options) - 1 if month_options else 0)

    # Convert to datetime
    month_start = datetime.strptime(selected_month, "%Y-%m")
    month_end = month_start + relativedelta(months=1) - timedelta(days=1)

    # Filter transactions for selected month
    monthly_df = df[(df['Date'] >= month_start) & (df['Date'] <= month_end)]

    # Get spending categories
    expense_df = monthly_df[monthly_df['Amount'] < 0].copy()
    if expense_df.empty:
        st.info("No expenses found for the selected month")
        return

    expense_df['Amount'] = expense_df['Amount'].abs()
    categories = expense_df['Category'].unique().tolist()

    st.divider()
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### 📝 Set Your Budgets")
        budgets = {}

        # Get existing budgets for this month
        existing_budgets = db.get_budgets(user_id, selected_month)

        for category in categories:
            default_value = existing_budgets.get(category, 0.0)
            budget = st.number_input(
                f"Budget for {category} (R)",
                min_value=0.0,
                value=float(default_value),
                key=f"budget_{category}_{selected_month}"  # Unique key per month
            )
            budgets[category] = budget

        if st.button("💾 Save Budgets"):
            db.save_budgets(user_id, budgets, selected_month)
            st.session_state.budgets = budgets
            st.success("Budgets saved successfully!")
            st.rerun()

    with col2:
        st.markdown("### 📊 Budget vs Actual Spending")

        # Calculate actual spending
        actual_spending = expense_df.groupby('Category')['Amount'].sum()

        # Create comparison dataframe
        comparison_data = []
        for category in categories:
            actual = actual_spending.get(category, 0.0)
            budget = budgets.get(category, 0.0)
            utilization = (actual / budget * 100) if budget > 0 else 0

            comparison_data.append({
                'Category': category,
                'Budget': budget,
                'Actual': actual,
                'Difference': budget - actual,
                'Utilization (%)': min(utilization, 200)  # Cap at 200% for visualization
            })

        comparison_df = pd.DataFrame(comparison_data)

        if not comparison_df.empty:
            # Bar chart comparing budget vs actual
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=comparison_df['Category'],
                y=comparison_df['Budget'],
                name='Budget',
                marker_color='#1f77b4'
            ))
            fig.add_trace(go.Bar(
                x=comparison_df['Category'],
                y=comparison_df['Actual'],
                name='Actual Spending',
                marker_color='#ff7f0e'
            ))

            fig.update_layout(
                barmode='group',
                title='Budget vs Actual Spending',
                xaxis_title='Category',
                yaxis_title='Amount (R)',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

            # Utilization gauge charts
            st.markdown("### 📈 Budget Utilization")
            cols = st.columns(3)
            for i, row in comparison_df.iterrows():
                with cols[i % 3]:
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=row['Utilization (%)'],
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': f"{row['Category']}"},
                        gauge={
                            'axis': {'range': [None, 200]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 80], 'color': "green"},
                                {'range': [80, 100], 'color': "yellow"},
                                {'range': [100, 200], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "black", 'width': 4},
                                'thickness': 0.75,
                                'value': row['Utilization (%)']
                            }
                        }
                    ))
                    fig_gauge.update_layout(height=250)
                    st.plotly_chart(fig_gauge, use_container_width=True)

            # Budget performance table
            st.markdown("### 📋 Budget Performance Details")
            comparison_df['Status'] = comparison_df.apply(
                lambda row: "✅ Within Budget" if row['Actual'] <= row['Budget'] else "⚠️ Over Budget",
                axis=1
            )
            st.dataframe(
                comparison_df[['Category', 'Budget', 'Actual', 'Difference', 'Utilization (%)', 'Status']],
                use_container_width=True
            )

            # Budget forecasting
            st.markdown("### 🔮 Budget Forecast")

            # Calculate daily spending rate
            today = datetime.now().date()
            if today < month_start.date() or today > month_end.date():
                st.info("Forecast only available during the current budget period")
            else:
                days_elapsed = (today - month_start.date()).days + 1
                days_remaining = (month_end.date() - today).days

                if days_elapsed > 0 and days_remaining > 0:
                    forecast_data = []
                    for category in categories:
                        daily_spend_rate = actual_spending.get(category, 0) / days_elapsed
                        projected_spend = daily_spend_rate * (days_elapsed + days_remaining)
                        budget = budgets.get(category, 0)

                        forecast_data.append({
                            'Category': category,
                            'Current Spending': actual_spending.get(category, 0),
                            'Projected Spending': projected_spend,
                            'Budget': budget,
                            'Projected Surplus/Shortfall': budget - projected_spend
                        })

                    forecast_df = pd.DataFrame(forecast_data)
                    st.session_state.budget_forecast = forecast_df

                    # Display forecast table
                    st.dataframe(
                        forecast_df.style.applymap(
                            lambda x: 'color: red' if x < 0 else 'color: green',
                            subset=['Projected Surplus/Shortfall']
                        ),
                        use_container_width=True
                    )

                    # Forecast visualization
                    fig_forecast = go.Figure()
                    for i, row in forecast_df.iterrows():
                        fig_forecast.add_trace(go.Bar(
                            x=[row['Category']],
                            y=[row['Current Spending']],
                            name='Current Spending',
                            marker_color='#1f77b4'
                        ))
                        fig_forecast.add_trace(go.Bar(
                            x=[row['Category']],
                            y=[row['Projected Spending'] - row['Current Spending']],
                            name='Projected Additional',
                            marker_color='#ff7f0e'
                        ))
                        fig_forecast.add_trace(go.Scatter(
                            x=[row['Category']],
                            y=[row['Budget']],
                            mode='markers',
                            name='Budget Limit',
                            marker=dict(color='red', size=12, symbol='line-ns-open')
                        ))

                    fig_forecast.update_layout(
                        barmode='stack',
                        title='Current vs Projected Spending',
                        xaxis_title='Category',
                        yaxis_title='Amount (R)',
                        height=500
                    )
                    st.plotly_chart(fig_forecast, use_container_width=True)
                else:
                    st.info("Forecast only available during the current budget period")
        else:
            st.info("No spending data available for the selected month")


def ai_financial_assistant():
    """AI Financial Assistant interface"""
    st.markdown("## 🤖 AI Financial Assistant")

    if st.session_state.transactions_df is None or st.session_state.transactions_df.empty:
        st.warning("Please upload and analyze a bank statement first")
        return

    # User input for questions
    user_question = st.text_area("Ask me anything about your finances:",
                                 placeholder="How can I save more money? What are my biggest expenses? How can I reduce my spending?")

    if st.button("Get AI Assistance"):
        if not user_question:
            st.warning("Please enter a question")
            return

        with st.spinner("Analyzing your finances..."):
            # Prepare context from the transaction data
            df = st.session_state.transactions_df
            total_income = df[df['Amount'] > 0]['Amount'].sum()
            total_expenses = abs(df[df['Amount'] < 0]['Amount'].sum())
            net_flow = total_income - total_expenses

            # Top spending categories
            spending_by_category = df[df['Amount'] < 0].groupby('Category')['Amount'].sum().abs().nlargest(5)
            top_categories = "\n".join([f"- {cat}: R{amt:.2f}" for cat, amt in spending_by_category.items()])

            # Time period
            start_date = df['Date'].min().strftime('%Y-%m-%d')
            end_date = df['Date'].max().strftime('%Y-%m-%d')

            # User profile info
            user_profile = st.session_state.user_profile
            name = user_profile.get('name', 'User')
            income = user_profile.get('monthly_income', 0)
            goals = user_profile.get('financial_goals', 'Not specified')

            # Construct prompt
            prompt = f"""
            You are a financial advisor helping {name} understand their bank statement from {st.session_state.detected_bank}.
            The statement covers the period from {start_date} to {end_date}.

            Financial Summary:
            - Total Income: R{total_income:.2f}
            - Total Expenses: R{total_expenses:.2f}
            - Net Cash Flow: R{net_flow:.2f}
            - Monthly Income: R{income:.2f}
            - Financial Goals: {goals}

            Top Spending Categories:
            {top_categories}

            User Question: {user_question}

            Provide a detailed, personalized response in markdown format. Include:
            1. Specific advice based on the financial data
            2. Actionable steps the user can take
            3. References to relevant spending categories
            4. Tips to achieve their financial goals
            5. Suggestions for budgeting and saving

            Structure your response with clear headings and bullet points. Use emojis to make it engaging.
            """

            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {MISTRAL_API_KEY}"
            }

            payload = {
                "model": "mistral-large-latest",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 1500
            }

            try:
                response = requests.post(MISTRAL_API_URL, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                ai_response = data['choices'][0]['message']['content']

                # Display response
                st.markdown("### 💡 AI Financial Advice")
                st.markdown(ai_response)

            except Exception as e:
                st.error(f"Error getting response from AI assistant: {str(e)}")


def main():
    """Main application function"""
    # Initialize database
    db = DatabaseManager()

    # Sidebar for user authentication
    with st.sidebar:
        st.markdown("<h1 style='text-align: center;'>🏦 Universal Bank Analyzer</h1>", unsafe_allow_html=True)
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

        if st.session_state.user_profile is None:
            tab1, tab2 = st.tabs(["🔑 Login", "✨ Sign Up"])

            with tab1:
                st.markdown("<h3 style='text-align: center; color: #4F8BF9;'>Welcome Back!</h3>", unsafe_allow_html=True)
                st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                
                with st.container():
                    email = st.text_input("📧 Email Address", key="login_email")
                    password = st.text_input("🔒 Password", type="password", key="login_password")
                    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        login_button = st.button("🔐 Login", key="login_btn", use_container_width=True)
                    
                    if login_button:
                        with st.spinner("Authenticating..."):
                            user_data = db.authenticate_user(email, password)
                            if user_data:
                                user_id, name = user_data
                                st.session_state.user_profile = db.get_user_profile(user_id)
                                st.success(f"Welcome back, {name}!")
                                st.rerun()
                            else:
                                st.error("Invalid credentials")

            with tab2:
                st.markdown("<h3 style='text-align: center; color: #4F8BF9;'>Create Your Account</h3>", unsafe_allow_html=True)
                st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                
                with st.container():
                    col1, col2 = st.columns(2)
                    with col1:
                        new_name = st.text_input("👤 Full Name", key="signup_name")
                    with col2:
                        new_email = st.text_input("📧 Email Address", key="signup_email")
                    
                    new_password = st.text_input("🔒 Password", type="password", key="signup_password", 
                                               help="Choose a strong password with at least 8 characters")
                    
                    st.markdown("<h4 style='color: #4F8BF9;'>Financial Profile</h4>", unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        monthly_income = st.number_input("💰 Monthly Income (R)", min_value=0.0, key="signup_income")
                    with col2:
                        risk_tolerance = st.selectbox("📊 Risk Tolerance", 
                                                    ["conservative", "moderate", "aggressive"],
                                                    key="signup_risk",
                                                    help="Conservative: Low risk, Moderate: Balanced, Aggressive: Higher risk/reward")
                    
                    financial_goals = st.text_area("🎯 Financial Goals", key="signup_goals", 
                                                placeholder="e.g., Save for retirement, Buy a house, Pay off debt...")
                    
                    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        signup_button = st.button("✅ Create Account", key="signup_btn", use_container_width=True)
                    
                    if signup_button:
                        if new_name and new_email and new_password:
                            if len(new_password) < 8:
                                st.warning("Password should be at least 8 characters long")
                            else:
                                with st.spinner("Creating your account..."):
                                    user_id = hashlib.md5(new_email.encode()).hexdigest()[:8]
                                    if db.create_user(user_id, new_name, new_email, new_password, financial_goals, risk_tolerance,
                                                    monthly_income):
                                        st.success("Account created successfully! Please login.")
                                    else:
                                        st.error("Email already exists")
                        else:
                            st.error("Please fill all required fields")

        else:
            # User profile card
            st.markdown(f"""
            <div style='background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);'>
                <h3 style='color: #4F8BF9; margin-top: 0;'>👋 Welcome!</h3>
                <p style='font-size: 18px; font-weight: 500;'>{st.session_state.user_profile['name']}</p>
                <p style='color: #757575; margin-bottom: 5px;'>{st.session_state.user_profile['email']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
            
            # Quick stats
            if st.session_state.transactions_df is not None:
                df = st.session_state.transactions_df
                total_income = df[df['Amount'] > 0]['Amount'].sum()
                total_expenses = abs(df[df['Amount'] < 0]['Amount'].sum())
                
                st.markdown("<h4 style='color: #4F8BF9;'>Quick Stats</h4>", unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Income", f"R{total_income:,.2f}")
                with col2:
                    st.metric("Expenses", f"R{total_expenses:,.2f}")
            
            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
            
            # Logout button
            if st.button("🚪 Logout", use_container_width=True):
                st.session_state.user_profile = None
                st.session_state.transactions_df = None
                st.session_state.analysis_complete = False
                st.session_state.detected_bank = None
                st.session_state.mistral_insights = None
                st.session_state.budgets = {}
                st.session_state.budget_forecast = None
                st.experimental_rerun()

    # Main content area
    if st.session_state.user_profile is None:
        # Hero section with modern design
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <div style='padding: 1.5rem 0;'>
                <h1 style='font-size: 3rem; font-weight: 700; color: #4F8BF9;'>🏦 Universal Bank Statement Analysis</h1>
                <p style='font-size: 1.2rem; color: #757575; margin-bottom: 2rem;'>The next generation of financial analysis for all major banks</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Call-to-action button
            st.markdown("""
            <div style='background-color: #F5F7FA; padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;'>
                <h3 style='color: #4F8BF9; margin-top: 0;'>Ready to take control of your finances?</h3>
                <p>Create an account or login to get started with your financial journey.</p>
                <p style='font-weight: 500;'>👉 Use the sidebar to login or create your account</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Decorative image or icon
            st.markdown("""
            <div style='display: flex; justify-content: center; align-items: center; height: 100%;'>
                <svg width="200" height="200" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <rect x="2" y="6" width="20" height="12" rx="2" fill="#4F8BF9" opacity="0.2"/>
                    <rect x="2" y="6" width="20" height="12" rx="2" stroke="#4F8BF9" stroke-width="2"/>
                    <path d="M12 2V4" stroke="#4F8BF9" stroke-width="2" stroke-linecap="round"/>
                    <path d="M12 20V22" stroke="#4F8BF9" stroke-width="2" stroke-linecap="round"/>
                    <path d="M8 12H16" stroke="#4F8BF9" stroke-width="2" stroke-linecap="round"/>
                    <path d="M12 9V15" stroke="#4F8BF9" stroke-width="2" stroke-linecap="round"/>
                </svg>
            </div>
            """, unsafe_allow_html=True)
        
        # Features section with cards
        st.markdown("""
        <h2 style='color: #4F8BF9; margin-top: 2rem;'>Key Features</h2>
        <div style='height: 20px;'></div>
        """, unsafe_allow_html=True)
        
        # Row 1 of feature cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style='background-color: white; padding: 1.5rem; border-radius: 10px; height: 200px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);'>
                <h3 style='color: #4F8BF9; font-size: 1.5rem;'>🤖 AI-Powered Insights</h3>
                <p>Advanced machine learning algorithms analyze your financial data to provide personalized recommendations and insights.</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div style='background-color: white; padding: 1.5rem; border-radius: 10px; height: 200px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);'>
                <h3 style='color: #4F8BF9; font-size: 1.5rem;'>📊 Comprehensive Analytics</h3>
                <p>Deep dive into your spending patterns with detailed visualizations and trend analysis to better understand your finances.</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown("""
            <div style='background-color: white; padding: 1.5rem; border-radius: 10px; height: 200px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);'>
                <h3 style='color: #4F8BF9; font-size: 1.5rem;'>🎯 Goal Tracking</h3>
                <p>Set financial goals and track your progress over time with our intuitive goal tracking system.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Row 2 of feature cards
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style='background-color: white; padding: 1.5rem; border-radius: 10px; height: 200px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);'>
                <h3 style='color: #4F8BF9; font-size: 1.5rem;'>🛡️ Anomaly Detection</h3>
                <p>Automatically identify unusual transactions and potential fraud with our advanced anomaly detection system.</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div style='background-color: white; padding: 1.5rem; border-radius: 10px; height: 200px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);'>
                <h3 style='color: #4F8BF9; font-size: 1.5rem;'>💰 Loan Eligibility</h3>
                <p>Check your loan eligibility instantly with our predictive models based on your financial history and patterns.</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown("""
            <div style='background-color: white; padding: 1.5rem; border-radius: 10px; height: 200px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);'>
                <h3 style='color: #4F8BF9; font-size: 1.5rem;'>💸 Budget Tracker</h3>
                <p>Create and monitor budgets for different spending categories to keep your finances on track.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Supported banks section
        st.markdown("""
        <div style='height: 40px;'></div>
        <h2 style='color: #4F8BF9;'>Supported Banks</h2>
        <div style='height: 20px;'></div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown("""
            <div style='background-color: white; padding: 1rem; border-radius: 10px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.05);'>
                <h4 style='color: #4F8BF9;'>FNB</h4>
                <p>First National Bank</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div style='background-color: white; padding: 1rem; border-radius: 10px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.05);'>
                <h4 style='color: #4F8BF9;'>Standard Bank</h4>
                <p>Standard Bank SA</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown("""
            <div style='background-color: white; padding: 1rem; border-radius: 10px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.05);'>
                <h4 style='color: #4F8BF9;'>Nedbank</h4>
                <p>Nedbank Group</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown("""
            <div style='background-color: white; padding: 1rem; border-radius: 10px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.05);'>
                <h4 style='color: #4F8BF9;'>ABSA</h4>
                <p>ABSA Group Limited</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col5:
            st.markdown("""
            <div style='background-color: white; padding: 1rem; border-radius: 10px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.05);'>
                <h4 style='color: #4F8BF9;'>Capitec</h4>
                <p>Capitec Bank</p>
            </div>
            """, unsafe_allow_html=True)
        
        return

    # Main analysis interface - Modern header with user profile
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"""
        <div style='background-color: #4F8BF9; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;'>
            <h1 style='color: white; margin: 0;'>🏦 Universal Bank Analysis Dashboard</h1>
            <p style='color: white; opacity: 0.9; margin: 0;'>Welcome back, {st.session_state.user_profile['name']}!</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # User profile card
        st.markdown(f"""
        <div style='background-color: white; padding: 1rem; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); text-align: center;'>
            <div style='background-color: #4F8BF9; width: 50px; height: 50px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; margin-bottom: 0.5rem;'>
                <span style='color: white; font-size: 1.5rem;'>{st.session_state.user_profile['name'][0].upper()}</span>
            </div>
            <p style='font-weight: bold; margin: 0;'>{st.session_state.user_profile['name']}</p>
            <p style='color: #757575; font-size: 0.8rem; margin: 0;'>{st.session_state.user_profile['email']}</p>
        </div>
        """, unsafe_allow_html=True)

    # File upload section with modern design
    st.markdown("""
    <div style='height: 20px;'></div>
    <h3 style='color: #4F8BF9;'>📄 Upload Your Bank Statement</h3>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a PDF bank statement",
            type="pdf",
            help="Upload your bank statement in PDF format for analysis"
        )
    
    with col2:
        st.markdown("""
        <div style='background-color: #F5F7FA; padding: 1rem; border-radius: 10px; height: 100%;'>
            <h4 style='color: #4F8BF9; margin-top: 0;'>Supported Banks</h4>
            <p style='margin: 0; font-size: 0.9rem;'>FNB, Standard Bank, Nedbank, ABSA, Capitec</p>
        </div>
        """, unsafe_allow_html=True)

    if uploaded_file is not None:
        try:
            # Parse PDF and identify bank
            with st.spinner("🔍 Analyzing bank statement..."):
                text = extract_text_from_pdf(uploaded_file)
                bank_name = identify_bank_from_text(text)

                if not bank_name:
                    st.error("Could not identify bank from statement. Please ensure it's from a supported bank.")
                    return

                st.session_state.detected_bank = bank_name
                st.success(f"Detected Bank: {bank_name}")

                # Extract metadata based on bank
                if bank_name == 'FNB':
                    account_holder_name, closing_balance = extract_bank_statement_metadata(text, bank_name)
                elif bank_name in ['Standard Bank', 'Nedbank', 'ABSA']:
                    account_holder_name, account_number, statement_period, closing_balance = extract_bank_statement_metadata(
                        text, bank_name)
                else:
                    account_holder_name, closing_balance = extract_bank_statement_metadata(text, bank_name)

                # Parse transactions
                df = process_text_to_df_enhanced(text, bank_name)

            if df.empty:
                st.warning("⚠️ No transactions found in the uploaded statement. Please check the file format.")
                return

            # Store in session state
            st.session_state.transactions_df = df

            # Enhance data with categories
            df['Category'] = df['Description'].apply(lambda x: categorize_expense_enhanced(x, bank_name))

            # Initialize analytics engines
            personalization = PersonalizationEngine(st.session_state.user_profile)
            analytics = AdvancedAnalytics()

            # Perform analysis
            with st.spinner("🧠 Analyzing your financial data..."):
                patterns = personalization.analyze_spending_patterns(df)
                recommendations = personalization.generate_personalized_recommendations(df, patterns)
                health_score, score_components = personalization.calculate_financial_health_score(df, patterns)
                patterns['score_components'] = score_components  # Store for visualization
                loan_prediction = analytics.enhanced_loan_prediction(df, bank_name)
                df_with_anomalies = analytics.detect_anomalies(df)

            st.session_state.analysis_complete = True

            # Display account info
            if bank_name == 'FNB':
                st.success(f"Account Holder: {account_holder_name}")
                st.info(f"Closing Balance: R{closing_balance:,.2f}")
            elif bank_name in ['Standard Bank', 'Nedbank', 'ABSA']:
                st.success(f"Account Holder: {account_holder_name}")
                st.info(f"Account Number: {account_number} | Statement Period: {statement_period}")
                st.info(f"Closing Balance: R{closing_balance:,.2f}")
            else:
                st.success(f"Account Holder: {account_holder_name}")
                st.info(f"Closing Balance: R{closing_balance:,.2f}")

            # Display results in tabs
            tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
                "📊 Overview", "💡 Recommendations", "🏥 Health Score",
                "🔍 Detailed Analysis", "🚨 Anomalies", "💰 Loan Eligibility",
                "📝 Budget Tracker", "🤖 AI Assistant"
            ])

            with tab1:
                # Modern Financial Overview with cards
                st.markdown("""
                <h3 style='color: #4F8BF9;'>📈 Financial Overview</h3>
                <div style='height: 10px;'></div>
                """, unsafe_allow_html=True)
                
                # Summary cards
                col1, col2, col3, col4 = st.columns(4)
                
                # Calculate summary metrics
                total_income = df[df['Amount'] > 0]['Amount'].sum()
                total_expenses = abs(df[df['Amount'] < 0]['Amount'].sum())
                net_flow = total_income - total_expenses
                avg_daily_spending = total_expenses / max(1, len(df['Date'].unique()))
                
                with col1:
                    st.markdown(f"""
                    <div style='background-color: white; padding: 1rem; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);'>
                        <p style='color: #757575; font-size: 0.9rem; margin: 0;'>Total Income</p>
                        <h2 style='color: #4F8BF9; margin: 0;'>R{total_income:,.2f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div style='background-color: white; padding: 1rem; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);'>
                        <p style='color: #757575; font-size: 0.9rem; margin: 0;'>Total Expenses</p>
                        <h2 style='color: #FF6B6B; margin: 0;'>R{total_expenses:,.2f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    color = "#4F8BF9" if net_flow >= 0 else "#FF6B6B"
                    st.markdown(f"""
                    <div style='background-color: white; padding: 1rem; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);'>
                        <p style='color: #757575; font-size: 0.9rem; margin: 0;'>Net Cash Flow</p>
                        <h2 style='color: {color}; margin: 0;'>R{net_flow:,.2f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div style='background-color: white; padding: 1rem; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);'>
                        <p style='color: #757575; font-size: 0.9rem; margin: 0;'>Avg. Daily Spending</p>
                        <h2 style='color: #757575; margin: 0;'>R{avg_daily_spending:,.2f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
                
                # Visualizations with better styling
                create_advanced_visualizations(df, patterns, recommendations)
                
                # Transaction history with modern styling
                st.markdown("""
                <div style='height: 20px;'></div>
                <h3 style='color: #4F8BF9;'>📋 Transaction History</h3>
                <div style='height: 10px;'></div>
                """, unsafe_allow_html=True)
                
                # Search and filter options
                col1, col2 = st.columns([3, 1])
                with col1:
                    search_term = st.text_input("🔍 Search transactions", "")
                with col2:
                    category_filter = st.selectbox("Filter by category", ["All Categories"] + sorted(df['Category'].unique().tolist()))
                
                # Apply filters
                filtered_df = df.copy()
                if search_term:
                    filtered_df = filtered_df[filtered_df['Description'].str.contains(search_term, case=False)]
                if category_filter != "All Categories":
                    filtered_df = filtered_df[filtered_df['Category'] == category_filter]
                
                # Display filtered transactions with styling
                st.dataframe(
                    filtered_df[['Date', 'Description', 'Category', 'Amount', 'Balance']],
                    use_container_width=True,
                    height=400
                )
                
                # Transaction summary
                st.markdown(f"""
                <div style='background-color: #F5F7FA; padding: 0.5rem 1rem; border-radius: 5px; font-size: 0.9rem;'>
                    Showing {len(filtered_df)} of {len(df)} transactions
                </div>
                """, unsafe_allow_html=True)

            with tab2:
                # Modern header with action button
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown("""
                    <h3 style='color: #4F8BF9;'>💡 Personalized Recommendations</h3>
                    """, unsafe_allow_html=True)
                with col2:
                    if st.button("✨ Generate AI Insights", use_container_width=True, type="primary"):
                        with st.spinner("Generating AI-powered insights..."):
                            st.session_state.mistral_insights = get_mistral_insights(
                                df,
                                st.session_state.user_profile,
                                bank_name
                            )
                
                # AI Insights section with modern styling
                if st.session_state.mistral_insights:
                    st.markdown("""
                    <div style='background-color: #F0F7FF; padding: 1.5rem; border-radius: 10px; border-left: 5px solid #4F8BF9; margin-bottom: 1.5rem;'>
                        <h4 style='color: #4F8BF9; margin-top: 0;'>🤖 AI-Powered Financial Insights</h4>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(st.session_state.mistral_insights)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Recommendations with card-based design
                if recommendations:
                    # Group recommendations by priority
                    high_priority = [rec for rec in recommendations if rec['priority'] == 'high']
                    medium_priority = [rec for rec in recommendations if rec['priority'] == 'medium']
                    low_priority = [rec for rec in recommendations if rec['priority'] == 'low']
                    
                    # Display high priority recommendations
                    if high_priority:
                        st.markdown("""
                        <h4 style='color: #FF6B6B;'>🔴 High Priority Recommendations</h4>
                        <div style='height: 10px;'></div>
                        """, unsafe_allow_html=True)
                        
                        for rec in high_priority:
                            st.markdown(f"""
                            <div style='background-color: white; padding: 1.5rem; border-radius: 10px; border-left: 5px solid #FF6B6B; margin-bottom: 1rem; box-shadow: 0 2px 5px rgba(0,0,0,0.05);'>
                                <h4 style='color: #FF6B6B; margin-top: 0;'>{rec['title']}</h4>
                                <p>{rec['description']}</p>
                                <p style='color: #757575; font-size: 0.9rem; margin-bottom: 0;'>Category: {rec['type'].title()} | Priority: High</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Display medium priority recommendations
                    if medium_priority:
                        st.markdown("""
                        <h4 style='color: #FFB347;'>🟡 Medium Priority Recommendations</h4>
                        <div style='height: 10px;'></div>
                        """, unsafe_allow_html=True)
                        
                        for rec in medium_priority:
                            st.markdown(f"""
                            <div style='background-color: white; padding: 1.5rem; border-radius: 10px; border-left: 5px solid #FFB347; margin-bottom: 1rem; box-shadow: 0 2px 5px rgba(0,0,0,0.05);'>
                                <h4 style='color: #FFB347; margin-top: 0;'>{rec['title']}</h4>
                                <p>{rec['description']}</p>
                                <p style='color: #757575; font-size: 0.9rem; margin-bottom: 0;'>Category: {rec['type'].title()} | Priority: Medium</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Display low priority recommendations
                    if low_priority:
                        st.markdown("""
                        <h4 style='color: #4CAF50;'>🟢 Low Priority Recommendations</h4>
                        <div style='height: 10px;'></div>
                        """, unsafe_allow_html=True)
                        
                        for rec in low_priority:
                            st.markdown(f"""
                            <div style='background-color: white; padding: 1.5rem; border-radius: 10px; border-left: 5px solid #4CAF50; margin-bottom: 1rem; box-shadow: 0 2px 5px rgba(0,0,0,0.05);'>
                                <h4 style='color: #4CAF50; margin-top: 0;'>{rec['title']}</h4>
                                <p>{rec['description']}</p>
                                <p style='color: #757575; font-size: 0.9rem; margin-bottom: 0;'>Category: {rec['type'].title()} | Priority: Low</p>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style='background-color: #E8F5E9; padding: 1.5rem; border-radius: 10px; border-left: 5px solid #4CAF50; text-align: center;'>
                        <h4 style='color: #4CAF50; margin-top: 0;'>💫 Great job!</h4>
                        <p>Your financial habits look healthy. Keep up the good work!</p>
                    </div>
                    """, unsafe_allow_html=True)

            with tab3:
                # Modern header
                st.markdown("""
                <h3 style='color: #4F8BF9;'>🏥 Financial Health Score</h3>
                <div style='height: 10px;'></div>
                """, unsafe_allow_html=True)
                
                # Score overview card
                score_color = "#4CAF50" if health_score >= 80 else "#FFB347" if health_score >= 60 else "#FF6B6B"
                score_text = "Excellent" if health_score >= 80 else "Good" if health_score >= 60 else "Needs Attention"
                
                st.markdown(f"""
                <div style='background-color: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); margin-bottom: 1.5rem;'>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <div>
                            <h2 style='color: {score_color}; margin: 0; font-size: 3rem;'>{health_score:.1f}</h2>
                            <p style='color: #757575; margin: 0;'>out of 100</p>
                            <p style='color: {score_color}; font-weight: bold; margin-top: 0.5rem;'>{score_text}</p>
                        </div>
                        <div style='width: 100px; height: 100px; border-radius: 50%; background: conic-gradient({score_color} {health_score}%, #f0f0f0 0); display: flex; align-items: center; justify-content: center;'>
                            <div style='width: 80px; height: 80px; border-radius: 50%; background-color: white; display: flex; align-items: center; justify-content: center;'>
                                <span style='font-weight: bold;'>{health_score:.1f}%</span>
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Display overall score with improved layout
                col1, col2 = st.columns([1, 2])

                with col1:
                    # Create gauge chart for health score with better styling
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=health_score,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Financial Health Score", 'font': {'color': "#4F8BF9", 'size': 16}},
                        delta={'reference': 75, 'increasing': {'color': "#4CAF50"}, 'decreasing': {'color': "#FF6B6B"}},
                        gauge={
                            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#4F8BF9"},
                            'bar': {'color': "#4F8BF9"},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "#4F8BF9",
                            'steps': [
                                {'range': [0, 25], 'color': "#FFCDD2"},  # Light red
                                {'range': [25, 50], 'color': "#FFECB3"},  # Light yellow
                                {'range': [50, 75], 'color': "#C8E6C9"},  # Light green
                                {'range': [75, 100], 'color': "#A5D6A7"}   # Medium green
                            ],
                            'threshold': {
                                'line': {'color': "#4CAF50", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        }
                    ))
                    fig_gauge.update_layout(
                        height=300,
                        margin=dict(l=20, r=20, t=50, b=20),
                        paper_bgcolor="white",
                        font={'color': "#4F8BF9", 'family': "Arial"}
                    )
                    st.plotly_chart(fig_gauge, use_container_width=True)

                with col2:
                    st.markdown("""
                    <h4 style='color: #4F8BF9;'>Score Breakdown</h4>
                    <div style='height: 10px;'></div>
                    """, unsafe_allow_html=True)
                    
                    # Component cards with better styling
                    for component, score in score_components.items():
                        component_name = component.replace('_', ' ').title()
                        component_color = "#4CAF50" if score >= 20 else "#FFB347" if score >= 15 else "#FF6B6B"
                        
                        st.markdown(f"""
                        <div style='background-color: white; padding: 1rem; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); margin-bottom: 0.5rem;'>
                            <p style='margin: 0; font-weight: bold;'>{component_name}</p>
                            <div style='display: flex; align-items: center; margin-top: 0.5rem;'>
                                <div style='flex-grow: 1; height: 8px; background-color: #f0f0f0; border-radius: 4px; margin-right: 10px;'>
                                    <div style='width: {(score/25)*100}%; height: 100%; background-color: {component_color}; border-radius: 4px;'></div>
                                </div>
                                <span style='color: {component_color}; font-weight: bold;'>{score:.1f}/25</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                # Health recommendations with modern styling
                if health_score < 60:
                    st.markdown("""
                    <div style='background-color: #FFEBEE; padding: 1rem; border-radius: 10px; border-left: 5px solid #FF6B6B; margin-top: 1.5rem;'>
                        <h4 style='color: #FF6B6B; margin-top: 0;'>⚠️ Action Required</h4>
                        <p>Your financial health needs attention. Consider the recommendations in the previous tab to improve your score.</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif health_score < 80:
                    st.markdown("""
                    <div style='background-color: #FFF8E1; padding: 1rem; border-radius: 10px; border-left: 5px solid #FFB347; margin-top: 1.5rem;'>
                        <h4 style='color: #FFB347; margin-top: 0;'>💡 Good Progress</h4>
                        <p>Your financial health is good! A few improvements could boost your score even further.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style='background-color: #E8F5E9; padding: 1rem; border-radius: 10px; border-left: 5px solid #4CAF50; margin-top: 1.5rem;'>
                        <h4 style='color: #4CAF50; margin-top: 0;'>🎉 Excellent!</h4>
                        <p>Your financial health is excellent! You're doing great with your money management.</p>
                    </div>
                    """, unsafe_allow_html=True)

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

                # Ensure the anomaly column exists
                if 'IsAnomaly' in df_with_anomalies.columns:
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
                else:
                    st.info("ℹ️ Anomaly detection not performed. Not enough transaction data.")

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

            with tab7:
                financial_budget_tracker()

            with tab8:
                ai_financial_assistant()

        except Exception as e:
            st.error(f"❌ An error occurred while processing your statement: {str(e)}")
            st.info("Please ensure your PDF is a valid bank statement and try again.")


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

    .ai-response {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        border-left: 4px solid #4e89e9;
    }

    .budget-card {
        background-color: #e6f7ff;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        border-left: 4px solid #1890ff;
    }

    .positive-budget {
        color: #52c41a;
        font-weight: bold;
    }

    .negative-budget {
        color: #f5222d;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)