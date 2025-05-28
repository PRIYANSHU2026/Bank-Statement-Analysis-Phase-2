"""
Ollama Local LLM Integration for Personalized Financial Assistant
This module provides AI-powered financial advice using local Llama 3.2 model
"""

import json
import requests
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any
import streamlit as st


class OllamaFinancialAssistant:
    """
    Personalized Financial Assistant using Ollama Local LLM (Llama 3.2)
    Provides contextual financial advice based on user's transaction data and profile
    """

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.model_name = "llama3.2"
        self.conversation_history = []
        self.user_context = {}

    def check_ollama_status(self) -> bool:
        """Check if Ollama service is running and model is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                return any(self.model_name in model.get('name', '') for model in models)
            return False
        except requests.exceptions.RequestException:
            return False

    def pull_model(self) -> bool:
        """Pull the Llama 3.2 model if not available"""
        try:
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": self.model_name},
                timeout=300  # 5 minutes timeout for model download
            )
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def set_user_context(self, user_profile: Dict, financial_data: pd.DataFrame,
                        patterns: Dict, health_score: float):
        """Set user context for personalized responses"""
        self.user_context = {
            "profile": user_profile,
            "financial_summary": self._create_financial_summary(financial_data),
            "spending_patterns": patterns,
            "health_score": health_score,
            "last_updated": datetime.now().isoformat()
        }

    def _create_financial_summary(self, df: pd.DataFrame) -> Dict:
        """Create a concise financial summary for the LLM"""
        if df.empty:
            return {}

        return {
            "total_income": float(df[df['Amount'] > 0]['Amount'].sum()),
            "total_expenses": float(abs(df[df['Amount'] < 0]['Amount'].sum())),
            "transaction_count": len(df),
            "date_range": {
                "start": df['Date'].min().strftime('%Y-%m-%d') if not df.empty else None,
                "end": df['Date'].max().strftime('%Y-%m-%d') if not df.empty else None
            },
            "top_expense_categories": df[df['Amount'] < 0].groupby('Category')['Amount'].sum().abs().nlargest(5).to_dict(),
            "average_daily_spending": float(df[df['Amount'] < 0]['Amount'].mean()) if len(df[df['Amount'] < 0]) > 0 else 0
        }

    def _create_system_prompt(self) -> str:
        """Create a comprehensive system prompt for the financial assistant"""
        return f"""
You are a highly skilled and empathetic personal financial advisor assistant. Your name is FinanceAI.

ROLE & PERSONALITY:
- You are a professional financial advisor with expertise in personal finance, budgeting, and investment planning
- You communicate in a friendly, supportive, and encouraging manner
- You provide actionable, practical advice tailored to the user's specific financial situation
- You always consider the user's risk tolerance and financial goals when giving advice
- You explain complex financial concepts in simple, understandable terms

USER CONTEXT:
- User Name: {self.user_context.get('profile', {}).get('name', 'User')}
- Monthly Income: R{self.user_context.get('profile', {}).get('monthly_income', 0):,.2f}
- Risk Tolerance: {self.user_context.get('profile', {}).get('risk_tolerance', 'moderate')}
- Financial Goals: {self.user_context.get('profile', {}).get('financial_goals', 'Not specified')}
- Financial Health Score: {self.user_context.get('health_score', 0)}/100

FINANCIAL SUMMARY:
- Total Monthly Income: R{self.user_context.get('financial_summary', {}).get('total_income', 0):,.2f}
- Total Monthly Expenses: R{self.user_context.get('financial_summary', {}).get('total_expenses', 0):,.2f}
- Number of Transactions: {self.user_context.get('financial_summary', {}).get('transaction_count', 0)}
- Average Daily Spending: R{abs(self.user_context.get('financial_summary', {}).get('average_daily_spending', 0)):,.2f}

GUIDELINES:
1. Always address the user by their name when possible
2. Reference their specific financial data in your responses
3. Provide actionable advice based on their actual spending patterns
4. Consider their risk tolerance when suggesting investments
5. Be encouraging about positive financial behaviors
6. Offer specific, measurable goals when suggesting improvements
7. Use South African Rand (R) currency format
8. Keep responses concise but comprehensive (2-4 paragraphs)
9. Always end with a supportive, motivational statement

RESPONSE FORMAT:
- Start with a personalized greeting acknowledging their question
- Provide specific advice based on their data
- Include 2-3 actionable recommendations
- End with encouragement and offer to help with follow-up questions
"""

    def generate_response(self, user_message: str, include_context: bool = True) -> str:
        """Generate AI response using Ollama"""
        try:
            # Prepare the prompt
            if include_context and self.user_context:
                system_prompt = self._create_system_prompt()
                full_prompt = f"{system_prompt}\n\nUser Question: {user_message}"
            else:
                full_prompt = f"You are a helpful financial advisor. User: {user_message}"

            # Make request to Ollama
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_tokens": 1000
                    }
                },
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                ai_response = result.get('response', '').strip()

                # Store conversation history
                self.conversation_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "user": user_message,
                    "assistant": ai_response
                })

                return ai_response
            else:
                return "I apologize, but I'm having trouble processing your request right now. Please try again in a moment."

        except requests.exceptions.RequestException as e:
            return f"I'm currently unavailable. Please ensure Ollama is running with the llama3.2 model. Error: {str(e)}"
        except Exception as e:
            return f"An unexpected error occurred: {str(e)}"

    def get_financial_insights(self) -> List[str]:
        """Generate automated financial insights based on user data"""
        if not self.user_context:
            return ["Upload your bank statement to receive personalized insights."]

        insights_prompts = [
            "Based on my spending patterns, what are the top 3 areas where I can reduce expenses?",
            "How can I improve my financial health score?",
            "What investment opportunities would you recommend for my risk tolerance?",
            "How am I doing compared to typical spending patterns for someone with my income level?"
        ]

        insights = []
        for prompt in insights_prompts:
            response = self.generate_response(prompt, include_context=True)
            insights.append(response)

        return insights

    def get_spending_advice(self, category: str, amount: float) -> str:
        """Get specific advice about spending in a particular category"""
        prompt = f"I spent R{amount:,.2f} on {category} this month. Is this reasonable, and how can I optimize this spending?"
        return self.generate_response(prompt, include_context=True)

    def get_savings_strategy(self, target_amount: float, timeframe_months: int) -> str:
        """Get a personalized savings strategy"""
        prompt = f"I want to save R{target_amount:,.2f} in {timeframe_months} months. Based on my current financial situation, what's the best strategy?"
        return self.generate_response(prompt, include_context=True)

    def get_budget_recommendations(self) -> str:
        """Get personalized budget recommendations"""
        prompt = "Based on my current spending patterns and income, can you suggest an optimized monthly budget breakdown?"
        return self.generate_response(prompt, include_context=True)

    def analyze_financial_goals(self) -> str:
        """Analyze progress toward financial goals"""
        goals = self.user_context.get('profile', {}).get('financial_goals', '')
        if goals:
            prompt = f"My financial goals are: {goals}. Based on my current financial situation, how am I progressing toward these goals and what should I focus on?"
        else:
            prompt = "I haven't set specific financial goals yet. Based on my financial situation, what goals should I consider and how can I achieve them?"

        return self.generate_response(prompt, include_context=True)


class ConversationManager:
    """Manages conversation flow and history for the AI assistant"""

    def __init__(self):
        self.conversations = []
        self.current_conversation_id = None

    def start_new_conversation(self, user_id: str) -> str:
        """Start a new conversation session"""
        conversation_id = f"{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_conversation_id = conversation_id
        self.conversations.append({
            "id": conversation_id,
            "user_id": user_id,
            "started_at": datetime.now().isoformat(),
            "messages": []
        })
        return conversation_id

    def add_message(self, user_message: str, ai_response: str):
        """Add a message pair to the current conversation"""
        if self.current_conversation_id:
            for conv in self.conversations:
                if conv["id"] == self.current_conversation_id:
                    conv["messages"].append({
                        "timestamp": datetime.now().isoformat(),
                        "user": user_message,
                        "assistant": ai_response
                    })
                    break

    def get_conversation_history(self, conversation_id: str = None) -> List[Dict]:
        """Get conversation history"""
        target_id = conversation_id or self.current_conversation_id
        for conv in self.conversations:
            if conv["id"] == target_id:
                return conv["messages"]
        return []

    def export_conversation(self, conversation_id: str = None) -> str:
        """Export conversation as formatted text"""
        messages = self.get_conversation_history(conversation_id)
        if not messages:
            return "No conversation history found."

        export_text = f"Financial Assistant Conversation Export\n"
        export_text += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        export_text += "=" * 50 + "\n\n"

        for msg in messages:
            timestamp = datetime.fromisoformat(msg["timestamp"]).strftime('%H:%M')
            export_text += f"[{timestamp}] You: {msg['user']}\n\n"
            export_text += f"[{timestamp}] FinanceAI: {msg['assistant']}\n\n"
            export_text += "-" * 30 + "\n\n"

        return export_text


# Quick setup functions for easy integration
def setup_ollama_assistant() -> OllamaFinancialAssistant:
    """Setup and verify Ollama assistant"""
    assistant = OllamaFinancialAssistant()

    if not assistant.check_ollama_status():
        st.warning("⚠️ Ollama service not detected or llama3.2 model not available.")

        with st.expander("🔧 Setup Instructions"):
            st.markdown("""
            ### Setting up Ollama for AI Assistant

            1. **Install Ollama** (if not already installed):
               ```bash
               # On Mac
               brew install ollama

               # On Linux
               curl -fsSL https://ollama.ai/install.sh | sh

               # On Windows
               # Download from https://ollama.ai/download
               ```

            2. **Start Ollama service**:
               ```bash
               ollama serve
               ```

            3. **Pull the Llama 3.2 model**:
               ```bash
               ollama pull llama3.2
               ```

            4. **Refresh this page** after completing setup.
            """)

        if st.button("🔄 Try to Auto-Pull Model", help="Attempt to automatically download llama3.2"):
            with st.spinner("Downloading llama3.2 model... This may take several minutes."):
                if assistant.pull_model():
                    st.success("✅ Model downloaded successfully! Please refresh the page.")
                else:
                    st.error("❌ Failed to download model. Please follow manual setup instructions.")

    return assistant


def create_assistant_interface(assistant: OllamaFinancialAssistant, user_profile: Dict,
                             financial_data: pd.DataFrame, patterns: Dict, health_score: float):
    """Create the AI assistant interface in Streamlit"""

    # Set user context
    assistant.set_user_context(user_profile, financial_data, patterns, health_score)

    st.markdown("### 🤖 Personal Financial Assistant (Powered by Llama 3.2)")

    # Check if Ollama is available
    if not assistant.check_ollama_status():
        st.error("🔌 AI Assistant is currently offline. Please check Ollama setup.")
        return

    # Initialize conversation manager
    if 'conversation_manager' not in st.session_state:
        st.session_state.conversation_manager = ConversationManager()
        st.session_state.conversation_manager.start_new_conversation(user_profile.get('user_id', 'anonymous'))

    # Quick action buttons
    st.markdown("#### 🚀 Quick Financial Insights")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("💡 Budget Analysis", use_container_width=True):
            with st.spinner("Analyzing your budget..."):
                response = assistant.get_budget_recommendations()
                st.session_state.last_ai_response = response

    with col2:
        if st.button("🎯 Goal Progress", use_container_width=True):
            with st.spinner("Reviewing your financial goals..."):
                response = assistant.analyze_financial_goals()
                st.session_state.last_ai_response = response

    with col3:
        if st.button("💰 Savings Strategy", use_container_width=True):
            target = st.number_input("Target amount (R)", value=10000, min_value=1000, step=1000, key="savings_target")
            months = st.number_input("Timeframe (months)", value=12, min_value=1, max_value=60, key="savings_months")
            with st.spinner("Creating savings strategy..."):
                response = assistant.get_savings_strategy(target, months)
                st.session_state.last_ai_response = response

    # Display AI response
    if 'last_ai_response' in st.session_state:
        st.markdown("#### 🎯 AI Recommendation")
        st.markdown(f"💬 **FinanceAI**: {st.session_state.last_ai_response}")
        st.divider()

    # Chat interface
    st.markdown("#### 💬 Ask Your Financial Assistant")

    # Chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for chat in st.session_state.chat_history:
        with st.container():
            st.markdown(f"**You**: {chat['user']}")
            st.markdown(f"**FinanceAI**: {chat['assistant']}")
            st.divider()

    # Chat input
    user_question = st.text_input(
        "Ask me anything about your finances...",
        placeholder="e.g., How can I reduce my grocery spending? Should I invest more?",
        key="user_question"
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        send_button = st.button("💬 Send", use_container_width=True)

    if send_button and user_question:
        with st.spinner("FinanceAI is thinking..."):
            ai_response = assistant.generate_response(user_question, include_context=True)

            # Add to chat history
            st.session_state.chat_history.append({
                'user': user_question,
                'assistant': ai_response
            })

            # Add to conversation manager
            st.session_state.conversation_manager.add_message(user_question, ai_response)

            # Clear input and rerun
            st.session_state.user_question = ""
            st.experimental_rerun()

    # Export conversation
    with st.expander("📥 Export Conversation"):
        if st.button("📄 Download Conversation History"):
            conversation_text = st.session_state.conversation_manager.export_conversation()
            st.download_button(
                label="💾 Download as Text File",
                data=conversation_text,
                file_name=f"financial_assistant_conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )


# Example usage and testing functions
def test_assistant():
    """Test function for the assistant"""
    assistant = OllamaFinancialAssistant()

    # Test basic functionality
    test_user_profile = {
        'name': 'John Doe',
        'monthly_income': 50000,
        'risk_tolerance': 'moderate',
        'financial_goals': 'Save for a house deposit'
    }

    test_df = pd.DataFrame({
        'Date': pd.date_range('2024-01-01', periods=30),
        'Amount': [-500, 1000, -200, -150, -300] * 6,
        'Category': ['Groceries', 'Salary', 'Transport', 'Entertainment', 'Utilities'] * 6
    })

    assistant.set_user_context(test_user_profile, test_df, {}, 75)

    # Test response
    response = assistant.generate_response("How can I improve my savings?")
    print("Test Response:", response)


if __name__ == "__main__":
    test_assistant()
