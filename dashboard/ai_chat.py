"""AI Assistant powered by Anthropic Claude."""
import streamlit as st

SYSTEM_PROMPT = """You are a professional financial analyst and trading assistant embedded in a personal trading dashboard.

The user is a retail investor with $250 starting capital, paper trading with a Long/Short Equity strategy using sector-paired momentum signals.

Your role:
- Analyze stocks, sectors, and market conditions clearly and concisely
- Explain trading strategies, financial concepts, and risk management
- Interpret financial data and metrics the user shares
- Help the user understand their dashboard signals and backtest results
- Provide educational insights about hedge fund strategies scaled to retail investing

Always:
- Be direct and specific — no vague advice
- Use plain language (the user is learning, not a CFA)
- Mention relevant risks alongside opportunities
- Remind the user this is educational, not personalized financial advice

Never:
- Guarantee returns
- Recommend illegal activities (front-running, insider trading)
- Make decisions for the user
"""


def get_client(api_key: str):
    if not api_key:
        return None
    try:
        import anthropic
        return anthropic.Anthropic(api_key=api_key)
    except Exception:
        return None


def chat(client, messages: list[dict]) -> str:
    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        return response.content[0].text
    except Exception as e:
        return f"Error: {e}"


def init_history():
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []


def add_message(role: str, content: str):
    st.session_state["chat_history"].append({"role": role, "content": content})


def clear_history():
    st.session_state["chat_history"] = []
