import streamlit as st
import yfinance as yf
import google.generativeai as genai
import plotly.graph_objects as go
from duckduckgo_search import DDGS
import pandas as pd
import requests

# 1. Configuration
st.set_page_config(page_title="Stock Sniper Pro (TradingView Style)", layout="wide")

st.markdown("""
<style>
    div[data-testid="stPills"] { gap: 10px; justify-content: flex-start; }
    .stButton>button { border-radius: 8px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# 2. Fetch S&P 500 Tickers
@st.cache_data(ttl=86400)
def get_sp500_tickers():
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        df = pd.read_html(response.text)[0]
        return dict(zip(df.Symbol, df.Security))
    except:
        return {"AAPL": "Apple", "TSLA": "Tesla", "NVDA": "NVIDIA"}

SP500_TICKERS = get_sp500_tickers()

# 3. Session State
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ['AAPL', 'TSLA', 'NVDA', 'GME']
if 'analysis_cache' not in st.session_state:
    st.session_state.analysis_cache = {}

# 4. Logic Functions
def get_stock_data(symbol, interval):
    period_map = {
        "1m": "1d", "5m": "5d", "15m": "1mo", "30m": "1mo", 
        "1h": "3mo", "1d": "1y", "1wk": "2y", "1mo": "5y",
        "YTD": "ytd", "1Y": "1y", "5Y": "5y"
    }
    actual_interval = "1d" if interval in ["YTD", "1Y", "5Y"] else interval
    period = period_map.get(interval, "1mo")
    
    stock = yf.Ticker(symbol)
    df = stock.history(period=period, interval=actual_interval)
    return df

def get_latest_news(symbol):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(f"{symbol} stock financial news", max_results=5))
            return "\n".join([f"- [{n['title']}]({n['href']})" for n in results]) if results else "ไม่พบข่าว"
    except:
        return "Error loading news"

def ai_analyze(news, price, symbol, key):
    if not key: return "กรุณาใส่ API Key"
    try:
        genai.configure(api_key=key)
        model = genai.GenerativeModel('models/gemini-2.0-flash')
        prompt = f"Expert Analyst. {symbol} at ${price:.2f}. News: {news}. Provide Sentiment, Levels, Action in Thai."
        return model.generate_content(prompt).text
    except Exception as e:
        return f"AI Error: {e}"

# 5. Sidebar
with st.sidebar:
    st.title("⚙️ Dashboard Settings")
    api_key = st.secrets["GEMINI_API_KEY"] if "GEMINI_API_KEY" in st.secrets else st.text_input("🔑 API Key:", type="password")
    
    st.divider()
    tab1, tab2 = st.tabs(["S&P 500", "Custom"])
    with tab1:
        selected_sp = st.selectbox("Select Stock:", [""] + [f"{k} - {v}" for k, v in SP500_TICKERS.items()])
        if selected_sp and st.button("Add S&P"):
            t = selected_sp.split(" - ")[0]
            if t not in st.session_state.watchlist: st.session_state.watchlist.append(t)
    with tab2:
        custom = st.text_input("Ticker:").upper()
        if st.button("Add Custom") and custom:
            if not yf.Ticker(custom).history(period="1d").empty:
                if custom not in st.session_state.watchlist: st.session_state.watchlist.append(custom)
    
    st.divider()
    target_stock = st.radio("My Watchlist:", st.session_state.watchlist)
    if st.button("❌ Remove Selected") and target_stock:
        st.session_state.watchlist.remove(target_stock)
        st.rerun()

# 6. Main UI
if target_stock:
    st.title(f"🚀 {target_stock} Trading Terminal")
    
    # 6.1 Stats Header
    data_raw = yf.Ticker(target_stock).history(period="2d")
    if not data_raw.empty:
        curr, prev = data_raw['Close'].iloc[-1], data_raw['Close'].iloc[-2]
        diff = curr - prev
        pct = (diff / prev) * 100
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Current Price", f"${curr:.2f}", f"{diff:.2f} ({pct:.2f}%)")
        m2.metric("High", f"${data_raw['High'].max():.2f}")
        m3.metric("Low", f"${data_raw['Low'].min():.2f}")
        m4.metric("Vol", f"{data_raw['Volume'].iloc[-1]:,}")

    # 6.2 Timeframe Pills
    interval = st.pills("Timeframe:", ["1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo", "YTD", "1Y", "5Y"], default="1h")

    # 6.3 Drawing The Chart (The TradingView Way)
    hist = get_stock_data(target_stock, interval)
    if not hist.empty:
        fig = go.Figure(data=[go.Candlestick(
            x=hist.index, open=hist['Open'], high=hist['High'],
            low=hist['Low'], close=hist['Close'], name='Price'
        )])

        # --- สำคัญ: ตัดช่วงเวลาที่ไม่มีการซื้อขายออก ---
        fig.update_xaxes(
            rangebreaks=[
                dict(bounds=["sat", "mon"]), # ตัดวันเสาร์-อาทิตย์
                dict(bounds=[16, 9.5], pattern="hour"), # ตัดช่วงเวลาหลังตลาดปิด (16:00 - 09:30)
            ]
        )

        fig.update_layout(
            height=600, template="plotly_dark",
            xaxis_rangeslider_visible=False,
            margin=dict(t=10, b=10, l=10, r=10),
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

        # 6.4 AI & News
        st.divider()
        c_left, c_right = st.columns(2)
        news = get_latest_news(target_stock)
        with c_right:
            st.subheader("📰 Latest News")
            st.markdown(news)
        with c_left:
            st.subheader("🤖 AI Insight")
            if st.button("⚡ Analyze Now", type="primary"):
                st.markdown(ai_analyze(news, curr, target_stock, api_key))
else:
    st.info("Select a stock to start.")
