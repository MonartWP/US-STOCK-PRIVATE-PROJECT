import streamlit as st
import yfinance as yf
import google.generativeai as genai
import plotly.graph_objects as go
from duckduckgo_search import DDGS
import pandas as pd
import requests

# ---------------------------------------------------------
# 1. Configuration & Setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Stock Sniper Ultimate 🚀",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: bold;
    }
    div[data-testid="stPills"] {
        gap: 10px;
        justify-content: center;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. Advanced Data Fetching (Auto-List)
# ---------------------------------------------------------
@st.cache_data(ttl=86400)
def get_sp500_tickers():
    """ดูดรายชื่อหุ้น S&P 500 แบบปลอมตัวเป็น Browser"""
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        tables = pd.read_html(response.text)
        df = tables[0]
        tickers = dict(zip(df.Symbol, df.Security))
        return tickers
    except Exception as e:
        return {
            "AAPL": "Apple Inc.", "TSLA": "Tesla, Inc.", "NVDA": "NVIDIA Corp.",
            "AMD": "Advanced Micro Devices", "MSFT": "Microsoft Corp.",
            "GOOGL": "Alphabet Inc.", "AMZN": "Amazon.com", "META": "Meta Platforms"
        }

# เรียกใช้งานฟังก์ชันเพื่อเก็บค่าลงตัวแปร (บรรทัดที่เคยหายไป)
SP500_TICKERS = get_sp500_tickers()

# ---------------------------------------------------------
# 3. Session State Management
# ---------------------------------------------------------
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ['AAPL', 'TSLA', 'NVDA', 'GME']
if 'analysis_cache' not in st.session_state:
    st.session_state.analysis_cache = {}
if 'news_cache' not in st.session_state:
    st.session_state.news_cache = {}

# ---------------------------------------------------------
# 4. Backend Logic
# ---------------------------------------------------------
def get_stock_data(symbol, interval):
    """ดึงข้อมูลราคา + ปรับ Period ให้สอดคล้องกับ Timeframe"""
    # Mapping ช่วงเวลาข้อมูล (Period) ให้สัมพันธ์กับปุ่มที่กด (Interval)
    # 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
    period_map = {
        "1m": "1d", "5m": "5d", "15m": "1mo", "30m": "1mo", 
        "1h": "3mo", "1d": "1y", "1wk": "2y", "1mo": "5y",
        "YTD": "ytd", "1Y": "1y", "5Y": "5y"
    }
    
    # สำหรับ YTD, 1Y, 5Y เราจะใช้ Interval เป็น '1d' เพื่อให้เห็นรายละเอียดแท่งเทียนรายวัน
    actual_interval = interval
    if interval in ["YTD", "1Y", "5Y"]:
        actual_interval = "1d"
        
    period = period_map.get(interval, "1mo")
    
    stock = yf.Ticker(symbol)
    history = stock.history(period=period, interval=actual_interval)
    return history

def get_latest_news(symbol):
    """ดึงข่าวจาก DuckDuckGo"""
    if symbol in st.session_state.news_cache:
        return st.session_state.news_cache[symbol]
    try:
        formatted_news = []
        with DDGS() as ddgs:
            results = list(ddgs.text(f"{symbol} stock financial news", max_results=5))
            if results:
                for news in results:
                    title = news.get('title')
                    link = news.get('href')
                    if title and link:
                        formatted_news.append(f"- [{title}]({link})")
        result_text = "\n".join(formatted_news) if formatted_news else "ไม่พบข่าวใหม่ในขณะนี้"
        st.session_state.news_cache[symbol] = result_text
        return result_text
    except Exception as e:
        return f"News Error: {str(e)}"

def ai_analyze(news_text, current_price, symbol, api_key):
    """AI Analysis"""
    if symbol in st.session_state.analysis_cache:
        return st.session_state.analysis_cache[symbol]
    if not api_key:
        return "⚠️ กรุณาใส่ API Key ก่อนครับ"
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('models/gemini-2.0-flash') 
        prompt = f"""Expert Stock Analyst Role. Symbol: {symbol}, Price: ${current_price:.2f}. News: {news_text}. Output in Thai with Sentiment, Impact, Levels, and Action."""
        response = model.generate_content(prompt)
        st.session_state.analysis_cache[symbol] = response.text
        return response.text
    except Exception as e:
        return f"AI Error: {str(e)}"

# ---------------------------------------------------------
# 5. Sidebar UI
# ---------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Control Panel")
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
    else:
        api_key = st.text_input("🔑 Gemini API Key:", type="password")
    
    st.divider()
    st.subheader("🔍 เพิ่มหุ้นลง Watchlist")
    tab1, tab2 = st.tabs(["List S&P500", "Custom Search"])
    
    with tab1:
        sp500_options = [f"{sym} - {name}" for sym, name in SP500_TICKERS.items()]
        selected_sp500 = st.selectbox("เลือกหุ้น S&P 500:", [""] + sp500_options)
        if selected_sp500:
            ticker = selected_sp500.split(" - ")[0]
            if st.button(f"➕ เพิ่ม {ticker}"):
                if ticker not in st.session_state.watchlist:
                    st.session_state.watchlist.append(ticker)
                    st.rerun()

    with tab2:
        custom_ticker = st.text_input("พิมพ์ชื่อย่อหุ้น:").upper()
        if st.button("➕ เพิ่มหุ้น Custom"):
            if custom_ticker:
                with st.spinner(f"กำลังตรวจสอบ {custom_ticker}..."):
                    check_stock = yf.Ticker(custom_ticker)
                    check_hist = check_stock.history(period="1d")
                    if not check_hist.empty:
                        if custom_ticker not in st.session_state.watchlist:
                            st.session_state.watchlist.append(custom_ticker)
                            st.rerun()
                    else:
                        st.error(f"❌ ไม่พบข้อมูลหุ้น: {custom_ticker}")

    st.divider()
    st.subheader("👀 My Watchlist")
    if st.session_state.watchlist:
        target_stock = st.radio("เลือกหุ้น:", st.session_state.watchlist)
        if st.button("❌ ลบตัวที่เลือก"):
            st.session_state.watchlist.remove(target_stock)
            st.rerun()
    else:
        target_stock = None

# ---------------------------------------------------------
# 6. Main Dashboard
# ---------------------------------------------------------
if target_stock:
    st.title(f"🚀 {target_stock} Analysis Dashboard")
    with st.spinner(f"Loading {target_stock}..."):
        try:
            temp_stock = yf.Ticker(target_stock)
            temp_hist = temp_stock.history(period="5d")
            
            if not temp_hist.empty:
                curr_price = temp_hist['Close'].iloc[-1]
                prev_price = temp_hist['Close'].iloc[-2] if len(temp_hist) > 1 else temp_hist['Open'].iloc[0]
                delta = curr_price - prev_price
                pct = (delta / prev_price) * 100
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Price", f"${curr_price:.2f}", f"{delta:.2f} ({pct:.2f}%)")
                m2.metric("Previous Close", f"${prev_price:.2f}")
                m3.metric("Day High", f"${temp_hist['High'].iloc[-1]:.2f}")
                m4.metric("Day Low", f"${temp_hist['Low'].iloc[-1]:.2f}")
            
            st.markdown("---")

            # --- เพิ่ม YTD, 1Y, 5Y ในปุ่ม Pills ---
            col_pills, _ = st.columns([3, 1])
            with col_pills:
                interval = st.pills("Timeframe:", 
                                   ["1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo", "YTD", "1Y", "5Y"], 
                                   default="5m")

            hist = get_stock_data(target_stock, interval)
            
            if hist.empty:
                st.error(f"❌ ไม่พบข้อมูลสำหรับช่วงเวลา {interval}")
            else:
                fig = go.Figure(data=[go.Candlestick(
                    x=hist.index, open=hist['Open'], high=hist['High'],
                    low=hist['Low'], close=hist['Close'], name='Price'
                )])
                fig.update_layout(title=f'{target_stock} ({interval})', height=500, template="plotly_dark", xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)

                st.divider()
                news_content = get_latest_news(target_stock)
                c_left, c_right = st.columns(2)
                with c_right:
                    st.subheader("📰 Latest News")
                    st.markdown(news_content)
                with c_left:
                    st.subheader("🤖 AI Analysis")
                    if st.button("⚡ Start AI Analysis", type="primary"):
                        analysis = ai_analyze(news_content, curr_price, target_stock, api_key)
                        st.markdown(analysis)

        except Exception as e:
            st.error(f"Error: {str(e)}")
