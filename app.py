import streamlit as st
import yfinance as yf
import google.generativeai as genai
import plotly.graph_objects as go
from duckduckgo_search import DDGS
import pandas as pd

# ---------------------------------------------------------
# 1. Configuration & Setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Stock Sniper Ultimate 🚀",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: bold;
    }
    /* ปรับแต่ง Table ให้สวยงาม */
    .stDataFrame { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. Advanced Data Fetching (Auto-List)
# ---------------------------------------------------------

@st.cache_data(ttl=86400) # Cache ไว้ 24 ชม. ไม่ต้องโหลดใหม่บ่อยๆ
def get_sp500_tickers():
    """ดูดรายชื่อหุ้น S&P 500 ทั้งหมดจาก Wikipedia อัตโนมัติ"""
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        tables = pd.read_html(url)
        df = tables[0]
        # สร้าง Dictionary {SYMBOL: Name}
        tickers = dict(zip(df.Symbol, df.Security))
        return tickers
    except Exception as e:
        # ถ้าเว็บล่ม ให้ใช้รายชื่อสำรอง
        return {
            "AAPL": "Apple Inc.", "TSLA": "Tesla, Inc.", "NVDA": "NVIDIA Corp.",
            "AMD": "Advanced Micro Devices", "MSFT": "Microsoft Corp.",
            "GOOGL": "Alphabet Inc.", "AMZN": "Amazon.com", "META": "Meta Platforms"
        }

# โหลดรายชื่อหุ้นรอไว้เลย
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
    """ดึงข้อมูลราคา + ปรับ Period อัตโนมัติ"""
    period_map = {
        "1m": "1d", "5m": "5d", "15m": "1mo", 
        "30m": "1mo", "1h": "3mo", "1d": "1y", "1wk": "2y"
    }
    period = period_map.get(interval, "1mo")
    
    stock = yf.Ticker(symbol)
    history = stock.history(period=period, interval=interval)
    info = stock.info
    return history, info

def get_latest_news(symbol):
    """ดึงข่าวจาก DuckDuckGo + Cache"""
    if symbol in st.session_state.news_cache:
        return st.session_state.news_cache[symbol]

    try:
        formatted_news = []
        with DDGS() as ddgs:
            # ค้นหาเจาะจงข่าว Finance
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
        # ใช้รุ่น Flash เพื่อความไว (ถ้ามีสิทธิ์ใช้ 2.5 ก็จะใช้ได้)
        model = genai.GenerativeModel('models/gemini-2.5-flash') 
        
        prompt = f"""
        Role: Expert Stock Analyst
        Symbol: {symbol} | Price: ${current_price:.2f}
        News: {news_text}
        
        Output (Thai Language, Bullet points):
        1. 📰 **สรุปข่าว:** (สั้นๆ ได้ใจความ)
        2. 🚦 **Sentiment:** (Bullish/Bearish/Neutral)
        3. 🎯 **Impact:** (ผลกระทบระยะสั้น)
        4. 🛡️ **Levels:** (แนวรับ-แนวต้าน จิตวิทยา)
        5. 💡 **Action:** (Wait / Buy / Sell พร้อมเหตุผล)
        """
        response = model.generate_content(prompt)
        st.session_state.analysis_cache[symbol] = response.text
        return response.text
    except Exception as e:
        return f"AI Error: {str(e)}"

# ---------------------------------------------------------
# 5. Sidebar UI (The Control Center)
# ---------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Control Panel")
    
    # API Key
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
        st.success("✅ Connected to System Key")
    else:
        api_key = st.text_input("🔑 Gemini API Key:", type="password")
    
    st.divider()
    
    # --- ส่วนค้นหาหุ้นแบบ Hybrid ---
    st.subheader("🔍 เพิ่มหุ้นลง Watchlist")
    
    # Tab 1: เลือกจาก S&P 500 (Dropdown)
    # Tab 2: พิมพ์เอง (Manual)
    tab1, tab2 = st.tabs(["List S&P500", "Custom Search"])
    
    with tab1:
        # แปลง Dict ให้เป็น List สวยๆ สำหรับ Search
        sp500_options = [f"{sym} - {name}" for sym, name in SP500_TICKERS.items()]
        selected_sp500 = st.selectbox("เลือกหุ้น S&P 500:", [""] + sp500_options)
        
        if selected_sp500:
            ticker = selected_sp500.split(" - ")[0]
            if st.button(f"➕ เพิ่ม {ticker}"):
                if ticker not in st.session_state.watchlist:
                    st.session_state.watchlist.append(ticker)
                    st.rerun()

    with tab2:
        custom_ticker = st.text_input("พิมพ์ชื่อย่อหุ้น (เช่น PLTR, COIN):").upper()
        if st.button("➕ เพิ่มหุ้น Custom"):
            if custom_ticker and custom_ticker not in st.session_state.watchlist:
                # ลองเช็คก่อนว่าหุ้นมีจริงไหม
                check = yf.Ticker(custom_ticker)
                try:
                    if check.info: # ถ้าดึง info ได้แปลว่ามีจริง
                        st.session_state.watchlist.append(custom_ticker)
                        st.rerun()
                except:
                    st.error("❌ ไม่พบข้อมูลหุ้นนี้")

    st.divider()
    
    # --- Watchlist Management ---
    st.subheader("👀 My Watchlist")
    
    if st.session_state.watchlist:
        # ใช้ Multiselect เพื่อให้ดูรายการทั้งหมดง่ายๆ
        # แต่เวลาเลือกดูใช้ Radio หรือ Selectbox แยก
        target_stock = st.radio("เลือกหุ้นที่ต้องการวิเคราะห์:", st.session_state.watchlist)
        
        col_del, col_clr = st.columns(2)
        with col_del:
            if st.button("❌ ลบตัวที่เลือก"):
                st.session_state.watchlist.remove(target_stock)
                st.rerun()
        with col_clr:
            if st.button("🗑️ ล้างทั้งหมด"):
                st.session_state.watchlist = []
                st.rerun()
    else:
        st.info("Watchlist ว่างเปล่า")
        target_stock = None

# ---------------------------------------------------------
# 6. Main Dashboard
# ---------------------------------------------------------
if target_stock:
    # Header
    st.title(f"🚀 {target_stock} Analysis Dashboard")
    
    # Timeframe Selector
    c_time, c_blank = st.columns([2, 5])
    with c_time:
        time_option = st.selectbox("⏳ Timeframe:", 
            ["1 Minute", "5 Minutes", "15 Minutes", "30 Minutes", "1 Hour", "1 Day", "1 Week"], index=1)
    
    # Map selection to interval
    interval_mapping = {
        "1 Minute": "1m", "5 Minutes": "5m", "15 Minutes": "15m", 
        "30 Minutes": "30m", "1 Hour": "1h", "1 Day": "1d", "1 Week": "1wk"
    }
    interval = interval_mapping[time_option]

    with st.spinner(f"Fetching {target_stock} data..."):
        try:
            hist, info = get_stock_data(target_stock, interval)
            
            if hist.empty:
                st.error("❌ ไม่พบข้อมูลราคาตลาด (Market Closed or Invalid Data)")
            else:
                # --- Price Banner ---
                curr_price = hist['Close'].iloc[-1]
                try:
                    prev_price = hist['Open'].iloc[0] # เทียบกับราคาเปิดของช่วงนั้น
                    delta = curr_price - prev_price
                    pct = (delta / prev_price) * 100
                except:
                    delta, pct = 0, 0
                
                # แสดงชื่อเต็มบริษัท
                long_name = info.get('longName', target_stock)
                st.caption(f"Company: {long_name}")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Price", f"${curr_price:.2f}", f"{delta:.2f} ({pct:.2f}%)")
                m2.metric("High", f"${hist['High'].max():.2f}")
                m3.metric("Low", f"${hist['Low'].min():.2f}")
                m4.metric("Volume", f"{hist['Volume'].sum():,}")

                # --- Graph ---
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=hist.index,
                    open=hist['Open'], high=hist['High'],
                    low=hist['Low'], close=hist['Close'],
                    name='Price'
                ))
                fig.update_layout(
                    title=f'{target_stock} ({time_option})',
                    height=550,
                    template="plotly_dark",
                    xaxis_rangeslider_visible=False,
                    margin=dict(t=30, b=0, l=0, r=0)
                )
                st.plotly_chart(fig, use_container_width=True)

                # --- AI & News Section ---
                st.markdown("---")
                
                # Auto-fetch news
                news_content = get_latest_news(target_stock)
                
                c_left, c_right = st.columns([1, 1])
                
                with c_right:
                    st.subheader(f"📰 News: {target_stock}")
                    if "ไม่พบข่าว" in news_content:
                        st.warning(news_content)
                    else:
                        st.markdown(news_content)

                with c_left:
                    st.subheader("🤖 AI Analyst Insight")
                    
                    # Caching Check
                    cached_result = st.session_state.analysis_cache.get(target_stock)
                    
                    if cached_result:
                        st.success("💡 Analysis Cached")
                        st.markdown(cached_result)
                        if st.button("🔄 Force Re-Analyze"):
                            del st.session_state.analysis_cache[target_stock]
                            st.rerun()
                    else:
                        if st.button("⚡ Start AI Analysis", type="primary"):
                            with st.spinner("AI is thinking..."):
                                analysis = ai_analyze(news_content, curr_price, target_stock, api_key)
                                st.markdown(analysis)

        except Exception as e:
            st.error(f"System Error: {str(e)}")
else:
    st.info("👈 กรุณาเลือกหุ้นจากเมนูด้านซ้ายเพื่อเริ่มต้น")
