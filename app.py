import streamlit as st
import yfinance as yf
import google.generativeai as genai
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from duckduckgo_search import DDGS
import pandas as pd
import requests

# 1. Configuration
st.set_page_config(page_title="AI Stock Sniper Elite 🚀", layout="wide")

# 2. Fetch S&P 500 Tickers (Stable Version)
@st.cache_data(ttl=86400)
def get_sp500_tickers():
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        df = pd.read_html(response.text)[0]
        return dict(zip(df.Symbol, df.Security))
    except:
        return {"AAPL": "Apple", "TSLA": "Tesla", "NVDA": "NVIDIA", "MSFT": "Microsoft"}

SP500_TICKERS = get_sp500_tickers()

# 3. Session State Initial
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ['AAPL', 'TSLA', 'NVDA']

# 4. Sidebar Controls
with st.sidebar:
    st.title("🛡️ Pro Terminal Settings")
    
    # API Key Management
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
        st.success("✅ System API Key Active")
    else:
        api_key = st.text_input("🔑 Gemini API Key:", type="password")
    
    st.divider()
    
    # Indicator Controls (ฟังก์ชันที่คุณขอมา)
    st.subheader("📈 Technical Indicators")
    show_ema = st.toggle("Show EMA Lines", value=True)
    ema_1 = st.number_input("EMA 1 Period:", value=20, min_value=1)
    ema_2 = st.number_input("EMA 2 Period:", value=50, min_value=1)
    ema_3 = st.number_input("EMA 3 Period:", value=200, min_value=1)
    
    st.divider()
    
    # Watchlist Management
    st.subheader("🔍 Add Stocks")
    tab_sp, tab_custom = st.tabs(["S&P 500", "Custom"])
    with tab_sp:
        selected = st.selectbox("Choose Stock:", [""] + [f"{k} - {v}" for k, v in SP500_TICKERS.items()])
        if st.button("Add to Watchlist") and selected:
            t = selected.split(" - ")[0]
            if t not in st.session_state.watchlist: st.session_state.watchlist.append(t)
    with tab_custom:
        custom = st.text_input("Ticker (e.g. BTC-USD):").upper()
        if st.button("Add Custom") and custom:
            st.session_state.watchlist.append(custom)
    
    st.divider()
    target_stock = st.radio("Current Watchlist:", st.session_state.watchlist)
    if st.button("🗑️ Remove Selected"):
        st.session_state.watchlist.remove(target_stock)
        st.rerun()

# 5. Logic Functions
def get_data(symbol, interval):
    period_map = {"1m":"1d","5m":"5d","15m":"1mo","30m":"1mo","1h":"3mo","1d":"1y","1wk":"2y","1mo":"5y","YTD":"ytd","1Y":"1y","5Y":"5y"}
    p = period_map.get(interval, "1mo")
    i = "1d" if interval in ["YTD", "1Y", "5Y"] else interval
    df = yf.Ticker(symbol).history(period=p, interval=i)
    return df

# 6. Main Dashboard
if target_stock:
    st.title(f"📊 {target_stock} Ultimate Terminal")
    
    # 6.1 Stats Header
    raw = yf.Ticker(target_stock).history(period="2d")
    if not raw.empty:
        curr = raw['Close'].iloc[-1]
        change = curr - raw['Close'].iloc[-2]
        pct = (change / raw['Close'].iloc[-2]) * 100
        cols = st.columns(4)
        cols[0].metric("Price", f"${curr:.2f}", f"{change:.2f} ({pct:.2f}%)")
        cols[1].metric("Day High", f"${raw['High'].max():.2f}")
        cols[2].metric("Day Low", f"${raw['Low'].min():.2f}")
        cols[3].metric("Volume", f"{raw['Volume'].iloc[-1]:,.0f}")

    # 6.2 Timeframe Selector
    interval = st.pills("Select Timeframe:", ["1m", "5m", "15m", "1h", "1d", "1wk", "1mo", "YTD", "1Y", "5Y"], default="1h")

    # 6.3 Advance Plotting
    hist = get_data(target_stock, interval)
    if not hist.empty:
        # สร้าง Subplots: กราฟราคาด้านบน (70%) กราฟ Volume ด้านล่าง (30%)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.03, row_heights=[0.7, 0.3])

        # 1. Candlestick
        fig.add_trace(go.Candlestick(
            x=hist.index, open=hist['Open'], high=hist['High'],
            low=hist['Low'], close=hist['Close'], name='Price'
        ), row=1, col=1)

        # 2. Add EMA Lines (เปิด-ปิดได้)
        if show_ema:
            for period, color in zip([ema_1, ema_2, ema_3], ['#2962FF', '#FF9800', '#F44336']):
                ema_data = hist['Close'].ewm(span=period, adjust=False).mean()
                fig.add_trace(go.Scatter(x=hist.index, y=ema_data, name=f'EMA {period}', 
                                         line=dict(width=1.5, color=color)), row=1, col=1)

        # 3. Volume Bar
        colors = ['#26a69a' if c >= o else '#ef5350' for o, c in zip(hist['Open'], hist['Close'])]
        fig.add_trace(go.Bar(x=hist.index, y=hist['Volume'], name='Volume', 
                             marker_color=colors, opacity=0.5), row=2, col=1)

        # 4. TradingView Style X-Axis (Fixed!)
        fig.update_xaxes(
            rangebreaks=[
                dict(bounds=["sat", "mon"]), # ตัดวันเสาร์-อาทิตย์
                dict(bounds=[16, 9.5], pattern="hour") if interval not in ["1d", "1wk", "1mo", "YTD", "1Y", "5Y"] else None
            ]
        )

        fig.update_layout(height=700, template="plotly_dark", xaxis_rangeslider_visible=False,
                          margin=dict(t=30, b=10, l=10, r=10), hovermode="x unified")
        
        st.plotly_chart(fig, use_container_width=True)

    # 6.4 AI & News Section
    st.divider()
    l, r = st.columns(2)
    with r:
        st.subheader("📰 Market Intelligence")
        try:
            with DDGS() as ddgs:
                news_list = list(ddgs.text(f"{target_stock} stock market news", max_results=5))
                news_txt = "\n".join([f"- [{n['title']}]({n['href']})" for n in news_list])
                st.markdown(news_txt if news_list else "No recent news found.")
        except:
            st.warning("News service temporary unavailable")
            news_txt = "No news data"
            
    with l:
        st.subheader("🤖 AI Analyst Report")
        if st.button("🚀 Run AI Tactical Analysis", type="primary"):
            if not api_key:
                st.error("Please provide Gemini API Key in sidebar")
            else:
                with st.spinner("AI analyzing chart patterns and news..."):
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('models/gemini-2.0-flash')
                    prompt = f"Analyze {target_stock} at price ${curr:.2f}. Consider this news: {news_txt}. Output in Thai: Sentiment, Key Levels, and Tactical Action (Buy/Sell/Hold)."
                    st.markdown(model.generate_content(prompt).text)

else:
    st.info("👈 Please select or add a stock from the sidebar to begin analysis.")
