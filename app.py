import streamlit as st
import yfinance as yf
import google.generativeai as genai
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_gsheets import GSheetsConnection
from duckduckgo_search import DDGS
import pandas as pd
import requests

# 1. Setup
st.set_page_config(page_title="AI Stock Terminal Pro", layout="wide")

# เชื่อมต่อ Google Sheets (ทำหน้าที่เป็น Profile เก็บข้อมูล)
conn = st.connection("gsheets", type=GSheetsConnection)

def get_sp500():
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        return dict(zip(pd.read_html(res.text)[0].Symbol, pd.read_html(res.text)[0].Security))
    except: return {"AAPL": "Apple", "TSLA": "Tesla"}

SP500 = get_sp500()

# 2. Profile Logic (ดึง/เซฟ Watchlist ลง Sheets)
def sync_watchlist(action, ticker=None):
    # อ่านข้อมูลปัจจุบัน
    try:
        data = conn.read(worksheet="Sheet1", usecols=[0])
        current_list = data.iloc[:, 0].tolist()
    except: current_list = []

    if action == "add" and ticker not in current_list:
        current_list.append(ticker)
    elif action == "remove" and ticker in current_list:
        current_list.remove(ticker)
    
    # บันทึกกลับลง Sheets (Profile ของเรา)
    new_df = pd.DataFrame(current_list, columns=["symbol"])
    conn.update(worksheet="Sheet1", data=new_df)
    return current_list

# โหลดข้อมูล Profile ตอนเริ่ม
if 'watchlist' not in st.session_state:
    try: st.session_state.watchlist = sync_watchlist("read")
    except: st.session_state.watchlist = ["AAPL", "NVDA"]

# 3. Sidebar Profile & Indicators
with st.sidebar:
    st.title("👤 My Profile")
    st.info(f"Watchlist synced with Google Sheets")
    
    st.divider()
    st.subheader("🛠️ Indicators")
    show_ema = st.toggle("เปิดเส้น EMA", value=True)
    ema_vals = st.multiselect("เลือกช่วงเวลา EMA:", [20, 50, 100, 200], default=[20, 50])
    
    st.divider()
    st.subheader("🔍 Add Stock")
    new_s = st.selectbox("S&P 500:", [""] + [f"{k}-{v}" for k,v in SP500.items()])
    if st.button("Add to Profile") and new_s:
        ticker = new_s.split("-")[0]
        st.session_state.watchlist = sync_watchlist("add", ticker)
        st.rerun()

    st.divider()
    target = st.radio("เลือกหุ้นวิเคราะห์:", st.session_state.watchlist)
    if st.button("🗑️ ลบจาก Profile"):
        st.session_state.watchlist = sync_watchlist("remove", target)
        st.rerun()

# 4. Dashboard Core
if target:
    st.title(f"🚀 {target} Terminal")
    
    # Timeframe Pills
    tf = st.pills("Timeframe:", ["1m", "5m", "15m", "1h", "1d", "1wk", "YTD", "1Y", "5Y"], default="1h")
    
    # Fetch Data
    p_map = {"1m":"1d","5m":"5d","15m":"1mo","1h":"3mo","1d":"1y","1wk":"2y","YTD":"ytd","1Y":"1y","5Y":"5y"}
    interval = "1d" if tf in ["YTD", "1Y", "5Y"] else tf
    hist = yf.Ticker(target).history(period=p_map[tf], interval=interval)

    if not hist.empty:
        # กราฟราคา + Volume แบบ TradingView
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
        
        # Candlestick
        fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name="Price"), row=1, col=1)
        
        # EMA (Dynamic)
        if show_ema:
            colors = ['blue', 'orange', 'red', 'green']
            for i, val in enumerate(ema_vals):
                ema = hist['Close'].ewm(span=val).mean()
                fig.add_trace(go.Scatter(x=hist.index, y=ema, name=f"EMA {val}", line=dict(width=1, color=colors[i%4])), row=1, col=1)

        # Volume
        fig.add_trace(go.Bar(x=hist.index, y=hist['Volume'], name="Volume", opacity=0.3), row=2, col=1)

        # ตัดช่องว่างตลาดปิด (TradingView Style)
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"]), dict(bounds=[16, 9.5], pattern="hour") if "m" in tf or "h" in tf else None])
        fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    # 5. AI Analyst (มีระบบกัน Error 429)
    st.divider()
    if st.button("⚡ Run AI Strategic Analysis", type="primary"):
        api_key = st.secrets.get("GEMINI_API_KEY")
        if not api_key: st.error("ไม่พบ API Key")
        else:
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('models/gemini-2.0-flash')
                res = model.generate_content(f"Analyze {target}. Current: {hist['Close'].iloc[-1]}. Output Thai.")
                st.write(res.text)
            except Exception as e:
                if "429" in str(e): st.warning("⚠️ โควต้า AI รายนาทีหมดครับ รอ 1 นาทีแล้วกดใหม่นะ")
                else: st.error(f"Error: {e}")

else: st.info("เพิ่มหุ้นลง Profile ที่แถบด้านซ้ายได้เลยครับ")
