import streamlit as st
import yfinance as yf
import google.generativeai as genai
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import requests

# 1. Configuration
st.set_page_config(page_title="AI Multi-Portfolio Terminal", layout="wide")

# เชื่อมต่อ Google Sheets
conn = st.connection("gsheets", type=GSheetsConnection)

@st.cache_data(ttl=86400)
def get_sp500():
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        df = pd.read_html(res.text)[0]
        return dict(zip(df.Symbol, df.Security))
    except: return {"AAPL": "Apple", "TSLA": "Tesla", "NVDA": "NVIDIA"}

SP500 = get_sp500()

# 2. Portfolio Logic (ระบบดึงและเซฟแยกแผ่น)
def sync_data(portfolio_name, action, ticker=None):
    try:
        # อ่านข้อมูลจาก Sheet ที่ชื่อตามพอร์ต (Dime หรือ Webull)
        df = conn.read(worksheet=portfolio_name, usecols=[0])
        current_list = df.iloc[:, 0].dropna().tolist()
    except:
        current_list = []

    if action == "add" and ticker and ticker not in current_list:
        current_list.append(ticker)
        new_df = pd.DataFrame(current_list, columns=["symbol"])
        conn.update(worksheet=portfolio_name, data=new_df)
    elif action == "remove" and ticker in current_list:
        current_list.remove(ticker)
        new_df = pd.DataFrame(current_list, columns=["symbol"])
        conn.update(worksheet=portfolio_name, data=new_df)
    
    return current_list

# 3. Sidebar: Portfolio Selection & Settings
with st.sidebar:
    st.title("🏦 My Portfolios")
    
    # เลือกพอร์ตที่จะดู
    selected_port = st.selectbox("เลือกพอร์ตที่ต้องการ:", ["Dime", "Webull"])
    
    # โหลด Watchlist ตามพอร์ตที่เลือก
    watchlist = sync_data(selected_port, "read")
    
    st.divider()
    st.subheader("📈 Indicator Settings")
    show_ema = st.toggle("เปิดเส้น EMA", value=True)
    ema_vals = st.multiselect("EMA Periods:", [20, 50, 100, 200], default=[20, 50])
    
    st.divider()
    st.subheader(f"➕ เพิ่มหุ้นใน {selected_port}")
    new_stock = st.selectbox("เลือกจาก S&P 500:", [""] + [f"{k} - {v}" for k,v in SP500.items()])
    if st.button("บันทึกลงพอร์ต") and new_stock:
        symbol = new_stock.split(" - ")[0]
        watchlist = sync_data(selected_port, "add", symbol)
        st.rerun()
        
    custom_stock = st.text_input("หรือพิมพ์ชื่อหุ้นเอง (เช่น TSLA):").upper()
    if st.button("เพิ่มหุ้น Custom") and custom_stock:
        watchlist = sync_data(selected_port, "add", custom_stock)
        st.rerun()

    st.divider()
    if watchlist:
        target = st.radio(f"หุ้นใน {selected_port}:", watchlist)
        if st.button("🗑️ ลบหุ้นนี้ออก"):
            sync_data(selected_port, "remove", target)
            st.rerun()
    else:
        target = None
        st.info("พอร์ตนี้ยังไม่มีหุ้น กรุณาเพิ่มหุ้นครับ")

# 4. Main Terminal
if target:
    st.title(f"🚀 {target} @ {selected_port} Portfolio")
    
    # Timeframe
    tf = st.pills("ช่วงเวลา:", ["1m", "5m", "15m", "1h", "1d", "1wk", "YTD", "1Y", "5Y"], default="1h")
    
    # Data Fetching
    p_map = {"1m":"1d","5m":"5d","15m":"1mo","1h":"3mo","1d":"1y","1wk":"2y","YTD":"ytd","1Y":"1y","5Y":"5y"}
    actual_i = "1d" if tf in ["YTD", "1Y", "5Y"] else tf
    hist = yf.Ticker(target).history(period=p_map[tf], interval=actual_i)

    if not hist.empty:
        # กราฟราคาแบบ TradingView (Subplots)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
        
        # แท่งเทียน
        fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name="Price"), row=1, col=1)
        
        # EMA
        if show_ema:
            colors = ['#2962FF', '#FF9800', '#F44336', '#4CAF50']
            for i, v in enumerate(ema_vals):
                ema = hist['Close'].ewm(span=v).mean()
                fig.add_trace(go.Scatter(x=hist.index, y=ema, name=f"EMA {v}", line=dict(width=1.2, color=colors[i%4])), row=1, col=1)

        # Volume
        v_colors = ['#26a69a' if c >= o else '#ef5350' for o, c in zip(hist['Open'], hist['Close'])]
        fig.add_trace(go.Bar(x=hist.index, y=hist['Volume'], name="Volume", marker_color=v_colors, opacity=0.5), row=2, col=1)

        # ตัดช่วงตลาดปิด (TradingView Style)
        fig.update_xaxes(rangebreaks=[
            dict(bounds=["sat", "mon"]), 
            dict(bounds=[16, 9.5], pattern="hour") if "m" in tf or "h" in tf else None
        ])
        
        fig.update_layout(height=700, template="plotly_dark", xaxis_rangeslider_visible=False, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    # AI Analysis Section
    st.divider()
    if st.button("⚡ วิเคราะห์กลยุทธ์ด้วย AI", type="primary"):
        api_key = st.secrets.get("GEMINI_API_KEY")
        if not api_key:
            st.error("กรุณาใส่ API Key ใน Settings")
        else:
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('models/gemini-2.0-flash')
                curr_p = hist['Close'].iloc[-1]
                prompt = f"ในฐานะเซียนหุ้น วิเคราะห์หุ้น {target} ราคาปัจจุบัน ${curr_p:.2f} สำหรับพอร์ต {selected_port} โดยให้คำแนะนำตามเทคนิค EMA และสถานะตลาดปัจจุบัน ออกมาเป็นภาษาไทย"
                response = model.generate_content(prompt)
                st.markdown(response.text)
            except Exception as e:
                st.error(f"AI Error: {e}")
