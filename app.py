import streamlit as st
import yfinance as yf
import google.generativeai as genai
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_gsheets import GSheetsConnection
from duckduckgo_search import DDGS
import pandas as pd
import requests

# --- 1. การตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="AI Multi-Portfolio Sniper Elite 🚀", layout="wide")

# --- 2. การเชื่อมต่อ Google Sheets (ต้องตั้งค่า Secrets แบบ TOML ให้ถูกต้อง) ---
try:
    conn = st.connection("gsheets", type=GSheetsConnection)
except Exception as e:
    st.error(f"❌ Connection Error: {e}")

@st.cache_data(ttl=86400)
def get_sp500():
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        df = pd.read_html(res.text)[0]
        return dict(zip(df.Symbol, df.Security))
    except:
        return {"AAPL": "Apple", "TSLA": "Tesla", "NVDA": "NVIDIA", "MSFT": "Microsoft"}

SP500 = get_sp500()

# --- 3. ระบบจัดการข้อมูลพอร์ต ---
def sync_data(portfolio_name, action, ticker=None):
    try:
        # อ่านข้อมูลพอร์ตแบบ Real-time
        df = conn.read(worksheet=portfolio_name, usecols=[0], ttl=0)
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

# --- 4. เมนูแถบข้าง (Sidebar) ---
with st.sidebar:
    st.title("🏦 My Terminal")
    selected_port = st.selectbox("เลือกพอร์ตที่ใช้งาน:", ["Dime", "Webull"])
    watchlist = sync_data(selected_port, "read")
    
    st.divider()
    st.subheader("📈 Technical Tools")
    show_ema = st.toggle("Show EMA Lines", value=True)
    ema_vals = st.multiselect("EMA Periods:", [20, 50, 100, 200], default=[20, 50])
    
    st.divider()
    st.subheader(f"➕ Add to {selected_port}")
    new_stock = st.selectbox("S&P 500:", [""] + [f"{k} - {v}" for k,v in SP500.items()])
    if st.button("บันทึก") and new_stock:
        sync_data(selected_port, "add", new_stock.split(" - ")[0])
        st.rerun()
        
    custom = st.text_input("Ticker (เช่น RKLB):").upper().strip()
    if st.button("Add Custom") and custom:
        if not yf.Ticker(custom).history(period="1d").empty:
            sync_data(selected_port, "add", custom)
            st.rerun()
        else: st.error("ไม่พบข้อมูลหุ้น")

    st.divider()
    if watchlist:
        target = st.radio(f"พอร์ต {selected_port}:", watchlist)
        if st.button("🗑️ ลบหุ้นที่เลือก"):
            sync_data(selected_port, "remove", target)
            st.rerun()
    else: target = None

# --- 5. การแสดงผล (Main Display) ---
if target:
    st.title(f"🚀 {target} @ {selected_port}")
    
    # สรุปราคาล่าสุด
    raw = yf.Ticker(target).history(period="5d")
    if not raw.empty:
        curr_p = raw['Close'].iloc[-1]
        change = curr_p - raw['Close'].iloc[-2]
        pct = (change / raw['Close'].iloc[-2]) * 100
        cols = st.columns(4)
        cols[0].metric("Price", f"${curr_p:.2f}", f"{change:.2f} ({pct:.2f}%)")
        cols[1].metric("High", f"${raw['High'].iloc[-1]:.2f}")
        cols[2].metric("Low", f"${raw['Low'].iloc[-1]:.2f}")
        cols[3].metric("Volume", f"{raw['Volume'].iloc[-1]:,.0f}")

    tf = st.pills("Timeframe:", ["1m", "5m", "15m", "1h", "1d", "1wk", "YTD", "1Y", "5Y"], default="1h")
    p_map = {"1m":"1d","5m":"5d","15m":"1mo","1h":"3mo","1d":"1y","1wk":"2y","YTD":"ytd","1Y":"1y","5Y":"5y"}
    hist = yf.Ticker(target).history(period=p_map[tf], interval=("1d" if tf in ["YTD","1Y","5Y"] else tf))

    if not hist.empty:
        # กราฟ TradingView Style
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name="Price"), row=1, col=1)
        
        if show_ema:
            colors = ['#2962FF', '#FF9800', '#F44336', '#4CAF50']
            for i, v in enumerate(ema_vals):
                ema = hist['Close'].ewm(span=v).mean()
                fig.add_trace(go.Scatter(x=hist.index, y=ema, name=f"EMA {v}", line=dict(width=1.5, color=colors[i%4])), row=1, col=1)

        v_colors = ['#26a69a' if c >= o else '#ef5350' for o, c in zip(hist['Open'], hist['Close'])]
        fig.add_trace(go.Bar(x=hist.index, y=hist['Volume'], name="Volume", marker_color=v_colors, opacity=0.5), row=2, col=1)

        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"]), dict(bounds=[16, 9.5], pattern="hour") if "m" in tf or "h" in tf else None])
        fig.update_layout(height=700, template="plotly_dark", xaxis_rangeslider_visible=False, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    # --- 6. AI & News (Ultimate Fallback Logic) ---
    st.divider()
    l_col, r_col = st.columns(2)
    with r_col:
        st.subheader("📰 Latest News")
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(f"{target} stock financial news", max_results=5))
                news_txt = "\n".join([f"- [{n['title']}]({n['href']})" for n in results])
                st.markdown(news_txt if results else "ไม่พบข่าวล่าสุด")
        except: news_txt = "ไม่มีข้อมูลข่าวสารล่าสุด"

    with l_col:
        st.subheader("🤖 AI Tactical Analysis")
        if st.button("🚀 Analyze Now", type="primary"):
            api_key = st.secrets.get("GEMINI_API_KEY")
            if api_key:
                with st.spinner("AI กำลังค้นหาโมเดลที่ใช้งานได้..."):
                    # รายชื่อโมเดลทั้งหมดที่คุณส่งมา
                    models_to_try = [
                        'models/gemini-2.0-flash', 
                        'models/gemini-2.0-flash-lite',
                        'models/gemini-1.5-flash-latest', 
                        'models/gemini-1.5-flash',
                        'models/gemini-1.5-pro',
                        'models/gemini-2.0-pro-exp-02-05' # เพิ่มตัวแรงสุดให้ด้วย
                    ]
                    success = False
                    for m_name in models_to_try:
                        try:
                            genai.configure(api_key=api_key)
                            model = genai.GenerativeModel(m_name)
                            prompt = f"ในฐานะเซียนหุ้น วิเคราะห์ {target} ราคา ${curr_p:.2f} พอร์ต {selected_port} ข่าว: {news_txt} สรุปกลยุทธ์ภาษาไทย"
                            res = model.generate_content(prompt)
                            st.success(f"✅ สำเร็จด้วยโมเดล: {m_name}")
                            st.markdown(res.text)
                            success = True
                            break # ถ้าสำเร็จให้หยุดลูปทันที
                        except Exception as e:
                            # ถ้าติด Error ให้ข้ามไปลองโมเดลตัวถัดไป
                            continue
                    
                    if not success: 
                        st.error("❌ ทุกโมเดลติดโควต้าหรือเข้าถึงไม่ได้ในขณะนี้ กรุณารอ 1-2 นาทีแล้วลองใหม่")
            else: st.error("ไม่พบ GEMINI_API_KEY ใน Secrets")
else: st.info("👈 เลือกหุ้นจากพอร์ตด้านซ้ายเพื่อเริ่มต้น")
