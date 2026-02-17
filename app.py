import streamlit as st
import yfinance as yf
import google.generativeai as genai
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_gsheets import GSheetsConnection
from duckduckgo_search import DDGS
import pandas as pd
import requests
import time

# --- ส่วนที่ 1: การตั้งค่าหน้าเว็บ (Page Config) ---
st.set_page_config(page_title="AI Multi-Portfolio Sniper Elite 🚀", layout="wide")

# ปรับแต่ง CSS เล็กน้อยเพื่อให้ดูสะอาดตา
st.markdown("""
<style>
    div[data-testid="stPills"] { gap: 10px; justify-content: flex-start; }
    .stButton>button { border-radius: 8px; font-weight: bold; }
    div[data-testid="stMetricValue"] { font-size: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# --- ส่วนที่ 2: เชื่อมต่อฐานข้อมูล (Database Connection) ---
try:
    # เชื่อมต่อ Google Sheets ผ่าน Service Account ที่ตั้งใน Secrets
    conn = st.connection("gsheets", type=GSheetsConnection)
except Exception as e:
    st.error(f"❌ เชื่อมต่อ Google Sheets ไม่ได้: {e}")

# --- ส่วนที่ 3: ฟังก์ชันเตรียมข้อมูล (Data Fetching) ---
@st.cache_data(ttl=86400) # เก็บ Cache ไว้ 24 ชม. จะได้ไม่ต้องโหลดใหม่บ่อยๆ
def get_sp500():
    try:
        # ดึงรายชื่อหุ้น S&P 500 จาก Wikipedia
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        df = pd.read_html(res.text)[0]
        return dict(zip(df.Symbol, df.Security))
    except:
        # ถ้าเน็ตหลุด ให้ใช้รายชื่อสำรองนี้แทน
        return {"AAPL": "Apple", "TSLA": "Tesla", "NVDA": "NVIDIA", "MSFT": "Microsoft"}

SP500 = get_sp500()

# --- ส่วนที่ 4: ระบบจัดการพอร์ต (Portfolio Logic) ---
def sync_data(portfolio_name, action, ticker=None):
    try:
        # อ่านข้อมูลจาก Sheet ตามชื่อพอร์ต (Dime หรือ Webull) แบบ Real-time (ttl=0)
        df = conn.read(worksheet=portfolio_name, usecols=[0], ttl=0)
        current_list = df.iloc[:, 0].dropna().tolist()
    except Exception as e:
        st.warning(f"⚠️ หา Tab ชื่อ '{portfolio_name}' ไม่เจอ หรือยังไม่ได้สร้าง: {e}")
        current_list = []

    # เพิ่มหุ้น (Add)
    if action == "add" and ticker and ticker not in current_list:
        current_list.append(ticker)
        new_df = pd.DataFrame(current_list, columns=["symbol"])
        conn.update(worksheet=portfolio_name, data=new_df)
        st.toast(f"✅ เพิ่ม {ticker} ลงพอร์ต {portfolio_name} แล้ว", icon="💾")
        
    # ลบหุ้น (Remove)
    elif action == "remove" and ticker in current_list:
        current_list.remove(ticker)
        new_df = pd.DataFrame(current_list, columns=["symbol"])
        conn.update(worksheet=portfolio_name, data=new_df)
        st.toast(f"🗑️ ลบ {ticker} เรียบร้อย", icon="👋")
    
    return current_list

# --- ส่วนที่ 5: เมนูควบคุมด้านข้าง (Sidebar) ---
with st.sidebar:
    st.title("🏦 My Terminal")
    
    # 5.1 เลือกพอร์ต
    selected_port = st.selectbox("เลือกพอร์ตที่ใช้งาน:", ["Dime", "Webull"])
    watchlist = sync_data(selected_port, "read")
    
    st.divider()
    
    # 5.2 ตั้งค่ากราฟ
    st.subheader("📈 Technical Tools")
    show_ema = st.toggle("แสดงเส้น EMA", value=True)
    ema_vals = st.multiselect("ค่า EMA:", [20, 50, 100, 200], default=[20, 50])
    
    st.divider()
    
    # 5.3 เพิ่มหุ้น (Add Stock)
    st.subheader(f"➕ เพิ่มหุ้นใน {selected_port}")
    # แบบเลือกจาก S&P 500
    new_stock = st.selectbox("เลือกจาก S&P 500:", [""] + [f"{k} - {v}" for k,v in SP500.items()])
    if st.button("บันทึก") and new_stock:
        sync_data(selected_port, "add", new_stock.split(" - ")[0])
        st.rerun() # รีโหลดหน้าเว็บทันที
        
    # แบบพิมพ์เอง (Custom)
    custom = st.text_input("หรือพิมพ์ชื่อหุ้น (เช่น RKLB):").upper().strip()
    if st.button("เพิ่ม Custom Stock") and custom:
        # เช็คก่อนว่ามีหุ้นจริงไหม กัน Error
        if not yf.Ticker(custom).history(period="1d").empty:
            sync_data(selected_port, "add", custom)
            st.rerun()
        else:
            st.error("❌ ไม่พบข้อมูลหุ้นตัวนี้ในตลาด")

    st.divider()
    
    # 5.4 ลบหุ้น
    if watchlist:
        target = st.radio(f"หุ้นในพอร์ต {selected_port}:", watchlist)
        if st.button("🗑️ ลบหุ้นที่เลือก"):
            sync_data(selected_port, "remove", target)
            st.rerun()
    else:
        target = None
        st.info("👈 พอร์ตว่างเปล่า เริ่มเพิ่มหุ้นได้เลย")

# --- ส่วนที่ 6: หน้าจอแสดงผลหลัก (Main Dashboard) ---
if target:
    st.title(f"🚀 {target} @ {selected_port}")
    
    # 6.1 ข้อมูลราคา Real-time
    raw = yf.Ticker(target).history(period="5d")
    if not raw.empty:
        curr_p = raw['Close'].iloc[-1]
        prev_p = raw['Close'].iloc[-2]
        change = curr_p - prev_p
        pct = (change / prev_p) * 100
        
        # แสดง 4 ช่องข้อมูลหลัก
        cols = st.columns(4)
        cols[0].metric("Price", f"${curr_p:.2f}", f"{change:.2f} ({pct:.2f}%)")
        cols[1].metric("High", f"${raw['High'].iloc[-1]:.2f}")
        cols[2].metric("Low", f"${raw['Low'].iloc[-1]:.2f}")
        cols[3].metric("Volume", f"{raw['Volume'].iloc[-1]:,.0f}")

    # 6.2 ตัวเลือก Timeframe
    tf = st.pills("Timeframe:", ["1m", "5m", "15m", "1h", "1d", "1wk", "YTD", "1Y", "5Y"], default="1h")
    
    # แปลง Timeframe ให้ yfinance เข้าใจ
    p_map = {"1m":"1d","5m":"5d","15m":"1mo","1h":"3mo","1d":"1y","1wk":"2y","YTD":"ytd","1Y":"1y","5Y":"5y"}
    actual_interval = "1d" if tf in ["YTD","1Y","5Y"] else tf
    hist = yf.Ticker(target).history(period=p_map[tf], interval=actual_interval)

    if not hist.empty:
        # 6.3 สร้างกราฟ TradingView Style
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
        
        # กราฟแท่งเทียน (Candlestick)
        fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name="Price"), row=1, col=1)
        
        # เส้น EMA (วาดตามที่เลือกใน Sidebar)
        if show_ema:
            colors = ['#2962FF', '#FF9800', '#F44336', '#4CAF50'] # น้ำเงิน, ส้ม, แดง, เขียว
            for i, v in enumerate(ema_vals):
                ema = hist['Close'].ewm(span=v).mean()
                fig.add_trace(go.Scatter(x=hist.index, y=ema, name=f"EMA {v}", line=dict(width=1.5, color=colors[i%4])), row=1, col=1)

        # กราฟ Volume (แท่งเขียว/แดง)
        v_colors = ['#26a69a' if c >= o else '#ef5350' for o, c in zip(hist['Open'], hist['Close'])]
        fig.add_trace(go.Bar(x=hist.index, y=hist['Volume'], name="Volume", marker_color=v_colors, opacity=0.5), row=2, col=1)

        # ตัดช่วงตลาดปิด (เสาร์-อาทิตย์ และกลางคืน) ออกจากกราฟ
        fig.update_xaxes(rangebreaks=[
            dict(bounds=["sat", "mon"]), 
            dict(bounds=[16, 9.5], pattern="hour") if "m" in tf or "h" in tf else None
        ])
        
        fig.update_layout(height=700, template="plotly_dark", xaxis_rangeslider_visible=False, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    # --- ส่วนที่ 7: AI & ข่าวสาร (Intelligence Layer) ---
    st.divider()
    l_col, r_col = st.columns(2)
    
    # 7.1 ดึงข่าวจาก DuckDuckGo
    with r_col:
        st.subheader("📰 ข่าวล่าสุด")
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(f"{target} stock financial news", max_results=5))
                news_txt = "\n".join([f"- [{n['title']}]({n['href']})" for n in results])
                st.markdown(news_txt if results else "ไม่พบข่าวใหม่ในช่วงนี้")
        except: 
            news_txt = "ไม่มีข้อมูลข่าวสาร"
            st.info("ไม่สามารถโหลดข่าวได้")

    # 7.2 ระบบวิเคราะห์ AI (Multi-Model Fallback)
    with l_col:
        st.subheader("🤖 AI Tactical Analysis")
        if st.button("🚀 วิเคราะห์ด้วย AI", type="primary"):
            api_key = st.secrets.get("GEMINI_API_KEY")
            
            if api_key:
                with st.spinner("AI กำลังค้นหาโมเดลที่ว่างอยู่..."):
                    # รายชื่อโมเดลเรียงตามความฉลาด
                    models_to_try = [
                        'models/gemini-2.0-flash', 
                        'models/gemini-2.0-flash-lite',
                        'models/gemini-1.5-flash-latest', 
                        'models/gemini-1.5-pro',
                        'models/gemini-2.0-pro-exp-02-05'
                    ]
                    
                    success = False
                    for m_name in models_to_try:
                        try:
                            # ตั้งค่าและลองเรียกใช้โมเดล
                            genai.configure(api_key=api_key)
                            model = genai.GenerativeModel(m_name)
                            
                            prompt = f"""วิเคราะห์หุ้น {target} ราคา ${curr_p:.2f} พอร์ต {selected_port} 
                            ข่าว: {news_txt} 
                            ขอคำแนะนำสั้นๆ ภาษาไทย: 1.แนวโน้ม 2.จุดสังเกต 3.กลยุทธ์(ซื้อ/ขาย/ถือ)"""
                            
                            res = model.generate_content(prompt)
                            
                            # ถ้าสำเร็จ ให้แสดงผลและหยุดลูป
                            st.success(f"✅ วิเคราะห์สำเร็จ (Model: {m_name})")
                            st.markdown(res.text)
                            success = True
                            break 
                        except Exception as e:
                            # ถ้า Error (เช่น 429) ให้ข้ามไปตัวถัดไป
                            continue
                    
                    if not success: 
                        st.error("❌ ทุกโมเดลทำงานหนักเกินไป กรุณารอ 1 นาทีแล้วกดใหม่ครับ")
            else: 
                st.error("ไม่พบ API Key ใน Secrets")

else:
    st.info("👈 กรุณาเลือกหุ้นจากเมนูด้านซ้ายเพื่อเริ่มต้น")
