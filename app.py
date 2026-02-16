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
st.set_page_config(page_title="AI Multi-Portfolio Sniper Pro 🚀", layout="wide")

st.markdown("""
<style>
    div[data-testid="stPills"] { gap: 10px; justify-content: flex-start; }
    .stButton>button { border-radius: 8px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- 2. การเชื่อมต่อฐานข้อมูล (Google Sheets) ---
# ใช้ระบบ Service Account ตามที่ตั้งค่าใน Secrets
try:
    conn = st.connection("gsheets", type=GSheetsConnection)
except Exception as e:
    st.error(f"❌ เชื่อมต่อ Google Sheets ล้มเหลว: {e}")
    st.info("ตรวจสอบ Secrets: [connections.gsheets] ว่าใส่ค่าครบและถูกต้องหรือไม่")

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

# --- 3. ระบบจัดการพอร์ต (Sync ข้อมูล) ---
def sync_data(portfolio_name, action, ticker=None):
    try:
        # อ่านข้อมูลแบบ Real-time (ttl=0) 
        # แก้ไข: เพิ่มการเช็คว่า Worksheet มีอยู่จริงไหม
        df = conn.read(worksheet=portfolio_name, usecols=[0], ttl=0)
        current_list = df.iloc[:, 0].dropna().tolist()
    except Exception as e:
        st.warning(f"⚠️ ไม่สามารถอ่านข้อมูลพอร์ต '{portfolio_name}' ได้: {e}")
        st.info(f"ตรวจสอบว่าใน Google Sheets มี Tab ชื่อ '{portfolio_name}' และมีหัวตารางที่ช่อง A1 ว่า 'symbol' หรือยัง")
        current_list = []

    if action == "add" and ticker and ticker not in current_list:
        try:
            current_list.append(ticker)
            new_df = pd.DataFrame(current_list, columns=["symbol"])
            conn.update(worksheet=portfolio_name, data=new_df)
            st.success(f"✅ บันทึก {ticker} ลงพอร์ต {portfolio_name} สำเร็จ")
        except Exception as e:
            st.error(f"❌ บันทึกไม่สำเร็จ: {e}")
            
    elif action == "remove" and ticker in current_list:
        try:
            current_list.remove(ticker)
            new_df = pd.DataFrame(current_list, columns=["symbol"])
            conn.update(worksheet=portfolio_name, data=new_df)
            st.success(f"🗑️ ลบ {ticker} เรียบร้อย")
        except Exception as e:
            st.error(f"❌ ลบไม่สำเร็จ: {e}")
    
    return current_list

# --- 4. เมนูแถบข้าง (Sidebar) ---
with st.sidebar:
    st.title("🏦 Portfolio Manager")
    
    # สลับพอร์ต Dime / Webull
    selected_port = st.selectbox("เลือกพอร์ตที่ใช้งาน:", ["Dime", "Webull"])
    watchlist = sync_data(selected_port, "read")
    
    st.divider()
    
    st.subheader("📈 Technical Setup")
    show_ema = st.toggle("เปิดใช้งานเส้น EMA", value=True)
    ema_vals = st.multiselect("EMA Periods:", [20, 50, 100, 200], default=[20, 50])
    
    st.divider()
    
    st.subheader(f"➕ Add to {selected_port}")
    new_stock = st.selectbox("จากรายชื่อ S&P 500:", [""] + [f"{k} - {v}" for k,v in SP500.items()])
    if st.button("บันทึกลงพอร์ต") and new_stock:
        symbol = new_stock.split(" - ")[0]
        sync_data(selected_port, "add", symbol)
        st.rerun()
        
    custom_stock = st.text_input("พิมพ์ชื่อย่อหุ้น (เช่น PLTR, RKLB):").upper().strip()
    if st.button("Add Custom") and custom_stock:
        with st.spinner(f"กำลังตรวจสอบ {custom_stock}..."):
            check_data = yf.Ticker(custom_stock).history(period="1d")
            if not check_data.empty:
                sync_data(selected_port, "add", custom_stock)
                st.rerun()
            else:
                st.error("❌ ไม่พบข้อมูลหุ้นตัวนี้ (หรือพิมพ์ผิด)")

    st.divider()
    
    if watchlist:
        target = st.radio(f"หุ้นในพอร์ต {selected_port}:", watchlist)
        if st.button("🗑️ ลบหุ้นที่เลือก"):
            sync_data(selected_port, "remove", target)
            st.rerun()
    else:
        target = None

# --- 5. หน้าจอหลัก (Main Terminal) ---
if target:
    st.title(f"🚀 {target} Terminal @ {selected_port}")
    
    try:
        raw_data = yf.Ticker(target).history(period="5d")
        if not raw_data.empty:
            curr_price = raw_data['Close'].iloc[-1]
            prev_close = raw_data['Close'].iloc[-2] if len(raw_data) > 1 else raw_data['Open'].iloc[0]
            change = curr_price - prev_close
            change_pct = (change / prev_close) * 100
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Current Price", f"${curr_price:.2f}", f"{change:.2f} ({change_pct:.2f}%)")
            c2.metric("Day High", f"${raw_data['High'].iloc[-1]:.2f}")
            c3.metric("Day Low", f"${raw_data['Low'].iloc[-1]:.2f}")
            c4.metric("Volume", f"{raw_data['Volume'].iloc[-1]:,.0f}")

        # การเลือกช่วงเวลา
        tf = st.pills("Timeframe:", ["1m", "5m", "15m", "1h", "1d", "1wk", "YTD", "1Y", "5Y"], default="1h")
        
        # ดึงข้อมูลกราฟ
        p_map = {"1m":"1d","5m":"5d","15m":"1mo","1h":"3mo","1d":"1y","1wk":"2y","YTD":"ytd","1Y":"1y","5Y":"5y"}
        actual_i = "1d" if tf in ["YTD", "1Y", "5Y"] else tf
        hist = yf.Ticker(target).history(period=p_map[tf], interval=actual_i)

        if not hist.empty:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                               vertical_spacing=0.03, row_heights=[0.7, 0.3])
            
            fig.add_trace(go.Candlestick(
                x=hist.index, open=hist['Open'], high=hist['High'],
                low=hist['Low'], close=hist['Close'], name="Candlestick"
            ), row=1, col=1)
            
            if show_ema:
                colors = ['#2962FF', '#FF9800', '#F44336', '#4CAF50']
                for idx, val in enumerate(ema_vals):
                    ema = hist['Close'].ewm(span=val, adjust=False).mean()
                    fig.add_trace(go.Scatter(x=hist.index, y=ema, name=f'EMA {val}', 
                                             line=dict(width=1.5, color=colors[idx%4])), row=1, col=1)

            v_colors = ['#26a69a' if c >= o else '#ef5350' for o, c in zip(hist['Open'], hist['Close'])]
            fig.add_trace(go.Bar(x=hist.index, y=hist['Volume'], name="Volume", 
                                 marker_color=v_colors, opacity=0.5), row=2, col=1)

            # ตัดช่วงเวลาตลาดปิด (TradingView Style)
            fig.update_xaxes(rangebreaks=[
                dict(bounds=["sat", "mon"]), 
                dict(bounds=[16, 9.5], pattern="hour") if "m" in tf or "h" in tf else None
            ])
            
            fig.update_layout(height=700, template="plotly_dark", 
                              xaxis_rangeslider_visible=False, hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

        # ส่วนข่าวและ AI วิเคราะห์
        st.divider()
        l_col, r_col = st.columns(2)
        
        with r_col:
            st.subheader("📰 Market News")
            try:
                with DDGS() as ddgs:
                    results = list(ddgs.text(f"{target} stock news", max_results=5))
                    news_txt = "\n".join([f"- [{n['title']}]({n['href']})" for n in results])
                    st.markdown(news_txt if results else "ไม่พบข่าวล่าสุด")
            except:
                news_txt = "ไม่สามารถดึงข้อมูลข่าวได้"
                st.warning(news_txt)

        with l_col:
            st.subheader("🤖 AI Tactical Analysis")
            if st.button("🚀 Run AI Analysis", type="primary"):
                api_key = st.secrets.get("GEMINI_API_KEY")
                if not api_key:
                    st.error("กรุณาใส่ API Key ใน Streamlit Secrets")
                else:
                    with st.spinner("AI กำลังประมวลผล..."):
                        try:
                            genai.configure(api_key=api_key)
                            # ใช้ gemini-1.5-flash เพื่อประหยัดโควต้าและลด Error 429
                            model = genai.GenerativeModel('models/gemini-1.5-flash')
                            prompt = f"""วิเคราะห์หุ้น {target} ราคา ${curr_price:.2f} พอร์ต {selected_port}
                            ข้อมูลข่าว: {news_txt}
                            แนะนำภาษาไทย: Sentiment, แนวโน้มเทียบกับ EMA, และ Action (ซื้อ/ถือ/ขาย)"""
                            response = model.generate_content(prompt)
                            st.markdown(response.text)
                        except Exception as e:
                            if "429" in str(e):
                                st.warning("⚠️ โควต้า AI เต็มชั่วคราว กรุณารอ 30 วินาทีแล้วลองใหม่")
                            else:
                                st.error(f"AI Error: {e}")
    except Exception as e:
        st.error(f"❌ ระบบขัดข้อง: {e}")
else:
    st.info("👈 กรุณาเลือกหรือเพิ่มหุ้นในพอร์ตที่เมนูด้านซ้ายเพื่อเริ่มต้น")
