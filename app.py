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

# --- 1. ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="AI Portfolio Commander 🚀", layout="wide")
st.markdown("""
<style>
    .stMetric { background-color: #1E1E1E; padding: 10px; border-radius: 10px; border: 1px solid #333; }
    div[data-testid="stExpander"] { background-color: #262730; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# --- 2. เชื่อมต่อ Google Sheets ---
try:
    conn = st.connection("gsheets", type=GSheetsConnection)
except Exception as e:
    st.error(f"❌ Database Connection Error: {e}")

# --- 3. ฟังก์ชันเตรียมข้อมูล ---
@st.cache_data(ttl=86400)
def get_sp500():
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        df = pd.read_html(res.text)[0]
        return dict(zip(df.Symbol, df.Security))
    except:
        return {"NVDA": "NVIDIA", "AAPL": "Apple", "TSLA": "Tesla"}

SP500 = get_sp500()

# --- 4. ระบบจัดการข้อมูล (Core Data Logic) ---
def get_portfolio_data(portfolio_name):
    # อ่านข้อมูลทั้งหมดจาก Sheets (รวม Cost, Qty, Note)
    try:
        df = conn.read(worksheet=portfolio_name, ttl=0)
        # แปลงให้เป็น Format ที่ถูกต้อง กัน Error
        if 'cost' not in df.columns: df['cost'] = 0.0
        if 'qty' not in df.columns: df['qty'] = 0.0
        if 'note' not in df.columns: df['note'] = ""
        return df
    except:
        return pd.DataFrame(columns=['symbol', 'cost', 'qty', 'note'])

def update_stock_data(portfolio_name, symbol, cost=None, qty=None, note=None, action="update"):
    df = get_portfolio_data(portfolio_name)
    
    # กรณีเพิ่มหุ้นใหม่
    if action == "add" and symbol not in df['symbol'].values:
        new_row = pd.DataFrame([{'symbol': symbol, 'cost': 0.0, 'qty': 0.0, 'note': ''}])
        df = pd.concat([df, new_row], ignore_index=True)
        st.toast(f"✅ เพิ่ม {symbol} แล้ว", icon="✨")

    # กรณีลบหุ้น
    elif action == "remove" and symbol in df['symbol'].values:
        df = df[df['symbol'] != symbol]
        st.toast(f"🗑️ ลบ {symbol} แล้ว", icon="👋")

    # กรณีอัปเดตข้อมูล (Cost/Qty/Note)
    elif action == "update" and symbol in df['symbol'].values:
        idx = df.index[df['symbol'] == symbol][0]
        if cost is not None: df.at[idx, 'cost'] = cost
        if qty is not None: df.at[idx, 'qty'] = qty
        if note is not None: df.at[idx, 'note'] = note
        st.toast("💾 บันทึกข้อมูลสำเร็จ", icon="floppy_disk")

    conn.update(worksheet=portfolio_name, data=df)
    return df

# --- 5. Sidebar Navigation ---
with st.sidebar:
    st.title("🏦 Commander Center")
    selected_port = st.selectbox("เลือกพอร์ต:", ["Dime", "Webull"])
    
    # โหลดข้อมูลพอร์ตล่าสุด
    df_port = get_portfolio_data(selected_port)
    watchlist = df_port['symbol'].tolist() if not df_port.empty else []
    
    st.divider()
    
    # ส่วนเพิ่มหุ้น (Add Stock)
    with st.expander("➕ เพิ่มหุ้นใหม่"):
        new_s = st.selectbox("S&P 500:", [""] + [f"{k}" for k in SP500.keys()])
        custom = st.text_input("หรือพิมพ์เอง (เช่น RKLB):").upper().strip()
        
        target_add = custom if custom else new_s
        if st.button("เพิ่มเข้าพอร์ต") and target_add:
            update_stock_data(selected_port, target_add, action="add")
            st.rerun()

    st.divider()
    
    # เมนูเลือกหุ้น
    if watchlist:
        st.subheader("📋 Watchlist")
        target = st.radio("หุ้นในพอร์ต:", watchlist)
        
        st.divider()
        if st.button("🗑️ ลบหุ้นที่เลือก", type="secondary"):
            update_stock_data(selected_port, target, action="remove")
            st.rerun()
    else:
        target = None
        st.info("พอร์ตว่างเปล่า")

# --- 6. Main Dashboard ---
if target:
    # ดึงข้อมูลหุ้นที่เลือกจาก DataFrame
    stock_info = df_port[df_port['symbol'] == target].iloc[0]
    my_cost = float(stock_info['cost'])
    my_qty = float(stock_info['qty'])
    my_note = str(stock_info['note'])

    st.title(f"🚀 {target} @ {selected_port}")

    # 6.1 ข้อมูลตลาด Real-time & P/L Calculation
    raw = yf.Ticker(target).history(period="5d")
    if not raw.empty:
        curr_p = raw['Close'].iloc[-1]
        prev_p = raw['Close'].iloc[-2]
        mkt_change = curr_p - prev_p
        mkt_pct = (mkt_change / prev_p) * 100
        
        # คำนวณกำไร/ขาดทุนส่วนตัว
        market_value = curr_p * my_qty
        total_cost = my_cost * my_qty
        unrealized_pl = market_value - total_cost
        pl_pct = (unrealized_pl / total_cost * 100) if total_cost > 0 else 0

        # แสดงผล 4 ช่อง (Metric)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ราคาตลาด", f"${curr_p:.2f}", f"{mkt_change:.2f} ({mkt_pct:.2f}%)")
        c2.metric("มูลค่าพอร์ต (MV)", f"${market_value:,.2f}", delta_color="off")
        c3.metric("ต้นทุนรวม", f"${total_cost:,.2f}", delta_color="off")
        c4.metric("กำไร/ขาดทุน (P/L)", f"${unrealized_pl:,.2f}", f"{pl_pct:.2f}%", delta_color="normal")

    # 6.2 ส่วนจัดการต้นทุน & Journal (อยู่ใต้กราฟราคา เพื่อความสะดวก)
    with st.expander(f"📝 บันทึกข้อมูลการเทรด: {target}", expanded=False):
        c_edit1, c_edit2 = st.columns(2)
        with c_edit1:
            new_cost = st.number_input("ราคาต้นทุนเฉลี่ย ($):", value=my_cost, format="%.2f")
            new_qty = st.number_input("จำนวนหุ้นที่ถือ:", value=my_qty, format="%.4f")
        with c_edit2:
            new_note = st.text_area("Trading Journal (เหตุผลที่ซื้อ/ขาย):", value=my_note, height=100)
            
        if st.button("💾 บันทึกข้อมูลส่วนตัว"):
            update_stock_data(selected_port, target, cost=new_cost, qty=new_qty, note=new_note)
            st.rerun()

    # 6.3 กราฟ TradingView Style
    tf = st.pills("Timeframe:", ["1m", "5m", "15m", "1h", "1d", "1wk"], default="1h")
    p_map = {"1m":"1d","5m":"5d","15m":"1mo","1h":"3mo","1d":"1y","1wk":"2y"}
    hist = yf.Ticker(target).history(period=p_map.get(tf,"1y"), interval=tf)

    if not hist.empty:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name="Price"), row=1, col=1)
        
        # เส้น EMA
        ema20 = hist['Close'].ewm(span=20).mean()
        ema50 = hist['Close'].ewm(span=50).mean()
        fig.add_trace(go.Scatter(x=hist.index, y=ema20, name="EMA 20", line=dict(color='orange', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=hist.index, y=ema50, name="EMA 50", line=dict(color='blue', width=1)), row=1, col=1)
        
        # Volume
        v_colors = ['#26a69a' if c >= o else '#ef5350' for o, c in zip(hist['Open'], hist['Close'])]
        fig.add_trace(go.Bar(x=hist.index, y=hist['Volume'], name="Volume", marker_color=v_colors), row=2, col=1)

        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"]), dict(bounds=[16, 9.5], pattern="hour") if "m" in tf or "h" in tf else None])
        fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    # --- 7. AI News Scoring System (New Feature) ---
    st.divider()
    st.subheader("🤖 AI News & Sentiment Score")
    
    col_news, col_ai = st.columns([1, 1])

    with col_news:
        st.caption("ข่าวล่าสุดจากตลาดโลก")
        try:
            with DDGS() as ddgs:
                news_results = list(ddgs.text(f"{target} stock financial news", max_results=5))
                if news_results:
                    news_content = ""
                    for n in news_results:
                        st.markdown(f"**[{n['title']}]({n['href']})**")
                        news_content += f"- {n['title']}\n"
                else:
                    news_content = "No news found."
                    st.info("ไม่พบข่าวล่าสุด")
        except:
            news_content = "Error fetching news."
            st.error("ดึงข้อมูลข่าวไม่ได้")

    with col_ai:
        if st.button("🔥 วิเคราะห์คะแนนข่าว (AI Score)", type="primary"):
            api_key = st.secrets.get("GEMINI_API_KEY")
            if api_key:
                with st.spinner("AI กำลังอ่านข่าวและให้คะแนน..."):
                    try:
                        genai.configure(api_key=api_key)
                        # ใช้โมเดล 2.0 Flash เพราะเร็วและประหยัดโควต้า
                        model = genai.GenerativeModel('models/gemini-2.0-flash') 
                        
                        prompt = f"""
                        Analyze these news headlines for {target}:
                        {news_content}
                        
                        Task:
                        1. Give a Sentiment Score from 0 (Extremely Bearish) to 100 (Extremely Bullish).
                        2. Summarize the key driver in 1 sentence (Thai).
                        
                        Format:
                        SCORE: [Number]
                        SUMMARY: [Text]
                        """
                        
                        response = model.generate_content(prompt)
                        text_res = response.text
                        
                        # ดึงค่าคะแนนออกมาแสดง
                        import re
                        score_match = re.search(r"SCORE: (\d+)", text_res)
                        score = int(score_match.group(1)) if score_match else 50
                        
                        # แสดงผลแบบ Gauge
                        st.metric("AI Sentiment Score", f"{score}/100", delta=score-50)
                        st.progress(score)
                        
                        if score >= 70: st.success("อารมณ์ตลาด: กระทิงดุ (Bullish) 🐂")
                        elif score <= 30: st.error("อารมณ์ตลาด: หมีตะปบ (Bearish) 🐻")
                        else: st.warning("อารมณ์ตลาด: ไซด์เวย์ (Neutral) ⚖️")
                        
                        st.write(text_res.split("SUMMARY:")[-1].strip())
                        
                    except Exception as e:
                        st.error(f"AI Error: {e}")
            else:
                st.error("ไม่พบ API Key")

else:
    st.info("👈 กรุณาเลือกหุ้นจากเมนูด้านซ้ายเพื่อเริ่มใช้งาน")
