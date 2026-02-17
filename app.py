import streamlit as st
import yfinance as yf
import google.generativeai as genai
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_gsheets import GSheetsConnection
from duckduckgo_search import DDGS
import pandas as pd
import requests
import re
import time

# --- 1. ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="Port to TheMoon Commander 🚀", layout="wide")
st.markdown("""
<style>
    .stMetric { background-color: #1E1E1E; border: 1px solid #333; border-radius: 10px; padding: 10px; }
    div[data-testid="stExpander"] { background-color: #262730; border-radius: 10px; }
    /* ปรับแต่ง Progress Bar ให้สวยขึ้น */
    div[data-testid="stProgressBar"] > div { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# --- 2. เชื่อมต่อ Google Sheets ---
try:
    conn = st.connection("gsheets", type=GSheetsConnection)
except Exception as e:
    st.error(f"❌ Connection Error: {e}")

# --- 3. เตรียมข้อมูล S&P 500 ---
@st.cache_data(ttl=86400)
def get_sp500():
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        df = pd.read_html(res.text)[0]
        return dict(zip(df.Symbol, df.Security))
    except:
        return {"NVDA": "NVIDIA", "TSLA": "Tesla", "AAPL": "Apple"}

SP500 = get_sp500()

# --- 4. ฟังก์ชันจัดการข้อมูล (Sheet Logic) ---
def clean_symbol(sym):
    # แปลง "NASDAQ:RKLB" -> "RKLB"
    if isinstance(sym, str):
        parts = sym.split(":")
        return parts[-1].strip()
    return str(sym)

def get_sheet_data(tab_name):
    try:
        df = conn.read(worksheet=tab_name, ttl=0)
        # Mapping Col: A=Symbol(0), C=Qty(2), D=Cost(3), K=Notes(10)
        # ตรวจสอบว่ามีคอลัมน์ครบไหม
        if len(df.columns) > 10:
            needed_cols = df.iloc[:, [0, 2, 3, 10]].copy()
            needed_cols.columns = ['raw_symbol', 'qty', 'cost', 'note']
            
            # Cleaning
            needed_cols['symbol'] = needed_cols['raw_symbol'].apply(clean_symbol)
            needed_cols['qty'] = pd.to_numeric(needed_cols['qty'], errors='coerce').fillna(0.0)
            needed_cols['cost'] = pd.to_numeric(needed_cols['cost'], errors='coerce').fillna(0.0)
            needed_cols['note'] = needed_cols['note'].fillna("")
            
            return needed_cols[needed_cols['symbol'] != ""]
        else:
            st.error("Format ไฟล์ Sheet ไม่ตรง (ต้องการอย่างน้อย 11 คอลัมน์)")
            return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame(columns=['symbol', 'qty', 'cost', 'note'])

def update_specific_cell(tab_name, symbol, cost=None, qty=None, note=None):
    try:
        sh = conn.client.open_by_url(st.secrets["connections"]["gsheets"]["spreadsheet"])
        wks = sh.worksheet(tab_name)
        cell = wks.find(symbol, in_column=1) # หาในคอลัมน์ A
        
        if cell:
            row = cell.row
            if qty is not None: wks.update_cell(row, 3, qty)   # Col C
            if cost is not None: wks.update_cell(row, 4, cost) # Col D
            if note is not None: wks.update_cell(row, 11, note) # Col K
            st.toast(f"💾 บันทึก {symbol} แล้ว", icon="✅")
        else:
            st.warning("⚠️ หุ้นนี้ยังไม่มีใน Sheet! (ระบบเพิ่มให้ต่อท้าย)")
            new_row = [symbol, "", qty or 0, cost or 0] + [""]*6 + [note or ""]
            wks.append_row(new_row)
            st.toast(f"✨ เพิ่ม {symbol} ใหม่", icon="🆕")
        return True
    except Exception as e:
        st.error(f"Update Error: {e}")
        return False

# --- 5. Sidebar ---
with st.sidebar:
    st.title("🌕 Commander")
    port_map = {"Dime": "PORTFOLIO(DIME)", "Webull": "PORTFOLIO(WEBULL)"}
    selected_key = st.selectbox("เลือกพอร์ต:", list(port_map.keys()))
    selected_tab = port_map[selected_key]
    
    df_port = get_sheet_data(selected_tab)
    watchlist = df_port['symbol'].tolist() if not df_port.empty else []
    
    st.divider()
    
    # เพิ่มหุ้น
    with st.expander("➕ เพิ่ม/แก้ไข หุ้น"):
        input_stock = st.text_input("ชื่อหุ้น (เช่น NVDA):").upper().strip()
        c1, c2 = st.columns(2)
        u_cost = c1.number_input("ทุนเฉลี่ย ($):", value=0.0)
        u_qty = c2.number_input("จำนวนหุ้น:", value=0.0)
        
        if st.button("บันทึกลง Sheet") and input_stock:
            update_specific_cell(selected_tab, input_stock, cost=u_cost, qty=u_qty)
            st.rerun()

    st.divider()
    if watchlist:
        target_symbol = st.radio("รายการหุ้น:", watchlist)
    else:
        target_symbol = None
        st.info("ไม่พบข้อมูลหุ้นใน Tab นี้")

# --- 6. Main Dashboard ---
if target_symbol:
    row_data = df_port[df_port['symbol'] == target_symbol].iloc[0]
    my_cost = float(row_data['cost'])
    my_qty = float(row_data['qty'])
    my_note = str(row_data['note'])
    real_sheet_symbol = row_data['raw_symbol']

    st.title(f"🚀 {target_symbol} Analysis")
    st.caption(f"Source: {selected_tab} | Original: {real_sheet_symbol}")

    # 6.1 ข้อมูลตลาด & P/L
    raw = yf.Ticker(target_symbol).history(period="5d")
    if not raw.empty:
        curr_p = raw['Close'].iloc[-1]
        change = curr_p - raw['Close'].iloc[-2]
        pct = (change / raw['Close'].iloc[-2]) * 100
        
        mkt_val = curr_p * my_qty
        tot_cost = my_cost * my_qty
        unrealized = mkt_val - tot_cost
        pl_pct = (unrealized / tot_cost * 100) if tot_cost > 0 else 0
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ราคาตลาด", f"${curr_p:.2f}", f"{pct:.2f}%")
        c2.metric("มูลค่าพอร์ต", f"${mkt_val:,.2f}")
        c3.metric("ต้นทุนรวม", f"${tot_cost:,.2f}")
        c4.metric("กำไร/ขาดทุน", f"${unrealized:,.2f}", f"{pl_pct:.2f}%", delta_color="normal")

    # 6.2 Journal
    with st.expander(f"📝 Trading Journal ({target_symbol})", expanded=False):
        col_input, col_note = st.columns([1, 2])
        with col_input:
            new_cost = st.number_input("แก้ต้นทุน ($):", value=my_cost, format="%.4f")
            new_qty = st.number_input("แก้จำนวนหุ้น:", value=my_qty, format="%.4f")
        with col_note:
            new_note = st.text_area("Note:", value=my_note, height=100)
        if st.button("💾 อัปเดตข้อมูล"):
            update_specific_cell(selected_tab, real_sheet_symbol, cost=new_cost, qty=new_qty, note=new_note)
            st.rerun()

    # 6.3 กราฟ (แก้บั๊กกราฟว่าง)
    tf = st.pills("Timeframe:", ["1m", "5m", "15m", "1h", "1d", "1wk"], default="1d")
    p_map = {"1m":"1d","5m":"5d","15m":"1mo","1h":"3mo","1d":"1y","1wk":"2y"}
    
    # ดึงข้อมูล
    hist = yf.Ticker(target_symbol).history(period=p_map.get(tf,"1y"), interval=tf)
    
    if not hist.empty:
        # **จุดสำคัญ**: ลบ Timezone ออก เพื่อแก้ปัญหากราฟ Plotly เพี้ยน
        hist.index = hist.index.tz_localize(None) 
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name="Price"), row=1, col=1)
        
        # EMA
        ema20 = hist['Close'].ewm(span=20).mean()
        ema50 = hist['Close'].ewm(span=50).mean()
        fig.add_trace(go.Scatter(x=hist.index, y=ema20, name="EMA 20", line=dict(color='orange', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=hist.index, y=ema50, name="EMA 50", line=dict(color='blue', width=1)), row=1, col=1)
        
        # Volume
        v_colors = ['#26a69a' if c >= o else '#ef5350' for o, c in zip(hist['Open'], hist['Close'])]
        fig.add_trace(go.Bar(x=hist.index, y=hist['Volume'], marker_color=v_colors), row=2, col=1)

        # ตั้งค่า Rangebreaks (ซ่อนวันหยุด) เฉพาะ timeframe ที่ไม่ใช่ Intraday (1m, 5m) จะได้ไม่บั๊ก
        if tf not in ['1m', '5m', '15m', '1h']:
            fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
            
        fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"ไม่พบข้อมูลกราฟสำหรับ {target_symbol} (อาจเป็นเพราะชื่อหุ้นผิด หรือตลาดปิด)")

    # 6.4 ข่าว & AI Scoring (เพิ่มกลับมาแล้ว!)
    st.divider()
    c_news, c_score = st.columns([1, 1])
    
    news_text_for_ai = "" # ตัวแปรเก็บเนื้อหาข่าวส่งให้ AI
    
    with c_news:
        st.subheader("📰 ข่าวล่าสุด")
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(f"{target_symbol} stock financial news", max_results=5))
                if results:
                    for n in results:
                        st.markdown(f"**[{n['title']}]({n['href']})**")
                        news_text_for_ai += f"- {n['title']}\n"
                else:
                    st.info("ไม่พบข่าวล่าสุดในช่วงนี้")
                    news_text_for_ai = "No specific news found."
        except:
            st.error("ไม่สามารถดึงข้อมูลข่าวได้")
            news_text_for_ai = "News fetch error."

    # ส่วนคะแนนข่าว (AI Score)
    with c_score:
        st.subheader("🔥 AI Sentiment Score")
        if st.button("ประเมินอารมณ์ตลาด", type="primary"):
            api_key = st.secrets.get("GEMINI_API_KEY")
            if api_key:
                with st.spinner("AI กำลังอ่านข่าวและให้คะแนน..."):
                    try:
                        genai.configure(api_key=api_key)
                        model = genai.GenerativeModel('models/gemini-2.0-flash') # ใช้ตัวเร็ว
                        
                        prompt = f"""
                        Analyze headlines for {target_symbol}:
                        {news_text_for_ai}
                        
                        Task:
                        1. Score 0 (Bearish) to 100 (Bullish).
                        2. Summarize driver in Thai.
                        
                        Output format:
                        SCORE: [Number]
                        SUMMARY: [Text]
                        """
                        res = model.generate_content(prompt)
                        text = res.text
                        
                        # ดึงคะแนน
                        import re
                        match = re.search(r"SCORE: (\d+)", text)
                        score = int(match.group(1)) if match else 50
                        
                        # แสดง Gauge Bar
                        st.metric("Sentiment Score", f"{score}/100", delta=score-50)
                        st.progress(score)
                        
                        if score >= 70: st.success("ตลาดกระทิง (Bullish) 🐂")
                        elif score <= 30: st.error("ตลาดหมี (Bearish) 🐻")
                        else: st.warning("ตลาดไซด์เวย์ (Neutral) ⚖️")
                        
                        summary = text.split("SUMMARY:")[-1].strip()
                        st.info(f"**สรุป:** {summary}")
                        
                    except Exception as e:
                        st.error(f"AI Error: {e}")
            else:
                st.error("No API Key")

    # 6.5 AI Analysis เต็มรูปแบบ
    st.divider()
    st.subheader("🤖 Deep Tactical Analysis")
    if st.button("วิเคราะห์กลยุทธ์เชิงลึก"):
        api_key = st.secrets.get("GEMINI_API_KEY")
        if api_key:
            with st.spinner("กำลังสแกน..."):
                models = ['models/gemini-2.5-flash', 'models/gemini-2.0-flash', 'models/gemini-1.5-pro']
                success = False
                for m in models:
                    try:
                        genai.configure(api_key=api_key)
                        model = genai.GenerativeModel(m)
                        prompt = f"หุ้น {target_symbol} ราคา ${curr_p} ข่าว: {news_text_for_ai}. วิเคราะห์แนวโน้มกราฟและแนะนำกลยุทธ์ (ไทย)"
                        res = model.generate_content(prompt)
                        st.success(f"Analysis by {m}")
                        st.markdown(res.text)
                        success = True
                        break
                    except: continue
                if not success: st.error("AI Busy.")

else:
    st.info("👈 เลือกหุ้นจากเมนูซ้ายมือ")
