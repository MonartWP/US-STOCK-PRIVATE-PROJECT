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

# --- 1. ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="Port to TheMoon Commander 🚀", layout="wide")
st.markdown("""
<style>
    .stMetric { background-color: #1E1E1E; border: 1px solid #333; border-radius: 10px; padding: 10px; }
    div[data-testid="stExpander"] { background-color: #262730; border-radius: 10px; }
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

# --- 4. ฟังก์ชันจัดการข้อมูล (Advanced Logic) ---
def clean_symbol(sym):
    # แปลง "NASDAQ:RKLB" -> "RKLB" เพื่อให้ yfinance/AI เข้าใจ
    if isinstance(sym, str):
        return sym.split(":")[-1].strip()
    return str(sym)

def get_sheet_data(tab_name):
    try:
        # อ่านข้อมูลดิบจาก Sheets
        df = conn.read(worksheet=tab_name, ttl=0)
        
        # Mapping คอลัมน์จากไฟล์ "Port to TheMoon" ของคุณ
        # Col A=Symbol, C=Shares(Qty), D=Avg Cost, K=Notes
        # ต้องเลือกคอลัมน์ให้ตรงตามตำแหน่ง (0, 2, 3, 10)
        needed_cols = df.iloc[:, [0, 2, 3, 10]].copy()
        needed_cols.columns = ['raw_symbol', 'qty', 'cost', 'note']
        
        # ล้างข้อมูลให้สะอาด
        needed_cols['symbol'] = needed_cols['raw_symbol'].apply(clean_symbol)
        needed_cols['qty'] = pd.to_numeric(needed_cols['qty'], errors='coerce').fillna(0.0)
        needed_cols['cost'] = pd.to_numeric(needed_cols['cost'], errors='coerce').fillna(0.0)
        needed_cols['note'] = needed_cols['note'].fillna("")
        
        # กรองเอาเฉพาะแถวที่มีชื่อหุ้น
        return needed_cols[needed_cols['symbol'] != ""]
    except Exception as e:
        # st.error(f"Error reading sheet: {e}")
        return pd.DataFrame(columns=['symbol', 'qty', 'cost', 'note'])

def update_specific_cell(tab_name, symbol, cost=None, qty=None, note=None):
    try:
        # ใช้ gspread client เพื่อแก้เฉพาะช่อง (ไม่ทับสูตร)
        sh = conn.client.open_by_url(st.secrets["connections"]["gsheets"]["spreadsheet"])
        wks = sh.worksheet(tab_name)
        
        # ค้นหาว่าหุ้นตัวนี้อยู่แถวไหน (ค้นหาในคอลัมน์ A)
        cell = wks.find(symbol, in_column=1)
        
        if cell:
            row = cell.row
            # อัปเดตเฉพาะคอลัมน์ C(3), D(4), K(11)
            if qty is not None: wks.update_cell(row, 3, qty)
            if cost is not None: wks.update_cell(row, 4, cost)
            if note is not None: wks.update_cell(row, 11, note)
            st.toast(f"💾 อัปเดต {symbol} เรียบร้อย", icon="✅")
        else:
            # ถ้าหาไม่เจอ ให้เพิ่มต่อท้าย (แต่ต้องเตือนเรื่องสูตร)
            st.warning("⚠️ หุ้นนี้ยังไม่มีใน Sheet! ระบบจะเพิ่มต่อท้าย (อย่าลืมลากสูตรใน Sheet ลงมาด้วยนะครับ)")
            # เพิ่มแถวใหม่: [Symbol, "", Qty, Cost, "", ..., Note]
            new_row = [symbol, "", qty if qty else 0, cost if cost else 0, "", "", "", "", "", "", note if note else ""]
            wks.append_row(new_row)
            st.toast(f"✨ เพิ่ม {symbol} ใหม่แล้ว", icon="🆕")
            
        return True
    except Exception as e:
        st.error(f"Update Error: {e}")
        return False

# --- 5. Sidebar ---
with st.sidebar:
    st.title("🌕 Commander")
    
    # เลือกพอร์ต (ชื่อ Tab ต้องตรงกับไฟล์จริง)
    port_map = {"Dime": "PORTFOLIO(DIME)", "Webull": "PORTFOLIO(WEBULL)"}
    selected_key = st.selectbox("เลือกพอร์ต:", list(port_map.keys()))
    selected_tab = port_map[selected_key]
    
    # โหลดข้อมูล
    df_port = get_sheet_data(selected_tab)
    watchlist = df_port['symbol'].tolist()
    
    st.divider()
    
    # เพิ่มหุ้น
    with st.expander("➕ เพิ่ม/แก้ไข หุ้น"):
        # ช่องพิมพ์ชื่อหุ้น (เช่น NASDAQ:NVDA หรือ NVDA เฉยๆ ก็ได้)
        input_stock = st.text_input("ชื่อหุ้น (เช่น NVDA):").upper().strip()
        
        # ถ้าจะเพิ่ม ต้องมีข้อมูลเบื้องต้น
        if input_stock:
            # แปลงให้มี Prefix ถ้าจำเป็น (หรือใส่เพียวๆ แล้วให้ Sheet จัดการ)
            # แต่เพื่อความง่าย ใส่เพียวๆ ไปก่อน แล้วคุณค่อยไปแก้ใน Sheet ให้มี NASDAQ: ก็ได้
            # หรือถ้าระบบฉลาดพอ Find จะหาเจอ
            c1, c2 = st.columns(2)
            u_cost = c1.number_input("ทุนเฉลี่ย ($):", value=0.0)
            u_qty = c2.number_input("จำนวนหุ้น:", value=0.0)
            
            if st.button("บันทึกลง Sheet"):
                # เช็คว่าใน Sheet มี Prefix ไหม (เช่น NASDAQ:)
                # เพื่อความชัวร์ ให้ App ส่งค่าที่พิมพ์ไป update เลย
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
    # ดึงข้อมูลหุ้นตัวนั้นจาก DataFrame
    row_data = df_port[df_port['symbol'] == target_symbol].iloc[0]
    my_cost = float(row_data['cost'])
    my_qty = float(row_data['qty'])
    my_note = str(row_data['note'])
    
    # ชื่อหุ้นจริงใน Sheet (อาจมี NASDAQ:)
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
        c4.metric("กำไร/ขาดทุน", f"${unrealized:,.2f}", f"{pl_pct:.2f}%", 
                 delta_color="normal")

    # 6.2 ส่วนบันทึก Journal (ส่งกลับไป Google Sheets)
    with st.expander(f"📝 Trading Journal & Update ({target_symbol})", expanded=True):
        col_input, col_note = st.columns([1, 2])
        
        with col_input:
            new_cost = st.number_input("แก้ต้นทุน ($):", value=my_cost, format="%.4f")
            new_qty = st.number_input("แก้จำนวนหุ้น:", value=my_qty, format="%.4f")
        
        with col_note:
            new_note = st.text_area("บันทึกช่วยจำ (Note):", value=my_note, height=100)
            
        if st.button("💾 อัปเดตข้อมูลกลับไปที่ Sheet"):
            # ส่งค่ากลับไปอัปเดต โดยใช้ชื่อหุ้นต้นฉบับ (ที่มี NASDAQ: ถ้ามี)
            update_specific_cell(selected_tab, real_sheet_symbol, 
                               cost=new_cost, qty=new_qty, note=new_note)
            st.rerun()

    # 6.3 กราฟราคา
    hist = yf.Ticker(target_symbol).history(period="1y", interval="1d")
    if not hist.empty:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name="Price"), row=1, col=1)
        # EMA
        fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'].ewm(span=20).mean(), name="EMA 20", line=dict(color='orange')), row=1, col=1)
        fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'].ewm(span=50).mean(), name="EMA 50", line=dict(color='blue')), row=1, col=1)
        # Volume
        v_colors = ['#26a69a' if c >= o else '#ef5350' for o, c in zip(hist['Open'], hist['Close'])]
        fig.add_trace(go.Bar(x=hist.index, y=hist['Volume'], marker_color=v_colors), row=2, col=1)
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"]), dict(bounds=[16, 9.5], pattern="hour")])
        fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    # 6.4 AI & News (รวมระบบเช็ค Error)
    st.divider()
    c_news, c_ai = st.columns(2)
    
    with c_news:
        st.subheader("📰 ข่าวที่เกี่ยวข้อง")
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(f"{target_symbol} stock news", max_results=5))
                news_text = ""
                if results:
                    for n in results:
                        st.markdown(f"- [{n['title']}]({n['href']})")
                        news_text += f"- {n['title']}\n"
                else: news_text = "No news found."
        except: news_text = "News Error."

    with c_ai:
        st.subheader("🤖 AI Analysis")
        if st.button("🔥 วิเคราะห์ด้วย AI", type="primary"):
            api_key = st.secrets.get("GEMINI_API_KEY")
            if api_key:
                with st.spinner("กำลังสแกนโมเดล..."):
                    models = ['models/gemini-2.5-flash', 'models/gemini-2.0-flash', 'models/gemini-1.5-pro']
                    success = False
                    for m in models:
                        try:
                            genai.configure(api_key=api_key)
                            model = genai.GenerativeModel(m)
                            prompt = f"หุ้น: {target_symbol} ราคา: ${curr_p:.2f} ข่าว: {news_text[:500]} วิเคราะห์แนวโน้มสั้นๆ ภาษาไทย"
                            res = model.generate_content(prompt)
                            st.success(f"Model: {m}")
                            st.markdown(res.text)
                            success = True
                            break
                        except: continue
                    if not success: st.error("AI ใช้งานไม่ได้ชั่วคราว (Quota เต็ม)")
            else: st.error("No API Key")

else:
    st.info("👈 เลือกหุ้นจากเมนูด้านซ้าย (ระบบจะอ่านชื่อ Tab: PORTFOLIO(DIME) และ PORTFOLIO(WEBULL) จากไฟล์ของคุณ)")
