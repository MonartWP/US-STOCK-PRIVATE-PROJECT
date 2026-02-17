import streamlit as st
import yfinance as yf
import google.generativeai as genai
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_gsheets import GSheetsConnection
from duckduckgo_search import DDGS
import pandas as pd
import requests
import datetime
import time

# --- 1. ตั้งค่าหน้าเว็บ & Session State ---
st.set_page_config(page_title="Port to TheMoon Commander 🚀", layout="wide")
st.markdown("""
<style>
    .stMetric { background-color: #1E1E1E; border: 1px solid #333; border-radius: 10px; padding: 10px; }
    div[data-testid="stExpander"] { background-color: #262730; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# เริ่มต้นตัวแปรความจำ (เพื่อให้ผล AI ไม่หายเวลากดปุ่มอื่น)
if 'ai_sentiment_result' not in st.session_state:
    st.session_state.ai_sentiment_result = None
if 'ai_tactical_result' not in st.session_state:
    st.session_state.ai_tactical_result = None
if 'news_cache' not in st.session_state:
    st.session_state.news_cache = ""

# --- 2. เชื่อมต่อ Google Sheets ---
try:
    conn = st.connection("gsheets", type=GSheetsConnection)
except Exception as e:
    st.error(f"❌ Connection Error: {e}")

# --- 3. ฟังก์ชันและ Helper ---
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

def clean_symbol(sym):
    if isinstance(sym, str):
        return sym.split(":")[-1].strip()
    return str(sym)

def get_sheet_data(tab_name):
    try:
        df = conn.read(worksheet=tab_name, ttl=0)
        # ตรวจสอบจำนวนคอลัมน์ (ต้องมีอย่างน้อย 11 เพื่อดึง Note ใน col K)
        if len(df.columns) >= 10: 
            # Col: A=0, C=2, D=3, K=10
            needed_cols = df.iloc[:, [0, 2, 3, 10]].copy()
            needed_cols.columns = ['raw_symbol', 'qty', 'cost', 'note']
            needed_cols['symbol'] = needed_cols['raw_symbol'].apply(clean_symbol)
            needed_cols['qty'] = pd.to_numeric(needed_cols['qty'], errors='coerce').fillna(0.0)
            needed_cols['cost'] = pd.to_numeric(needed_cols['cost'], errors='coerce').fillna(0.0)
            needed_cols['note'] = needed_cols['note'].fillna("")
            return needed_cols[needed_cols['symbol'] != ""]
        return pd.DataFrame()
    except:
        return pd.DataFrame(columns=['symbol', 'qty', 'cost', 'note'])

def update_specific_cell(tab_name, symbol, cost=None, qty=None, note=None):
    try:
        sh = conn.client.open_by_url(st.secrets["connections"]["gsheets"]["spreadsheet"])
        wks = sh.worksheet(tab_name)
        cell = wks.find(symbol, in_column=1)
        if cell:
            row = cell.row
            if qty is not None: wks.update_cell(row, 3, qty)
            if cost is not None: wks.update_cell(row, 4, cost)
            if note is not None: wks.update_cell(row, 11, note)
            st.toast(f"💾 อัปเดต {symbol} แล้ว", icon="✅")
        else:
            new_row = [symbol, "", qty or 0, cost or 0] + [""]*6 + [note or ""]
            wks.append_row(new_row)
            st.toast(f"✨ เพิ่ม {symbol} ใหม่", icon="🆕")
    except Exception as e:
        st.error(f"Update Error: {e}")

def add_transaction(date, broker, symbol, action, qty, price, fees, ex_rate, total_thb, total_amt, note):
    try:
        sh = conn.client.open_by_url(st.secrets["connections"]["gsheets"]["spreadsheet"])
        wks = sh.worksheet("TRANSACTIONS") # ต้องชื่อ Tab นี้เป๊ะๆ
        # เรียงลำดับข้อมูลตาม Column ในรูปภาพที่คุณส่งมา (A-K)
        row_data = [
            str(date),      # A: Date
            broker,         # B: Broker
            symbol,         # C: Symbol
            action,         # D: Action
            qty,            # E: Quantity
            price,          # F: Price
            fees,           # G: Fees
            ex_rate,        # H: Exchange Rate
            total_thb,      # I: Total Paid (THB)
            total_amt,      # J: Total Amount
            note            # K: Note
        ]
        wks.append_row(row_data)
        st.success(f"✅ บันทึกรายการ {action} {symbol} เรียบร้อย!")
        time.sleep(1)
        st.rerun()
    except Exception as e:
        st.error(f"Transaction Error: {e}")

# --- 4. Sidebar Main Menu ---
with st.sidebar:
    st.title("🌕 Commander")
    page = st.radio("เมนูหลัก", ["📊 Portfolio Analysis", "📝 Transaction Logger"])
    st.divider()

# ==========================================
# PAGE 1: PORTFOLIO ANALYSIS
# ==========================================
if page == "📊 Portfolio Analysis":
    with st.sidebar:
        port_map = {"Dime": "PORTFOLIO(DIME)", "Webull": "PORTFOLIO(WEBULL)"}
        selected_key = st.selectbox("เลือกพอร์ตดูข้อมูล:", list(port_map.keys()))
        selected_tab = port_map[selected_key]
        
        df_port = get_sheet_data(selected_tab)
        watchlist = df_port['symbol'].tolist() if not df_port.empty else []
        
        if watchlist:
            target_symbol = st.radio("หุ้นในพอร์ต:", watchlist)
        else:
            target_symbol = None
            st.info("ไม่พบหุ้น")

    if target_symbol:
        # ล้างค่า AI เก่าทิ้ง ถ้าเปลี่ยนหุ้น
        if 'last_symbol' not in st.session_state or st.session_state.last_symbol != target_symbol:
            st.session_state.ai_sentiment_result = None
            st.session_state.ai_tactical_result = None
            st.session_state.news_cache = ""
            st.session_state.last_symbol = target_symbol

        row_data = df_port[df_port['symbol'] == target_symbol].iloc[0]
        curr_qty = float(row_data['qty'])
        curr_cost = float(row_data['cost'])
        curr_note = str(row_data['note'])
        real_sym = row_data['raw_symbol']

        st.title(f"🚀 {target_symbol} Analysis")
        
        # --- Market Data & Chart ---
        raw = yf.Ticker(target_symbol).history(period="5d")
        if not raw.empty:
            curr_p = raw['Close'].iloc[-1]
            chg = curr_p - raw['Close'].iloc[-2]
            pct = (chg / raw['Close'].iloc[-2]) * 100
            
            mkt_val = curr_p * curr_qty
            tot_cost = curr_cost * curr_qty
            unrealized = mkt_val - tot_cost
            pl_pct = (unrealized / tot_cost * 100) if tot_cost > 0 else 0
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Price", f"${curr_p:.2f}", f"{pct:.2f}%")
            c2.metric("Market Value", f"${mkt_val:,.2f}")
            c3.metric("Total Cost", f"${tot_cost:,.2f}")
            c4.metric("P/L", f"${unrealized:,.2f}", f"{pl_pct:.2f}%", delta_color="normal")

            # Chart
            hist = yf.Ticker(target_symbol).history(period="1y")
            if not hist.empty:
                hist.index = hist.index.tz_localize(None)
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7,0.3])
                fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name="Price"), row=1, col=1)
                fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'].ewm(span=20).mean(), name="EMA20", line=dict(color='orange')), row=1, col=1)
                fig.add_trace(go.Bar(x=hist.index, y=hist['Volume'], name="Vol"), row=2, col=1)
                fig.update_layout(height=500, template="plotly_dark", xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)

        # --- AI Section (Unified) ---
        st.divider()
        col_news, col_ai = st.columns([1, 1])
        
        with col_news:
            st.subheader("📰 ข่าวล่าสุด")
            # โหลดข่าวแค่ครั้งเดียวแล้วจำไว้
            if not st.session_state.news_cache:
                try:
                    with DDGS() as ddgs:
                        results = list(ddgs.text(f"{target_symbol} stock news", max_results=5))
                        if results:
                            txt = ""
                            for n in results:
                                st.markdown(f"- [{n['title']}]({n['href']})")
                                txt += f"- {n['title']}\n"
                            st.session_state.news_cache = txt
                        else:
                            st.session_state.news_cache = "No news found."
                            st.info("ไม่พบข่าว")
                except:
                    st.session_state.news_cache = "News Error"
            else:
                # แสดงข่าวจาก Cache (ถ้ามี) แบบ Link
                # (ในที่นี้แสดงแบบ Text เพื่อความง่าย หรือคุณจะ parse กลับมาก็ได้)
                st.info("ข่าวโหลดแล้ว (พร้อมสำหรับ AI)")

        with col_ai:
            st.subheader("🤖 Intelligence Center")
            
            # ปุ่มกดวิเคราะห์ (แยกกันแต่อยู่ด้วยกัน)
            c_btn1, c_btn2 = st.columns(2)
            if c_btn1.button("🔥 Sentiment Score", use_container_width=True):
                api_key = st.secrets.get("GEMINI_API_KEY")
                if api_key:
                    with st.spinner("Giving Score..."):
                        try:
                            genai.configure(api_key=api_key)
                            model = genai.GenerativeModel('models/gemini-2.0-flash')
                            prompt = f"Analyze news for {target_symbol}: {st.session_state.news_cache}. Give Score 0-100 (0=Bear,100=Bull). Format: SCORE: 80"
                            res = model.generate_content(prompt)
                            # จำใส่ Session State
                            st.session_state.ai_sentiment_result = res.text
                        except Exception as e: st.error(str(e))

            if c_btn2.button("🧠 Deep Tactics", use_container_width=True):
                api_key = st.secrets.get("GEMINI_API_KEY")
                if api_key:
                    with st.spinner("Analyzing..."):
                        try:
                            genai.configure(api_key=api_key)
                            model = genai.GenerativeModel('models/gemini-2.0-flash')
                            prompt = f"Analyze {target_symbol} price ${curr_p}. News: {st.session_state.news_cache}. Short tactical advice (Thai)."
                            res = model.generate_content(prompt)
                            # จำใส่ Session State
                            st.session_state.ai_tactical_result = res.text
                        except Exception as e: st.error(str(e))

            # --- ส่วนแสดงผล (Display Results form Session State) ---
            # 1. Sentiment Result
            if st.session_state.ai_sentiment_result:
                st.markdown("---")
                text = st.session_state.ai_sentiment_result
                import re
                match = re.search(r"SCORE: (\d+)", text)
                score = int(match.group(1)) if match else 50
                st.metric("AI Score", f"{score}/100", delta=score-50)
                st.progress(score)
                if score >= 70: st.success("Bullish 🐂")
                elif score <= 30: st.error("Bearish 🐻")
                else: st.warning("Neutral ⚖️")

            # 2. Tactical Result
            if st.session_state.ai_tactical_result:
                st.markdown("---")
                st.caption("Strategic Advice")
                st.markdown(st.session_state.ai_tactical_result)

    else:
        st.info("เลือกหุ้นจาก Sidebar")

# ==========================================
# PAGE 2: TRANSACTION LOGGER (New Feature)
# ==========================================
elif page == "📝 Transaction Logger":
    st.header("บันทึกธุรกรรม (Transactions)")
    st.info("ข้อมูลจะถูกบันทึกลงใน Tab: **TRANSACTIONS** ใน Google Sheets")

    with st.form("trans_form"):
        c1, c2, c3 = st.columns(3)
        date = c1.date_input("Date", datetime.date.today())
        broker = c2.selectbox("Broker", ["Dime!", "Webull", "InnovestX", "Streaming"])
        action = c3.selectbox("Action", ["Buy", "Sell", "Dividend"])
        
        c4, c5 = st.columns(2)
        symbol = c4.text_input("Symbol (e.g. NVDA, NASDAQ:RKLB)").upper().strip()
        exchange_rate = c5.number_input("Exchange Rate (THB/USD)", value=34.50)
        
        st.divider()
        c6, c7, c8 = st.columns(3)
        qty = c6.number_input("Quantity", value=0.0, format="%.6f")
        price = c7.number_input("Price ($)", value=0.0, format="%.4f")
        fees = c8.number_input("Fees ($)", value=0.0, format="%.2f")
        
        # คำนวณยอดรวมให้อัตโนมัติ (Visual Only)
        total_amt_usd = (qty * price) + fees
        total_thb_est = total_amt_usd * exchange_rate
        note = st.text_area("Note / Reason")
        
        st.markdown(f"**ยอดรวมโดยประมาณ:** ${total_amt_usd:,.2f} (~฿{total_thb_est:,.2f})")
        
        submitted = st.form_submit_button("💾 บันทึก Transaction", type="primary")
        
        if submitted:
            if symbol and qty > 0 and price > 0:
                # เรียกฟังก์ชันบันทึก
                add_transaction(
                    date, broker, symbol, action, 
                    qty, price, fees, exchange_rate, 
                    total_thb_est, total_amt_usd, note
                )
            else:
                st.error("กรุณากรอก Symbol, Quantity และ Price ให้ครบถ้วน")
