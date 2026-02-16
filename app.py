import streamlit as st
import yfinance as yf
import google.generativeai as genai
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime

# ---------------------------------------------------------
# 1. ตั้งค่าหน้าเว็บ (Page Config)
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Stock Sniper Pro 📈",
    page_icon="🤖",
    layout="wide"  # ใช้พื้นที่เต็มจอ
)

# Custom CSS ให้ดูโปรขึ้น
st.markdown("""
<style>
    .metric-container {
        background-color: #1E1E1E;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #333;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. Sidebar: Settings & Watchlist
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Control Panel")
    
    # ช่องใส่ API Key (ใส่ครั้งเดียวจบ)
    api_key = st.text_input("🔑 ใส่ Gemini API Key ตรงนี้:", type="password")
    
    st.markdown("---")
    st.subheader("👀 My Watchlist")
    
    # ระบบ Watchlist (เก็บค่าไว้ใน Session)
    if 'watchlist' not in st.session_state:
        st.session_state.watchlist = ['AAPL', 'TSLA', 'NVDA', 'AMD']
    
    new_ticker = st.text_input("เพิ่มหุ้น (เช่น MSFT):").upper()
    if st.button("Add to Watchlist"):
        if new_ticker and new_ticker not in st.session_state.watchlist:
            st.session_state.watchlist.append(new_ticker)

    # แสดงรายการหุ้นใน Watchlist
    selected_ticker = st.radio("เลือกหุ้นที่ต้องการดู:", st.session_state.watchlist)
    
    if st.button("Clear Watchlist"):
        st.session_state.watchlist = []

# ---------------------------------------------------------
# 3. ฟังก์ชันดึงข้อมูล (Backend Logic)
# ---------------------------------------------------------
def get_stock_data(symbol):
    """ดึงข้อมูลราคาและประวัติ"""
    stock = yf.Ticker(symbol)
    # ดึงราคาย้อนหลัง 1 วัน (กราฟรายนาที) เพื่อความ Real-time
    history = stock.history(period="1d", interval="5m")
    info = stock.info
    return history, info, stock

def get_latest_news(stock_obj):
    try:
        news_list = stock_obj.news
        formatted_news = []
        if news_list:
            for n in news_list[:5]: # ลองดึงเยอะขึ้นเผื่อบางอันไม่มี title
                title = n.get('title') # ลองดึง title
                publisher = n.get('publisher')
                # เช็คว่ามี title จริงๆ ถึงจะเอามาโชว์
                if title and publisher: 
                    formatted_news.append(f"- {title} (Source: {publisher})")
        
        # ถ้าไม่มีข่าวเลย ให้บอกว่าไม่พบ
        return "\n".join(formatted_news) if formatted_news else "ไม่พบหัวข้อข่าวล่าสุด (Data Unavailable)"
    except Exception as e:
        return f"Error loading news: {str(e)}"

def ai_analyze(news_text, current_price, symbol):
    """ให้ AI วิเคราะห์ข่าวและกราฟ"""
    if not api_key:
        return "⚠️ กรุณาใส่ API Key ในแถบด้านซ้ายก่อนครับ"
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        prompt = f"""
        Role: คุณคือนักวิเคราะห์หุ้น Wall Street มืออาชีพ ที่เก่งเรื่อง Technical และ Fundamental
        Task: วิเคราะห์หุ้น {symbol} ที่ราคาปัจจุบัน ${current_price:.2f}
        
        News Context:
        {news_list}
        
        Output Requirement (ตอบเป็นภาษาไทย สั้น กระชับ):
        1. 📰 **สรุปข่าว:** (สรุปประเด็นสำคัญใน 1 บรรทัด)
        2. 🚦 **Sentiment:** (Bullish/Bearish/Neutral)
        3. 🎯 **Impact:** (ผลกระทบระยะสั้น: บวก/ลบ)
        4. 🛡️ **Support/Resistance:** (ประเมินแนวรับแนวต้านจิตวิทยา จากราคาปัจจุบัน)
        5. 💡 **Action:** (แนะนำ: Wait & See / Buy on Dip / Panic Sell)
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# ---------------------------------------------------------
# 4. หน้าจอหลัก (Main Dashboard)
# ---------------------------------------------------------
st.title(f"🚀 AI Stock Analysis: {selected_ticker}")

if selected_ticker:
    # ดึงข้อมูล
    try:
        df, info, stock_obj = get_stock_data(selected_ticker)
        
        if df.empty:
            st.error("ไม่พบข้อมูลหุ้น หรือตลาดปิด")
        else:
            # ส่วนแสดงราคา (Header Metrics)
            current_price = info.get('currentPrice', df['Close'].iloc[-1])
            previous_close = info.get('previousClose', df['Open'].iloc[0])
            delta = current_price - previous_close
            delta_percent = (delta / previous_close) * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="Current Price", value=f"${current_price:.2f}", delta=f"{delta:.2f} ({delta_percent:.2f}%)")
            with col2:
                st.metric(label="Day High", value=f"${df['High'].max():.2f}")
            with col3:
                st.metric(label="Day Low", value=f"${df['Low'].min():.2f}")

            # -----------------------------------------------------
            # ส่วนกราฟ (Interactive Chart)
            # -----------------------------------------------------
            fig = go.Figure(data=[go.Candlestick(x=df.index,
                            open=df['Open'],
                            high=df['High'],
                            low=df['Low'],
                            close=df['Close'])])
            
            fig.update_layout(title=f'{selected_ticker} Real-time Chart (5m Interval)',
                              yaxis_title='Price (USD)',
                              xaxis_rangeslider_visible=False,
                              template="plotly_dark") # ใช้ธีมมืดให้ดูโปร
            st.plotly_chart(fig, use_container_width=True)

            # -----------------------------------------------------
            # ส่วน AI Analysis & News
            # -----------------------------------------------------
            st.markdown("---")
            col_ai, col_news = st.columns([1, 1])

            news_list = get_latest_news(stock_obj)

            with col_ai:
                st.subheader("🤖 AI Analyst Insight")
                if st.button("⚡ กดเพื่อให้ AI วิเคราะห์เดี๋ยวนี้"):
                    with st.spinner('AI กำลังอ่านข่าวและดูกราฟ...'):
                        analysis_result = ai_analyze(news_list, current_price, selected_ticker)
                        st.success("วิเคราะห์เสร็จสิ้น!")
                        st.markdown(analysis_result)
                else:
                    st.info("กดปุ่มเพื่อเริ่มการวิเคราะห์ (ประหยัดโควต้า API)")

            with col_news:
                st.subheader("📰 Latest News Headlines")
                if news_list:
                    st.text_area("หัวข้อข่าวล่าสุด", news_list, height=200)
                else:
                    st.write("ไม่มีข่าวด่วนในช่วงนี้")

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาด: {e}")