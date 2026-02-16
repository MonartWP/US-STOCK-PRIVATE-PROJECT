import streamlit as st
import yfinance as yf
import google.generativeai as genai
import plotly.graph_objects as go
from duckduckgo_search import DDGS

# ---------------------------------------------------------
# 1. ตั้งค่าหน้าเว็บ (Page Config)
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Stock Sniper Pro 📈",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
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
    
    # เช็ค API Key
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
        st.success("✅ เชื่อมต่อ API Key จากระบบสำเร็จ")
    else:
        api_key = st.text_input("🔑 ใส่ Gemini API Key ตรงนี้:", type="password")
    
    st.markdown("---")
    st.subheader("👀 My Watchlist")
    
    if 'watchlist' not in st.session_state:
        st.session_state.watchlist = ['AAPL', 'TSLA', 'NVDA', 'AMD']
    
    new_ticker = st.text_input("เพิ่มหุ้น (เช่น MSFT):").upper()
    if st.button("Add"):
        if new_ticker and new_ticker not in st.session_state.watchlist:
            st.session_state.watchlist.append(new_ticker)

    selected_ticker = st.radio("เลือกหุ้น:", st.session_state.watchlist)
    
    if st.button("Clear Watchlist"):
        st.session_state.watchlist = []

# ---------------------------------------------------------
# 3. ฟังก์ชันดึงข้อมูล (Backend Logic)
# ---------------------------------------------------------
def get_stock_data(symbol):
    """ดึงข้อมูลราคาและประวัติ"""
    stock = yf.Ticker(symbol)
    history = stock.history(period="1d", interval="5m")
    info = stock.info
    return history, info

def get_latest_news(symbol):
    """ดึงข่าวด่วนโดยใช้ DuckDuckGo Search"""
    try:
        formatted_news = []
        with DDGS() as ddgs:
            # ค้นหาข่าว 5 อันดับแรก
            results = list(ddgs.text(f"{symbol} stock news", max_results=5))
            
            if results:
                for news in results:
                    title = news.get('title')
                    link = news.get('href')
                    if title and link:
                        formatted_news.append(f"- [{title}]({link})")
            
        if not formatted_news:
            return "ไม่พบข่าวใหม่ในขณะนี้ (No recent news found)"
            
        return "\n".join(formatted_news)
        
    except Exception as e:
        return f"Error searching news: {str(e)}"

def ai_analyze(news_text, current_price, symbol):
    """ให้ AI วิเคราะห์ข่าวและกราฟ"""
    if not api_key:
        return "⚠️ กรุณาใส่ API Key ในแถบด้านซ้ายก่อนครับ"
    
    try:
        # ใช้ Model 2.5 Flash ตามที่คุณมีสิทธิ์
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('models/gemini-2.5-flash')
        
        prompt = f"""
        Role: คุณคือนักวิเคราะห์หุ้น Wall Street มืออาชีพ
        Task: วิเคราะห์หุ้น {symbol} ที่ราคาปัจจุบัน ${current_price:.2f}
        
        News Context:
        {news_text}
        
        Output Requirement (ตอบเป็นภาษาไทย สั้น กระชับ):
        1. 📰 **สรุปข่าว:** (สรุปประเด็นสำคัญ)
        2. 🚦 **Sentiment:** (Bullish/Bearish/Neutral)
        3. 🎯 **Impact:** (ผลกระทบระยะสั้น: บวก/ลบ)
        4. 🛡️ **Support/Resistance:** (ประเมินแนวรับแนวต้าน)
        5. 💡 **Action:** (แนะนำ: Wait / Buy / Sell)
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
    try:
        df, info = get_stock_data(selected_ticker)
        
        if df.empty:
            st.error("ไม่พบข้อมูลหุ้น หรือตลาดปิด")
        else:
            # ส่วนแสดงราคา
            current_price = info.get('currentPrice', df['Close'].iloc[-1])
            previous_close = info.get('previousClose', df['Open'].iloc[0])
            delta = current_price - previous_close
            delta_percent = (delta / previous_close) * 100
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Current Price", f"${current_price:.2f}", f"{delta:.2f} ({delta_percent:.2f}%)")
            col2.metric("Day High", f"${df['High'].max():.2f}")
            col3.metric("Day Low", f"${df['Low'].min():.2f}")

            # กราฟ
            fig = go.Figure(data=[go.Candlestick(x=df.index,
                            open=df['Open'], high=df['High'],
                            low=df['Low'], close=df['Close'])])
            
            fig.update_layout(title=f'{selected_ticker} Real-time Chart',
                              yaxis_title='Price (USD)',
                              template="plotly_dark",
                              height=500)
            st.plotly_chart(fig, use_container_width=True)

            # ส่วน AI และข่าว
            st.markdown("---")
            col_ai, col_news = st.columns([1, 1])
            
            # ดึงข่าว (แก้เป็นส่งชื่อหุ้นตรงๆ)
            news_list = get_latest_news(selected_ticker)

            with col_ai:
                st.subheader("🤖 AI Analyst Insight")
                if st.button("⚡ วิเคราะห์เดี๋ยวนี้"):
                    with st.spinner('AI กำลังทำงาน...'):
                        result = ai_analyze(news_list, current_price, selected_ticker)
                        st.success("เรียบร้อย!")
                        st.markdown(result)

            with col_news:
                st.subheader("📰 ข่าวล่าสุด")
                st.info(news_list)

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาด: {e}")
