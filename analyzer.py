import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import asyncio
from datetime import datetime, timedelta
import time

# Page config must be the first Streamlit command
st.set_page_config(
    page_title="Bybit Trading Analyzer 2026",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Gradient titles */
    .gradient-title {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        padding: 1rem 0;
    }
    
    /* Card styling */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        color: white;
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    /* Signal cards */
    .signal-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        border: 1px solid rgba(102, 126, 234, 0.1);
        transition: all 0.3s ease;
    }
    .signal-card:hover {
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.2);
        border-color: #667eea;
    }
    
    /* Badge styling */
    .badge-buy {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: 600;
        display: inline-block;
        animation: pulse 2s infinite;
    }
    .badge-sell {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-neutral {
        background: linear-gradient(135deg, #6b7280 0%, #4b5563 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: 600;
        display: inline-block;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    /* Progress bar styling */
    .confidence-bar {
        height: 10px;
        border-radius: 5px;
        background: linear-gradient(90deg, #10b981, #f59e0b, #ef4444);
        margin: 10px 0;
    }
    
    /* Info text */
    .info-text {
        color: #6b7280;
        font-size: 0.9rem;
        margin: 0.5rem 0;
    }
    
    /* Divider */
    .custom-divider {
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        height: 2px;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

from bybit_client import BybitClient
from strategies import SmartIndicatorStrategy   # MLStrategy removed
from market_regime import MarketRegimeDetector
from config import config

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.last_update = None
    st.session_state.data_cache = {}

# Initialize components
@st.cache_resource
def init_components():
    return {
        'client': BybitClient(),
        'indicator': SmartIndicatorStrategy(),
        'regime_detector': MarketRegimeDetector()
    }

components = init_components()
client = components['client']
indicator = components['indicator']
regime_detector = components['regime_detector']

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ **Dashboard Controls**")
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    
    # Symbol selector with icons
    symbol_icons = {
        'BTC/USDT': '₿',
        'ETH/USDT': 'Ξ',
        'SOL/USDT': '◎'
    }
    selected_symbol = st.selectbox(
        "**Select Asset**",
        config.SYMBOLS,
        format_func=lambda x: f"{symbol_icons.get(x, '')} {x}"
    )
    
    # Timeframe with visual indicators
    st.markdown("### ⏱️ **Time Settings**")
    col1, col2 = st.columns(2)
    with col1:
        selected_timeframe = st.selectbox(
            "Timeframe",
            config.TIMEFRAMES,
            index=3  # Default to 1h
        )
    with col2:
        lookback_days = st.slider(
            "Lookback",
            min_value=1,
            max_value=90,
            value=30,
            help="Number of days to analyze"
        )
    
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    
    # Market Regime Info
    st.markdown("### 🌊 **Market Regime Guide**")
    regime_info = {
        "Ranging / Low Volatility": "📊 Sideways movement, mean reversion expected",
        "Trending Bull / High Volatility": "📈 Strong upward momentum, trend following",
        "Trending Bear / High Volatility": "📉 Strong downward momentum, trend following"
    }
    
    for regime, desc in regime_info.items():
        with st.expander(regime):
            st.caption(desc)
    
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    
    # Update button with animation
    if st.button("🔄 **Refresh Data**", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    # Last update time
    if st.session_state.last_update:
        st.caption(f"Last updated: {st.session_state.last_update.strftime('%H:%M:%S')}")

# Main content
st.markdown("<h1 class='gradient-title'>📊 Bybit Trading Analyzer 2026</h1>", unsafe_allow_html=True)

# Loading data function
@st.cache_data(ttl=300, show_spinner="🚀 Fetching live market data...")
def load_data(symbol, timeframe, days):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Calculate limit based on timeframe
    tf_minutes = int(timeframe.replace('m', '').replace('h', '')) * (60 if 'h' in timeframe else 1)
    limit = days * 24 * 60 // tf_minutes
    
    data = loop.run_until_complete(client.fetch_ohlcv(symbol, timeframe, limit=min(limit, 1000)))
    loop.close()
    
    st.session_state.last_update = datetime.now()
    return data

# Load data with progress indicator
with st.spinner("📡 Connecting to Bybit..."):
    data = load_data(selected_symbol, selected_timeframe, lookback_days)

if data.empty:
    st.error("⚠️ No data available. Please try again later.")
    st.stop()

# Train models
with st.spinner("🤖 Training AI models..."):
    regime_detector.fit(data)
    # No ML model to train anymore

# Get current analysis
current_regime = regime_detector.predict_regime(data)
regime_name = regime_detector.get_regime_name(current_regime)
characteristics = regime_detector.get_regime_characteristics(data)
indicator_signal = indicator.generate_signal(selected_symbol, data)

# Top metrics row
st.markdown("## 📈 **Market Overview**")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    current_price = data['close'].iloc[-1]
    price_change = ((data['close'].iloc[-1] / data['close'].iloc[-2] - 1) * 100)
    delta_color = "inverse" if price_change >= 0 else "normal"
    
    st.markdown(f"""
    <div class='metric-card'>
        <div style='font-size: 0.9rem; opacity: 0.9;'>Current Price</div>
        <div style='font-size: 1.8rem; font-weight: 700;'>${current_price:,.2f}</div>
        <div style='color: {"#10b981" if price_change >= 0 else "#ef4444"}; font-weight: 600;'>
            {"▲" if price_change >= 0 else "▼"} {abs(price_change):.2f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class='metric-card'>
        <div style='font-size: 0.9rem; opacity: 0.9;'>Market Regime</div>
        <div style='font-size: 1.2rem; font-weight: 700;'>{regime_name}</div>
        <div style='font-size: 0.9rem; opacity: 0.9;'>Volatility: {characteristics['volatility']:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    volume_24h = data['volume'].tail(24).sum()
    avg_volume = data['volume'].tail(24*7).mean() if len(data) > 24*7 else data['volume'].mean()
    volume_ratio = volume_24h / avg_volume if avg_volume > 0 else 1
    
    st.markdown(f"""
    <div class='metric-card'>
        <div style='font-size: 0.9rem; opacity: 0.9;'>24h Volume</div>
        <div style='font-size: 1.2rem; font-weight: 700;'>${volume_24h:,.0f}</div>
        <div style='color: {"#10b981" if volume_ratio > 1.2 else "#f59e0b"};'>
            {volume_ratio:.1f}x average
        </div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    high_24h = data['high'].tail(24).max()
    low_24h = data['low'].tail(24).min()
    range_pct = ((high_24h - low_24h) / low_24h) * 100
    
    st.markdown(f"""
    <div class='metric-card'>
        <div style='font-size: 0.9rem; opacity: 0.9;'>24h Range</div>
        <div style='font-size: 1rem; font-weight: 700;'>${low_24h:.2f} - ${high_24h:.2f}</div>
        <div style='font-size: 0.9rem;'>{range_pct:.1f}% range</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    trend_strength = characteristics['trend_strength']
    trend_icon = "📈" if trend_strength > 0.5 else "📉" if trend_strength < -0.5 else "📊"
    
    st.markdown(f"""
    <div class='metric-card'>
        <div style='font-size: 0.9rem; opacity: 0.9;'>Trend Strength</div>
        <div style='font-size: 2rem;'>{trend_icon}</div>
        <div style='font-size: 0.9rem;'>{abs(trend_strength):.2f}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

# Price Chart
st.markdown("## 📊 **Advanced Chart Analysis**")

# Create advanced candlestick chart with indicators
fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    row_heights=[0.6, 0.2, 0.2],
    subplot_titles=("Price Action", "Volume", "RSI")
)

# Candlestick chart with gradient colors
fig.add_trace(go.Candlestick(
    x=data.index,
    open=data['open'],
    high=data['high'],
    low=data['low'],
    close=data['close'],
    name="OHLC",
    increasing_line_color='#10b981',
    decreasing_line_color='#ef4444',
    increasing_fillcolor='rgba(16, 185, 129, 0.1)',
    decreasing_fillcolor='rgba(239, 68, 68, 0.1)'
), row=1, col=1)

# Add moving averages
data['sma_20'] = data['close'].rolling(20).mean()
data['sma_50'] = data['close'].rolling(50).mean()

fig.add_trace(go.Scatter(
    x=data.index, y=data['sma_20'],
    name="SMA 20",
    line=dict(color='#667eea', width=2, dash='dash')
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=data.index, y=data['sma_50'],
    name="SMA 50",
    line=dict(color='#764ba2', width=2, dash='dash')
), row=1, col=1)

# Volume bars with color based on price movement
colors = ['#10b981' if data['close'].iloc[i] >= data['open'].iloc[i] else '#ef4444' 
          for i in range(len(data))]

fig.add_trace(go.Bar(
    x=data.index, y=data['volume'],
    name="Volume",
    marker_color=colors,
    opacity=0.7
), row=2, col=1)

# Add volume moving average
data['volume_sma'] = data['volume'].rolling(20).mean()
fig.add_trace(go.Scatter(
    x=data.index, y=data['volume_sma'],
    name="Volume MA",
    line=dict(color='#f59e0b', width=2)
), row=2, col=1)

# RSI
data['rsi'] = data['close'].diff().rolling(14).apply(
    lambda x: 100 - (100 / (1 + (x[x > 0].mean() / -x[x < 0].mean()))) if len(x) > 0 else 50
)

fig.add_trace(go.Scatter(
    x=data.index, y=data['rsi'],
    name="RSI",
    line=dict(color='#10b981', width=2)
), row=3, col=1)

# Add RSI levels
fig.add_hline(y=70, line_dash="dash", line_color="#ef4444", opacity=0.5, row=3, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="#10b981", opacity=0.5, row=3, col=1)
fig.add_hline(y=50, line_dash="dot", line_color="#6b7280", opacity=0.3, row=3, col=1)

# Update layout
fig.update_layout(
    template='plotly_dark',
    height=800,
    showlegend=True,
    hovermode='x unified',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#e5e7eb'),
    legend=dict(
        bgcolor='rgba(0,0,0,0.5)',
        bordercolor='#667eea',
        borderwidth=1
    )
)

fig.update_xaxes(gridcolor='rgba(102, 126, 234, 0.1)', gridwidth=1)
fig.update_yaxes(gridcolor='rgba(102, 126, 234, 0.1)', gridwidth=1)

st.plotly_chart(fig, use_container_width=True)

st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

# Strategy Signals
st.markdown("## 🤖 **AI Trading Signals**")

# Determine badge class and text based on signal value using config constants
if indicator_signal:
    if indicator_signal.signal == config.SIGNAL_BUY:
        badge_class = "badge-buy"
        badge_text = "BUY 🔥"
    elif indicator_signal.signal == config.SIGNAL_SELL:
        badge_class = "badge-sell"
        badge_text = "SELL ⚡"
    else:
        badge_class = "badge-neutral"
        badge_text = "HOLD 💤"
else:
    badge_class = "badge-neutral"
    badge_text = "HOLD 💤"

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    <div class='signal-card'>
        <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;'>
            <span style='font-size: 1.3rem; font-weight: 700;'>📊 Smart Indicator</span>
            <span class='{badge_class}'>{badge_text}</span>
        </div>
    """, unsafe_allow_html=True)
    
    if indicator_signal:
        # Confidence bar
        confidence_pct = indicator_signal.confidence * 100
        st.markdown(f"""
        <div style='margin: 1rem 0;'>
            <div style='display: flex; justify-content: space-between; margin-bottom: 0.5rem;'>
                <span>Confidence</span>
                <span style='font-weight: 700;'>{confidence_pct:.1f}%</span>
            </div>
            <div class='confidence-bar'>
                <div style='width: {confidence_pct}%; height: 100%; background: linear-gradient(90deg, #10b981, #f59e0b, #ef4444); border-radius: 5px;'></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Reasons
        if indicator_signal.reasons:
            st.markdown("**Analysis Reasons:**")
            for reason in indicator_signal.reasons:
                st.markdown(f"• {reason}")
        
        st.markdown(f"**Signal Strength:** {indicator_signal.strength:.1f}/10")
    else:
        st.info("No signal generated - waiting for clear patterns")
    
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    # Placeholder for other signals (RL agent not shown)
    st.markdown(f"""
    <div class='signal-card'>
        <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;'>
            <span style='font-size: 1.3rem; font-weight: 700;'>🤖 RL Master</span>
            <span class='badge-neutral'>HOLD 💤</span>
        </div>
        <div style='text-align: center; padding: 1rem;'>
            <p style='color: #6b7280;'>RL agent signals are processed internally and not shown in this dashboard.</p>
            <p style='color: #6b7280; font-size: 0.9rem;'>Check console logs for RL Master output.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

# Market Analysis Dashboard
st.markdown("## 📋 **Deep Market Analysis**")

tab1, tab2, tab3 = st.tabs(["📈 Technical Indicators", "📊 Market Statistics", "📜 Signal History"])

with tab1:
    # Technical indicators grid
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate indicators
    latest = data.iloc[-1]
    
    with col1:
        rsi_value = data['rsi'].iloc[-1] if 'rsi' in data.columns else 50
        rsi_color = "#10b981" if rsi_value < 30 else "#ef4444" if rsi_value > 70 else "#f59e0b"
        
        st.markdown(f"""
        <div style='background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 10px; text-align: center;'>
            <div style='color: #6b7280; font-size: 0.9rem;'>RSI (14)</div>
            <div style='font-size: 2rem; font-weight: 700; color: {rsi_color};'>{rsi_value:.1f}</div>
            <div style='color: {rsi_color};'>{'Oversold' if rsi_value < 30 else 'Overbought' if rsi_value > 70 else 'Neutral'}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # MACD
        macd = latest.get('macd', 0)
        macd_signal = latest.get('macd_signal', 0)
        macd_diff = macd - macd_signal
        
        st.markdown(f"""
        <div style='background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 10px; text-align: center;'>
            <div style='color: #6b7280; font-size: 0.9rem;'>MACD</div>
            <div style='font-size: 1.2rem; font-weight: 700;'>{macd:.2f}</div>
            <div style='color: {"#10b981" if macd_diff > 0 else "#ef4444"};'>
                {'Bullish' if macd_diff > 0 else 'Bearish'} crossover
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Bollinger Position
        bb_position = latest.get('bb_position', 0.5)
        
        st.markdown(f"""
        <div style='background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 10px; text-align: center;'>
            <div style='color: #6b7280; font-size: 0.9rem;'>Bollinger Position</div>
            <div style='font-size: 2rem; font-weight: 700;'>{bb_position:.2f}</div>
            <div>{'Lower band' if bb_position < 0.2 else 'Upper band' if bb_position > 0.8 else 'Middle'}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        # Volume Profile
        volume_ratio = latest.get('volume_ratio', 1)
        
        st.markdown(f"""
        <div style='background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 10px; text-align: center;'>
            <div style='color: #6b7280; font-size: 0.9rem;'>Volume Ratio</div>
            <div style='font-size: 2rem; font-weight: 700;'>{volume_ratio:.1f}x</div>
            <div style='color: {"#10b981" if volume_ratio > 1.2 else "#f59e0b"};'>
                {'High' if volume_ratio > 1.2 else 'Normal'} volume
            </div>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    # Market statistics
    col1, col2 = st.columns(2)
    
    with col1:
        # Returns distribution
        returns = data['close'].pct_change().dropna()
        
        fig_hist = px.histogram(
            returns, 
            nbins=50,
            title="Returns Distribution",
            labels={'value': 'Return', 'count': 'Frequency'},
            color_discrete_sequence=['#667eea']
        )
        fig_hist.update_layout(
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        # Volatility over time
        data['volatility_20'] = returns.rolling(20).std() * np.sqrt(252)
        
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(
            x=data.index, y=data['volatility_20'],
            fill='tozeroy',
            line=dict(color='#764ba2', width=2),
            name='Volatility (20d)'
        ))
        fig_vol.update_layout(
            template='plotly_dark',
            title="Rolling Volatility (20-day)",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            yaxis_title="Annualized Volatility"
        )
        st.plotly_chart(fig_vol, use_container_width=True)

with tab3:
    # Signal history placeholder
    st.info("📝 Signal history will appear here as the bot runs longer")
    
    # Create sample history if none exists
    if hasattr(client, 'trade_history') and client.trade_history:
        history_df = pd.DataFrame(client.trade_history)
        st.dataframe(
            history_df,
            use_container_width=True,
            column_config={
                "timestamp": "Time",
                "symbol": "Symbol",
                "type": "Type",
                "price": st.column_config.NumberColumn("Price", format="$%.2f"),
                "amount": st.column_config.NumberColumn("Amount", format="%.6f")
            }
        )
    else:
        st.caption("No signals generated yet. Let the bot run longer to see history.")

# Footer
st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.caption(f"🔧 Mode: {'🟢 DRY RUN (Testing)' if config.DRY_RUN else '🔴 LIVE TRADING'}")
with col2:
    st.caption(f"🔄 Auto-refresh: Every 5 minutes")
with col3:
    if st.session_state.last_update:
        st.caption(f"⏱️ Data age: {(datetime.now() - st.session_state.last_update).seconds}s")

# Cleanup on exit
import atexit
atexit.register(lambda: asyncio.run(client.close()))