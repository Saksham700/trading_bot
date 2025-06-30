import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import google.generativeai as genai
import time
import ta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

GOOGLE_API_KEY = "AIzaSyB46mW-7p4MIrKSe-oudQLpjxWli6XjVpE"
genai.configure(api_key=GOOGLE_API_KEY)

class TradingSignalBot:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-2.0-flash-001')
        self.previous_signals = {}     
        self.optimal_trading_times = {
            "EURUSD=X": ["06:00-09:00 UTC", "12:00-15:00 UTC"],
            "GBPUSD=X": ["07:00-10:00 UTC", "12:00-15:00 UTC"],
            "USDJPY=X": ["23:00-02:00 UTC", "11:00-14:00 UTC"],
            "USDCHF=X": ["07:00-10:00 UTC", "13:00-16:00 UTC"],
            "AUDUSD=X": ["21:00-00:00 UTC", "04:00-07:00 UTC"],
            "USDCAD=X": ["12:00-15:00 UTC", "18:00-21:00 UTC"],
            "NZDUSD=X": ["21:00-00:00 UTC", "04:00-07:00 UTC"],
            "EURJPY=X": ["07:00-10:00 UTC", "12:00-15:00 UTC"],
            "GBPJPY=X": ["07:00-10:00 UTC", "12:00-15:00 UTC"],
            "EURGBP=X": ["07:00-10:00 UTC", "12:00-15:00 UTC"]
        }
        
    def get_next_trading_window(self, symbol: str) -> Tuple[str, timedelta]:
        """Get the next optimal trading window and time until it starts"""
        now_utc = datetime.utcnow()
        current_time = now_utc.strftime("%H:%M")
        
        if symbol not in self.optimal_trading_times:
            return "No optimal time data", timedelta(0)
            
        windows = self.optimal_trading_times[symbol]
        for window in windows:
            # Extract just the time part (remove "UTC")
            time_part = window.split(' UTC')[0].strip()
            start_str, end_str = time_part.split('-')
            
            # Parse times without UTC suffix
            start_time = datetime.strptime(start_str.strip(), "%H:%M").time()
            end_time = datetime.strptime(end_str.strip(), "%H:%M").time()
            
            # Handle overnight sessions
            if start_str > end_str:
                # Session crosses midnight
                if now_utc.time() >= start_time or now_utc.time() <= end_time:
                    return f"NOW! (Ends at {end_str} UTC)", timedelta(0)
            else:
                # Normal session within same day
                if start_time <= now_utc.time() <= end_time:
                    return f"NOW! (Ends at {end_str} UTC)", timedelta(0)
                    
            # If window is later today
            if now_utc.time() < start_time:
                start_dt = datetime.combine(now_utc.date(), start_time)
                time_left = start_dt - now_utc
                return f"Today at {start_str} UTC", time_left
                
        # If next window is tomorrow
        next_window = windows[0]
        time_part = next_window.split(' UTC')[0].strip()
        start_str = time_part.split('-')[0].strip()
        start_time = datetime.strptime(start_str, "%H:%M").time()
        start_dt = datetime.combine(now_utc.date() + timedelta(days=1), start_time)
        time_left = start_dt - now_utc
        return f"Tomorrow at {start_str} UTC", time_left

    def get_market_data(self, symbol: str, period: str = "1d", interval: str = "2m") -> pd.DataFrame:
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            return data
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()  
    
    def calculate_support_resistance(self, data: pd.DataFrame, window: int = 20) -> Tuple[List, List]:
        if data.empty:
            return [], []      
        highs = data['High'].rolling(window=window, center=True).max()
        lows = data['Low'].rolling(window=window, center=True).min()      
        resistance_levels = []
        support_levels = []       
        for i in range(window, len(data) - window):
            if data['High'].iloc[i] == highs.iloc[i]:
                resistance_levels.append((data.index[i], data['High'].iloc[i]))               
            if data['Low'].iloc[i] == lows.iloc[i]:
                support_levels.append((data.index[i], data['Low'].iloc[i]))       
        return support_levels, resistance_levels   
    
    def check_breakout_signals(self, data: pd.DataFrame, support_levels: List, resistance_levels: List) -> Dict:
        if data.empty or len(data) < 10:
            return {"signal": "NO DATA", "strength": 0, "reason": "Insufficient data"}   
        current_price = data['Close'].iloc[-1]
        previous_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
        price_change = ((current_price - previous_price) / previous_price) * 100
        current_volume = data['Volume'].iloc[-1] if 'Volume' in data.columns else 0
        avg_volume = data['Volume'].rolling(window=20).mean().iloc[-1] if 'Volume' in data.columns else 0
        volume_spike = current_volume > (avg_volume * 1.5) if avg_volume > 0 else False
        try:
            data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
            data['MACD'] = ta.trend.MACD(data['Close']).macd()
            data['MACD_signal'] = ta.trend.MACD(data['Close']).macd_signal()
            bb_indicator = ta.volatility.BollingerBands(data['Close'])
            data['BB_upper'] = bb_indicator.bollinger_hband()
            data['BB_lower'] = bb_indicator.bollinger_lband()
            data['BB_middle'] = bb_indicator.bollinger_mavg()
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['EMA_12'] = data['Close'].ewm(span=12).mean()
            data['EMA_26'] = data['Close'].ewm(span=26).mean()           
        except Exception as e:
            st.warning(f"Technical indicator calculation error: {e}")
            return {"signal": "ERROR", "strength": 0, "reason": "Technical analysis failed"}
        current_rsi = data['RSI'].iloc[-1] if not pd.isna(data['RSI'].iloc[-1]) else 50
        current_macd = data['MACD'].iloc[-1] if not pd.isna(data['MACD'].iloc[-1]) else 0
        current_macd_signal = data['MACD_signal'].iloc[-1] if not pd.isna(data['MACD_signal'].iloc[-1]) else 0
        current_bb_upper = data['BB_upper'].iloc[-1]
        current_bb_lower = data['BB_lower'].iloc[-1]
        current_sma20 = data['SMA_20'].iloc[-1]
        nearest_support = None
        nearest_resistance = None
        support_distance = float('inf')
        resistance_distance = float('inf')      
        for timestamp, level in support_levels:
            distance = abs(current_price - level)
            if distance < support_distance:
                support_distance = distance
                nearest_support = level      
        for timestamp, level in resistance_levels:
            distance = abs(current_price - level)
            if distance < resistance_distance:
                resistance_distance = distance
                nearest_resistance = level
        signal_strength = 0
        signal_type = "HOLD"
        reasons = []
        alerts = []
        timing_suggestions = []
        support_threshold = current_price * 0.002
        resistance_threshold = current_price * 0.002        
        if nearest_support and support_distance <= support_threshold:
            if current_price > nearest_support:
                signal_strength += 40
                signal_type = "BUY"
                reasons.append(f"🟢 BOUNCING OFF SUPPORT: {nearest_support:.4f}")
                alerts.append(f"🚨 SUPPORT BOUNCE DETECTED at {nearest_support:.4f}")
                timing_suggestions.append("⌚ Enter long position on next 1-2 minute candle close above support")     
        if nearest_resistance and resistance_distance <= resistance_threshold:
            if current_price < nearest_resistance:
                signal_strength += 40
                signal_type = "SELL"
                reasons.append(f"🔴 REJECTED AT RESISTANCE: {nearest_resistance:.4f}")
                alerts.append(f"🚨 RESISTANCE REJECTION at {nearest_resistance:.4f}")
                timing_suggestions.append("⌚ Enter short position on next 1-2 minute candle close below resistance")
        if nearest_support and current_price < nearest_support and volume_spike:
            signal_strength += 50
            signal_type = "SELL"
            reasons.append(f"🔻 SUPPORT BREAKOUT: {nearest_support:.4f}")
            alerts.append(f"🚨 MAJOR ALERT: SUPPORT BROKEN at {nearest_support:.4f}")
            timing_suggestions.append("⌚ Immediate entry - breakdown likely to accelerate")     
        if nearest_resistance and current_price > nearest_resistance and volume_spike:
            signal_strength += 50
            signal_type = "BUY"
            reasons.append(f"🔺 RESISTANCE BREAKOUT: {nearest_resistance:.4f}")
            alerts.append(f"🚨 MAJOR ALERT: RESISTANCE BROKEN at {nearest_resistance:.4f}")
            timing_suggestions.append("⌚ Immediate entry - breakout likely to continue")
        if current_rsi < 25: 
            signal_strength += 30
            if signal_type != "SELL":
                signal_type = "BUY"
            reasons.append(f"📈 EXTREMELY OVERSOLD RSI: {current_rsi:.1f}")
            alerts.append(f"🚨 RSI EXTREME OVERSOLD: {current_rsi:.1f}")
            timing_suggestions.append("⌚ Enter long position on next bullish reversal candle")
        elif current_rsi > 75:
            signal_strength += 30
            if signal_type != "BUY":
                signal_type = "SELL"
            reasons.append(f"📉 EXTREMELY OVERBOUGHT RSI: {current_rsi:.1f}")
            alerts.append(f"🚨 RSI EXTREME OVERBOUGHT: {current_rsi:.1f}")
            timing_suggestions.append("⌚ Enter short position on next bearish reversal candle")
        previous_macd = data['MACD'].iloc[-2] if len(data) > 1 and not pd.isna(data['MACD'].iloc[-2]) else current_macd
        previous_macd_signal = data['MACD_signal'].iloc[-2] if len(data) > 1 and not pd.isna(data['MACD_signal'].iloc[-2]) else current_macd_signal
        if previous_macd < previous_macd_signal and current_macd > current_macd_signal:
            signal_strength += 25
            if signal_type != "SELL":
                signal_type = "BUY"
            reasons.append("📈 MACD BULLISH CROSSOVER")
            alerts.append("🚨 MACD BULLISH CROSSOVER DETECTED")
            timing_suggestions.append("⌚ Enter long position within next 5 minutes")
        if previous_macd > previous_macd_signal and current_macd < current_macd_signal:
            signal_strength += 25
            if signal_type != "BUY":
                signal_type = "SELL"
            reasons.append("📉 MACD BEARISH CROSSOVER")
            alerts.append("🚨 MACD BEARISH CROSSOVER DETECTED")
            timing_suggestions.append("⌚ Enter short position within next 5 minutes")
        if current_price < current_bb_lower:
            signal_strength += 20
            if signal_type != "SELL":
                signal_type = "BUY"
            reasons.append("📈 PRICE BELOW BOLLINGER LOWER BAND")
            alerts.append("🚨 PRICE BELOW BOLLINGER LOWER BAND")
            timing_suggestions.append("⌚ Enter long position on bounce confirmation")     
        if current_price > current_bb_upper:
            signal_strength += 20
            if signal_type != "BUY":
                signal_type = "SELL"
            reasons.append("📉 PRICE ABOVE BOLLINGER UPPER BAND")
            alerts.append("🚨 PRICE ABOVE BOLLINGER UPPER BAND")
            timing_suggestions.append("⌚ Enter short position on rejection confirmation")
        if current_price > current_sma20 and data['Close'].iloc[-2] <= data['SMA_20'].iloc[-2]:
            signal_strength += 15
            reasons.append("📈 PRICE CROSSED ABOVE SMA20")
            alerts.append("🚨 PRICE CROSSED ABOVE SMA20")     
        if current_price < current_sma20 and data['Close'].iloc[-2] >= data['SMA_20'].iloc[-2]:
            signal_strength += 15
            reasons.append("📉 PRICE CROSSED BELOW SMA20")
            alerts.append("🚨 PRICE CROSSED BELOW SMA20")
        if volume_spike:
            signal_strength += 10
            reasons.append(f"📊 HIGH VOLUME CONFIRMATION ({current_volume/avg_volume:.1f}x avg)")
        if abs(price_change) > 0.5:
            signal_strength += 15
            if price_change > 0:
                reasons.append(f"🚀 STRONG UPWARD MOVE: +{price_change:.2f}%")
            else:
                reasons.append(f"🔻 STRONG DOWNWARD MOVE: {price_change:.2f}%")
        return {
            "signal": signal_type,
            "strength": min(signal_strength, 100),
            "current_price": current_price,
            "price_change": price_change,
            "nearest_support": nearest_support,
            "nearest_resistance": nearest_resistance,
            "rsi": current_rsi,
            "macd": current_macd,
            "volume_spike": volume_spike,
            "reasons": reasons,
            "alerts": alerts,
            "timing_suggestions": timing_suggestions,
            "technical_data": {
                "bb_upper": current_bb_upper,
                "bb_lower": current_bb_lower,
                "sma20": current_sma20,
                "current_volume": current_volume,
                "avg_volume": avg_volume
            }
        }   
    
    def get_market_news(self) -> List[Dict]:
        try:
            url = "https://api.marketaux.com/v1/news/all?countries=us&filter_entities=true&limit=10&published_after=2024-01-01&api_token=demo"
            response = requests.get(url, timeout=10)           
            if response.status_code == 200:
                news_data = response.json()
                return news_data.get('data', [])[:5]
            else:
                return [{"title": "News API unavailable", "description": "Using demo mode", "url": "#"}]
        except:
            return [{"title": "News service temporarily unavailable", "description": "Please check back later", "url": "#"}]  
    
    def analyze_market_sentiment(self, news: List[Dict], symbol: str) -> str:
        try:
            news_text = "\n".join([f"- {item.get('title', '')}: {item.get('description', '')}" for item in news[:3]])           
            prompt = f"""
            Analyze the market sentiment for {symbol} based on this recent news:
            
            {news_text}
            
            Provide a brief analysis (2-3 sentences) focusing on:
            1. Overall market sentiment (Bullish/Bearish/Neutral)
            2. Key factors affecting the market
            3. Short-term outlook for trading
            
            Keep the response concise and trading-focused.
            """
            
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Sentiment analysis unavailable: {str(e)}"   
    
    def create_chart(self, data: pd.DataFrame, support_levels: List, resistance_levels: List, symbol: str, interval: str = "2m", signals: Dict = None):
        if data.empty:
            return go.Figure()      
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            subplot_titles=(f'{symbol} Price Chart ({interval})', 'RSI', 'Volume'),
            row_heights=[0.6, 0.2, 0.2]
        )
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Add buy/sell markers if signals exist
        if signals and signals['signal'] in ['BUY', 'SELL']:
            marker_color = 'green' if signals['signal'] == 'BUY' else 'red'
            marker_symbol = 'triangle-up' if signals['signal'] == 'BUY' else 'triangle-down'
            fig.add_trace(
                go.Scatter(
                    x=[data.index[-1]],
                    y=[data['Close'].iloc[-1]],
                    mode='markers',
                    marker=dict(
                        color=marker_color,
                        size=12,
                        symbol=marker_symbol
                    ),
                    name=f"{signals['signal']} Signal"
                ),
                row=1, col=1
            )
        
        if 'BB_upper' in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data['BB_upper'], name='BB Upper', 
                          line=dict(color='rgba(250,250,250,0.5)', width=1)),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=data.index, y=data['BB_lower'], name='BB Lower', 
                          line=dict(color='rgba(250,250,250,0.5)', width=1), 
                          fill='tonexty', fillcolor='rgba(250,250,250,0.1)'),
                row=1, col=1
            )
        if 'SMA_20' in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data['SMA_20'], name='SMA20', 
                          line=dict(color='orange', width=2)),
                row=1, col=1
            )
        for timestamp, level in support_levels[-5:]:
            fig.add_hline(
                y=level,
                line_dash="dash",
                line_color="green",
                annotation_text=f"Support: {level:.4f}",
                row=1, col=1
            )      
        for timestamp, level in resistance_levels[-5:]:
            fig.add_hline(
                y=level,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Resistance: {level:.4f}",
                row=1, col=1
            )
        if 'RSI' in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='purple')),
                row=2, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                name='Volume',
                marker_color='rgba(158,202,225,0.8)'
            ),
            row=3, col=1
        )      
        fig.update_layout(
            title=f'{symbol} Advanced Trading Analysis - {interval} Chart',
            height=800,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )     
        return fig 
    
    def get_quotex_otc_data(self, pair: str) -> pd.DataFrame:
        try:
            otc_pairs_mapping = {
                "EUR/USD OTC": "EURUSD=X",
                "GBP/USD OTC": "GBPUSD=X", 
                "USD/JPY OTC": "USDJPY=X",
                "AUD/USD OTC": "AUDUSD=X",
                "USD/CHF OTC": "USDCHF=X",
                "USD/CAD OTC": "USDCAD=X",
                "NZD/USD OTC": "NZDUSD=X",
                "EUR/GBP OTC": "EURGBP=X",
                "EUR/JPY OTC": "EURJPY=X",
                "GBP/JPY OTC": "GBPJPY=X"
            }           
            yahoo_symbol = otc_pairs_mapping.get(pair, "EURUSD=X")
            ticker = yf.Ticker(yahoo_symbol)
            data = ticker.history(period="1d", interval="1m")
            if not data.empty:
                variation = np.random.normal(0, 0.0001, len(data))
                data['Close'] = data['Close'] * (1 + variation)
                data['Open'] = data['Open'] * (1 + variation)
                data['High'] = data['High'] * (1 + variation)
                data['Low'] = data['Low'] * (1 + variation)          
            return data          
        except Exception as e:
            st.error(f"Error fetching OTC data for {pair}: {str(e)}")
            return pd.DataFrame()

def display_signal_box(signal_data, title="Trading Signal"):
    signal_color = {
        "BUY": "#00FF00",
        "SELL": "#FF0000", 
        "HOLD": "#FFA500",
        "NO DATA": "#808080"
    }                    
    signal_emoji = {
        "BUY": "🟢📈",
        "SELL": "🔴📉",
        "HOLD": "🟡⏸️",
        "NO DATA": "⚪❓"
    }
    
    # Add timing information to the box
    timing_info = ""
    if signal_data.get('timing_suggestions'):
        timing_info = f"<p><strong>⏱️ Timing:</strong> {signal_data['timing_suggestions'][0]}</p>"
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {signal_color.get(signal_data['signal'], 'gray')}, {signal_color.get(signal_data['signal'], 'gray')}aa); 
               color: white; padding: 25px; border-radius: 15px; text-align: center; margin: 20px 0; box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
        <h1>{signal_emoji.get(signal_data['signal'], '')} {title}: {signal_data['signal']}</h1>
        <h2>Strength: {signal_data['strength']}/100</h2>
        <h3>Current Price: {signal_data.get('current_price', 'N/A'):.4f}</h3>
        <h4>Price Change: {signal_data.get('price_change', 0):.3f}%</h4>
        {timing_info}
    </div>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="Enhanced Quotex Trading Signal Bot",
        page_icon="📈",
        layout="wide"
    )    
    st.title("📈 Enhanced Quotex Trading Signal Bot")
    st.markdown("*Advanced Support & Resistance Strategy with Real-time Alerts*")
    tab1, tab2 = st.tabs(["🔥 Live Trading Signals", "📊 Quotex OTC Charts"]) 
    bot = TradingSignalBot()  
    with tab1:
        st.sidebar.header("Trading Parameters")        
        symbols = [
            "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "AUDUSD=X",
            "USDCAD=X", "NZDUSD=X", "EURJPY=X", "GBPJPY=X", "EURGBP=X"
        ]      
        selected_symbol = st.sidebar.selectbox("Select Currency Pair", symbols)      
        timeframe_options = {
            "1 Minute (1 Hour Data)": ("1d", "1m"),
            "2 Minute (2 Hour Data)": ("1d", "2m"),
            "5 Minute (1 Day Data)": ("1d", "5m"),
            "15 Minute (5 Days Data)": ("5d", "15m"),
            "30 Minute (1 Month Data)": ("1mo", "30m"),
            "1 Hour (3 Months Data)": ("3mo", "1h")
        }    
        selected_timeframe = st.sidebar.selectbox("Select Chart Timeframe", list(timeframe_options.keys()), index=1)
        period, interval = timeframe_options[selected_timeframe]     
        auto_refresh = st.sidebar.checkbox("Auto Refresh (15s)", value=False)
        alert_threshold = st.sidebar.slider("Signal Strength Alert Threshold", 0, 100, 70)     
        if st.sidebar.button("🔄 Refresh Signals") or auto_refresh:
            col1, col2 = st.columns([2, 1])           
            with col1:
                st.subheader(f"📊 {selected_symbol} Live Analysis") 
                
                # Show optimal trading times
                next_window, time_left = bot.get_next_trading_window(selected_symbol)
                st.info(f"⏱️ **Optimal Trading Times:** {', '.join(bot.optimal_trading_times.get(selected_symbol, ['N/A']))}")
                
                if time_left.total_seconds() > 0:
                    # Format time display
                    hours, remainder = divmod(time_left.seconds, 3600)
                    minutes, seconds = divmod(remainder, 60)
                    time_display = f"{hours}h {minutes}m {seconds}s"
                    st.warning(f"⏳ Next trading window: {next_window} ({time_display} remaining)")
                elif next_window.startswith("NOW!"):
                    st.success(f"✅ Currently in optimal trading window: {next_window}")
                else:
                    st.info(f"⏱️ {next_window}")
                
                with st.spinner("Fetching live market data..."):
                    data = bot.get_market_data(selected_symbol, period=period, interval=interval)              
                if not data.empty:
                    support_levels, resistance_levels = bot.calculate_support_resistance(data)
                    signal_data = bot.check_breakout_signals(data, support_levels, resistance_levels)
                    
                    # Display signal box with timing suggestions
                    display_signal_box(signal_data, title=f"{selected_symbol} Signal")
                    
                    if signal_data.get('alerts'):
                        st.subheader("🚨 LIVE ALERTS")
                        for alert in signal_data['alerts']:
                            st.error(alert)
                    if signal_data['strength'] >= alert_threshold:
                        st.balloons()
                        st.success(f"🎯 HIGH CONFIDENCE SIGNAL DETECTED! Strength: {signal_data['strength']}/100")
                    
                    # Show detailed timing suggestions
                    if signal_data.get('timing_suggestions'):
                        st.subheader("⏱️ Entry Timing Recommendations")
                        for suggestion in signal_data['timing_suggestions']:
                            st.info(suggestion)
                    
                    chart = bot.create_chart(data, support_levels, resistance_levels, selected_symbol, interval, signal_data)
                    st.plotly_chart(chart, use_container_width=True)
                    
                    col_support, col_resistance = st.columns(2)                    
                    with col_support:
                        st.subheader("🟢 Support Levels")
                        if signal_data.get('nearest_support'):
                            distance = abs(signal_data['current_price'] - signal_data['nearest_support'])
                            distance_pct = (distance / signal_data['current_price']) * 100
                            st.metric("Nearest Support", f"{signal_data['nearest_support']:.4f}", 
                                     f"{distance_pct:.3f}% away")                      
                        recent_supports = support_levels[-3:] if len(support_levels) >= 3 else support_levels
                        for i, (timestamp, level) in enumerate(recent_supports):
                            st.write(f"{i+1}. {level:.4f}")                  
                    with col_resistance:
                        st.subheader("🔴 Resistance Levels")
                        if signal_data.get('nearest_resistance'):
                            distance = abs(signal_data['current_price'] - signal_data['nearest_resistance'])
                            distance_pct = (distance / signal_data['current_price']) * 100
                            st.metric("Nearest Resistance", f"{signal_data['nearest_resistance']:.4f}", 
                                     f"{distance_pct:.3f}% away")                      
                        recent_resistances = resistance_levels[-3:] if len(resistance_levels) >= 3 else resistance_levels
                        for i, (timestamp, level) in enumerate(recent_resistances):
                            st.write(f"{i+1}. {level:.4f}")              
                else:
                    st.error("❌ Unable to fetch market data. Please try again.")           
            with col2:
                st.subheader("📰 Market News & Sentiment")
                with st.spinner("Loading market news..."):
                    news = bot.get_market_news()               
                for item in news[:3]:
                    with st.expander(f"📰 {item.get('title', 'News')[:50]}..."):
                        st.write(item.get('description', 'No description available'))
                        if item.get('url') and item['url'] != '#':
                            st.markdown(f"[Read more]({item['url']})")              
                st.subheader("🤖 AI Market Sentiment")
                with st.spinner("Analyzing market sentiment..."):
                    sentiment = bot.analyze_market_sentiment(news, selected_symbol)
                st.write(sentiment)           
                if not data.empty and 'technical_data' in signal_data:
                    st.subheader("📊 Technical Indicators")
                    rsi_color = "🟢" if signal_data['rsi'] < 30 else "🔴" if signal_data['rsi'] > 70 else "🟡"
                    st.metric("RSI", f"{rsi_color} {signal_data['rsi']:.1f}")
                    macd_color = "🟢" if signal_data['macd'] > 0 else "🔴"
                    st.metric("MACD", f"{macd_color} {signal_data['macd']:.6f}")
                    if signal_data.get('volume_spike'):
                        st.metric("Volume", "🔥 HIGH VOLUME", "Volume Spike Detected")
                    else:
                        st.metric("Volume", "📊 Normal", "Regular Volume")
                    tech_data = signal_data['technical_data']
                    if signal_data['current_price'] > tech_data['bb_upper']:
                        st.metric("BB Position", "🔴 Above Upper Band", "Potentially Overbought")
                    elif signal_data['current_price'] < tech_data['bb_lower']:
                        st.metric("BB Position", "🟢 Below Lower Band", "Potentially Oversold")
                    else:
                        st.metric("BB Position", "🟡 Normal Range", "Within Bands")       
        if auto_refresh:
            time.sleep(15)
            st.rerun()   
    with tab2:
        st.subheader("📊 Quotex OTC Trading Charts")
        st.info("🔔 Note: This simulates OTC data. In production, connect to actual Quotex OTC feed.")       
        col1, col2 = st.columns([1, 3])       
        with col1:
            st.subheader("OTC Pairs")
            otc_pairs = [
                "EUR/USD OTC",
                "GBP/USD OTC", 
                "USD/JPY OTC",
                "AUD/USD OTC",
                "USD/CHF OTC",
                "USD/CAD OTC",
                "NZD/USD OTC",
                "EUR/GBP OTC",
                "EUR/JPY OTC",
                "GBP/JPY OTC"
            ]           
            selected_otc_pair = st.selectbox("Select OTC Pair", otc_pairs)            
            otc_timeframes = {
                "1 Minute": ("1d", "1m"),
                "2 Minutes": ("1d", "2m"),
                "5 Minutes": ("1d", "5m")
            }           
            selected_otc_timeframe = st.selectbox("Select OTC Timeframe", list(otc_timeframes.keys()))
            otc_period, otc_interval = otc_timeframes[selected_otc_timeframe]           
            if st.button("🔄 Load OTC Chart"):
                with st.spinner("Loading OTC data..."):
                    otc_data = bot.get_quotex_otc_data(selected_otc_pair)                   
                    if not otc_data.empty:
                        st.session_state.otc_data = otc_data
                        st.session_state.otc_pair = selected_otc_pair
                        st.session_state.otc_interval = otc_interval
                        st.success(f"✅ Loaded {selected_otc_pair} data successfully!")
                    else:
                        st.error("❌ Failed to load OTC data")
            otc_auto_refresh = st.checkbox("Auto Refresh OTC (10s)", value=False)           
            if otc_auto_refresh and 'otc_data' in st.session_state:
                time.sleep(10)
                st.rerun()      
        with col2:
            st.subheader(f"📈 {selected_otc_pair} Live Chart")         
            if 'otc_data' in st.session_state and not st.session_state.otc_data.empty:
                otc_data = st.session_state.otc_data
                otc_support_levels, otc_resistance_levels = bot.calculate_support_resistance(otc_data)
                otc_signals = bot.check_breakout_signals(otc_data, otc_support_levels, otc_resistance_levels)
                
                # Display OTC signal box
                display_signal_box(otc_signals, title=f"{selected_otc_pair} OTC Signal")
                
                if otc_signals.get('timing_suggestions'):
                    st.subheader("⏱️ OTC Entry Timing")
                    for suggestion in otc_signals['timing_suggestions']:
                        st.info(suggestion)
                
                otc_fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.03,
                    subplot_titles=(f'{selected_otc_pair} - {otc_interval}', 'Volume'),
                    row_heights=[0.8, 0.2]
                )
                otc_fig.add_trace(
                    go.Candlestick(
                        x=otc_data.index,
                        open=otc_data['Open'],
                        high=otc_data['High'],
                        low=otc_data['Low'],
                        close=otc_data['Close'],
                        name='OTC Price'
                    ),
                    row=1, col=1
                )
                
                # Add buy/sell marker for OTC
                if otc_signals['signal'] in ['BUY', 'SELL']:
                    marker_color = 'green' if otc_signals['signal'] == 'BUY' else 'red'
                    marker_symbol = 'triangle-up' if otc_signals['signal'] == 'BUY' else 'triangle-down'
                    otc_fig.add_trace(
                        go.Scatter(
                            x=[otc_data.index[-1]],
                            y=[otc_data['Close'].iloc[-1]],
                            mode='markers',
                            marker=dict(
                                color=marker_color,
                                size=12,
                                symbol=marker_symbol
                            ),
                            name=f"{otc_signals['signal']} Signal"
                        ),
                        row=1, col=1
                    )
                
                for timestamp, level in otc_support_levels[-3:]:
                    otc_fig.add_hline(
                        y=level,
                        line_dash="dash",
                        line_color="green",
                        annotation_text=f"S: {level:.4f}",
                        row=1, col=1
                    )              
                for timestamp, level in otc_resistance_levels[-3:]:
                    otc_fig.add_hline(
                        y=level,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"R: {level:.4f}",
                        row=1, col=1
                    )
                otc_fig.add_trace(
                    go.Bar(
                        x=otc_data.index,
                        y=otc_data['Volume'],
                        name='Volume',
                        marker_color='rgba(158,202,225,0.8)'
                    ),
                    row=2, col=1
                )               
                otc_fig.update_layout(
                    title=f'{selected_otc_pair} OTC Trading Chart',
                    height=600,
                    showlegend=True,
                    xaxis_rangeslider_visible=False
                )               
                st.plotly_chart(otc_fig, use_container_width=True)
                col_otc1, col_otc2 = st.columns(2)            
                with col_otc1:
                    st.subheader("🎯 OTC Signal Analysis")
                    if otc_signals.get('reasons'):
                        for reason in otc_signals['reasons'][:3]:
                            st.write(f"• {reason}")
                    if otc_signals.get('nearest_support'):
                        st.metric("Nearest Support", f"{otc_signals['nearest_support']:.4f}")
                    if otc_signals.get('nearest_resistance'):
                        st.metric("Nearest Resistance", f"{otc_signals['nearest_resistance']:.4f}")              
                with col_otc2:
                    st.subheader("📊 OTC Indicators")
                    st.metric("RSI", f"{otc_signals.get('rsi', 50):.1f}")
                    st.metric("MACD", f"{otc_signals.get('macd', 0):.6f}")
                    price_change = otc_signals.get('price_change', 0)
                    change_color = "🟢" if price_change > 0 else "🔴" if price_change < 0 else "🟡"
                    st.metric("Price Change", f"{change_color} {price_change:.3f}%")            
            else:
                st.info("👆 Select an OTC pair and click 'Load OTC Chart' to view live data")
                st.subheader("📋 Available OTC Pairs")
                otc_info = pd.DataFrame({
                    'Pair': ['EUR/USD OTC', 'GBP/USD OTC', 'USD/JPY OTC', 'AUD/USD OTC'],
                    'Status': ['🟢 Active', '🟢 Active', '🟢 Active', '🟢 Active'],
                    'Spread': ['0.8 pips', '1.2 pips', '0.9 pips', '1.1 pips'],
                    'Market': ['24/7 OTC', '24/7 OTC', '24/7 OTC', '24/7 OTC']
                })
                st.table(otc_info)

if __name__ == "__main__":
    main()