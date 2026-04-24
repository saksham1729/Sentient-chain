import yfinance as yf
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

def fetch_intraday_metrics(ticker_symbol: str):
    """Fetches intraday data and calculates volatility for the target asset."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        # Fetch today's data at 5-minute intervals
        hist = ticker.history(period="1d", interval="5m")
        if hist.empty:
            # FIX: Added the 'f' prefix to properly format the string
            return f"Intraday data unavailable for {ticker_symbol}. Market may be closed."
        
        current_price = hist['Close'].iloc[-1]
        daily_high = hist['High'].max()
        daily_low = hist['Low'].min()
        volatility_spread = daily_high - daily_low
        
        return (f"Asset: {ticker_symbol} | "
                f"Current Price: ₹{current_price:.2f} | "
                f"Intraday High: ₹{daily_high:.2f} | "
                f"Intraday Low: ₹{daily_low:.2f} | "
                f"Volatility Spread: ₹{volatility_spread:.2f}")
    except Exception as e:
        return f"Error fetching market data for {ticker_symbol}: {e}"

def analyze_quantitative_impact(risk_assessment: dict, ticker: str):
    # 1. Fetch live market context
    market_data = fetch_intraday_metrics(ticker)

    # 2. Initialize LLM (Slightly higher temperature for analysis generation)
    llm = OllamaLLM(model="llama3.2", temperature=0.2)
    
    # 3. Create the prompt
    # FIX: Removed "silver assets" and properly placed the {ticker} variable in the text
    prompt = PromptTemplate(
        template="""You are a Quantitative Analyst. 
        Assess the financial impact of the risk on {ticker} using live INTRADAY market data.
        
        Risk Assessment: {assessment}
        Live Intraday Market Data: {market_data}
        
        Analyze the volatility spread and current price. Provide a highly concise, 2-sentence quantitative analysis of the immediate market reaction and potential cost impact.
        """,
        input_variables=["assessment", "market_data", "ticker"],
    )
    
    chain = prompt | llm
    
    # 4. Execute the chain
    result = chain.invoke({"assessment": str(risk_assessment), "market_data": market_data, "ticker": ticker})
    return result