import streamlit as st
import requests
import pandas as pd
import json
from datetime import datetime
import anthropic
import openai
from typing import Optional, Dict, Any, Union
import requests
requests.packages.urllib3.disable_warnings()
requests.packages.urllib3.util.ssl_.DEFAULT_CERTS = False
from openai import OpenAI
# Initialize session state variables
if 'api_key_set' not in st.session_state:
    st.session_state.api_key_set = False
if 'selected_stocks' not in st.session_state:
    st.session_state.selected_stocks = []
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'selected_category' not in st.session_state:
    st.session_state.selected_category = None
if 'ai_provider' not in st.session_state:
    st.session_state.ai_provider = None
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = None
if 'claude_api_key' not in st.session_state:
    st.session_state.claude_api_key = None



# Load stock data from CSV
@st.cache_data
def load_stock_data():
    df = pd.read_csv('stock_data.csv')
    return df



# Function to get unique categories from stock data
@st.cache_data
def get_stock_categories(df):
    # Assuming your CSV has a 'Category' column
    return sorted(df['Industry'].unique())


# Function to get stocks within a category
def get_stocks_in_category(df, category):
    category_stocks = df[df['Industry'] == category]
    return category_stocks

 #Fecthing company overview


def fetch_company_overview(symbol, api_key):
    url = f'https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={api_key}'
    r = requests.get(url,verify= False)
    data = r.json()
    
    if not data or "Symbol" not in data:
        st.error(f"Error fetching company overview for {symbol}")
        return None
        
    return data

# Function to display company metrics
def display_company_metrics(overview_data):
    if overview_data is None:
        return
    
    # Create three columns for metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        market_cap = overview_data.get("MarketCapitalization", "N/A")

# Check if the value is numeric
        if market_cap.isdigit():
            market_cap_in_billion = int(market_cap) / 1_000_000_000
            #print(f"Market Capitalization: {market_cap_in_billion:.2f} Billions")
        else:
            print("Market Capitalization: N/A")

        # st.metric("Market Cap", overview_data.get("MarketCapitalization", "N/A"))
        st.markdown(
                f"""
                <style>
                .metric-value {{
                    font-size: 20px !important; /* Adjust size */
                }}
                </style>
                <div class="metric-value">Market Cap: {market_cap_in_billion:.2f} B</div>
                """,
                unsafe_allow_html=True
            )
        st.metric("P/E Ratio", overview_data.get("PERatio", "N/A"))
        st.metric("Dividend Yield", overview_data.get("DividendYield", "N/A"))
    
    with col2:
        st.metric("52W High", overview_data.get("52WeekHigh", "N/A"))
        st.metric("52W Low", overview_data.get("52WeekLow", "N/A"))
        st.metric("Beta", overview_data.get("Beta", "N/A"))
    
    with col3:
        st.metric("Profit Margin", overview_data.get("ProfitMargin", "N/A"))
        st.metric("Operating Margin", overview_data.get("OperatingMarginTTM", "N/A"))
        st.metric("ROE", overview_data.get("ReturnOnEquityTTM", "N/A"))



# Function to fetch stock data from Alpha Vantage
def fetch_stock_data(symbol, api_key):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}'
    r = requests.get(url, verify= False)
    data = r.json()
    
    if "Time Series (Daily)" not in data:
        st.error(f"Error fetching data for {symbol}. Please check the API key and symbol.")
        return None

    # Process the data
    time_series = data["Time Series (Daily)"]
    df = pd.DataFrame.from_dict(time_series, orient="index")
    df.index = pd.to_datetime(df.index)
    df = df.rename(columns={
        "1. open": "Open",
        "2. high": "High",
        "3. low": "Low",
        "4. close": "Close",
        "5. volume": "Volume"
    })
    
    # Convert to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

# Function to fetch news sentiment
def fetch_news_sentiment(symbol, api_key):
    url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={api_key}'
    r = requests.get(url , verify= False)
    data = r.json()
    
    if 'feed' not in data:
        st.error(f"Error fetching news sentiment for {symbol}")
        return None
        
    # Process news sentiment data
    news_df = pd.DataFrame(data['feed'])
    if not news_df.empty:
        news_df['time_published'] = pd.to_datetime(news_df['time_published'])
        news_df = news_df[['time_published', 'title', 'summary', 'overall_sentiment_score', 'overall_sentiment_label']]
    
    return news_df




def custom_json_converter(obj):
    if isinstance(obj, (datetime, pd.Timestamp)):  # Convert datetime to string
        return obj.isoformat()
    elif isinstance(obj, set):  # Convert set to list
        return list(obj)
    elif isinstance(obj, pd.DataFrame):  # Convert DataFrame to list of dictionaries
        return obj.to_dict('records')
    elif isinstance(obj, pd.Series):  # Convert Series to list
        return obj.tolist()
    else:
        raise TypeError(f"Type {type(obj)} not serializable")



def display_income_statement_analysis(annual_reports, quarterly_reports):
    if annual_reports is None or quarterly_reports is None:
        return
    
    # Display tabs for annual and quarterly data
    tab1, tab2 = st.tabs(["Annual Reports", "Quarterly Reports"])
    
    with tab1:
        st.subheader("Annual Financial Performance")
        
        # Latest annual metrics
        latest_annual = annual_reports.iloc[0]
        cols = st.columns(3)
        
        with cols[0]:
            st.metric(
                "Revenue (USD)",
                f"${int(latest_annual['totalRevenue']):,}",
                delta=f"{((float(latest_annual['totalRevenue']) - float(annual_reports.iloc[1]['totalRevenue'])) / float(annual_reports.iloc[1]['totalRevenue']) * 100):.1f}%"
            )
            
        with cols[1]:
            st.metric(
                "Net Income (USD)",
                f"${int(latest_annual['netIncome']):,}",
                delta=f"{((float(latest_annual['netIncome']) - float(annual_reports.iloc[1]['netIncome'])) / float(annual_reports.iloc[1]['netIncome']) * 100):.1f}%"
            )
            
        with cols[2]:
            gross_margin = (float(latest_annual['grossProfit']) / float(latest_annual['totalRevenue']) * 100)
            prev_gross_margin = (float(annual_reports.iloc[1]['grossProfit']) / float(annual_reports.iloc[1]['totalRevenue']) * 100)
            st.metric(
                "Gross Margin",
                f"{gross_margin:.1f}%",
                delta=f"{(gross_margin - prev_gross_margin):.1f}%"
            )
        
        # Annual trends chart
        st.subheader("Annual Trends")
        annual_chart_data = annual_reports[['fiscalDateEnding', 'totalRevenue', 'grossProfit', 'netIncome']].copy()
        annual_chart_data.set_index('fiscalDateEnding', inplace=True)
        st.line_chart(annual_chart_data)
        
        # Detailed annual data
        st.subheader("Detailed Annual Reports")
        st.dataframe(annual_reports)
    
    with tab2:
        st.subheader("Quarterly Financial Performance")
        
        # Latest quarterly metrics
        latest_quarter = quarterly_reports.iloc[0]
        cols = st.columns(3)
        
        with cols[0]:
            st.metric(
                "Revenue (USD)",
                f"${int(latest_quarter['totalRevenue']):,}",
                delta=f"{((float(latest_quarter['totalRevenue']) - float(quarterly_reports.iloc[1]['totalRevenue'])) / float(quarterly_reports.iloc[1]['totalRevenue']) * 100):.1f}%"
            )
            
        with cols[1]:
            st.metric(
                "Net Income (USD)",
                f"${int(latest_quarter['netIncome']):,}",
                delta=f"{((float(latest_quarter['netIncome']) - float(quarterly_reports.iloc[1]['netIncome'])) / float(quarterly_reports.iloc[1]['netIncome']) * 100):.1f}%"
            )
            
        with cols[2]:
            gross_margin = (float(latest_quarter['grossProfit']) / float(latest_quarter['totalRevenue']) * 100)
            prev_gross_margin = (float(quarterly_reports.iloc[1]['grossProfit']) / float(quarterly_reports.iloc[1]['totalRevenue']) * 100)
            st.metric(
                "Gross Margin",
                f"{gross_margin:.1f}%",
                delta=f"{(gross_margin - prev_gross_margin):.1f}%"
            )
        
        # Quarterly trends chart
        st.subheader("Quarterly Trends")
        quarterly_chart_data = quarterly_reports[['fiscalDateEnding', 'totalRevenue', 'grossProfit', 'netIncome']].copy()
        quarterly_chart_data.set_index('fiscalDateEnding', inplace=True)
        st.line_chart(quarterly_chart_data)
        
        # Detailed quarterly data
        st.subheader("Detailed Quarterly Reports")
        st.dataframe(quarterly_reports)




def get_system_prompt():
    base_prompt = """
    You are an AI Financial Analyst. Given company financials, you are asked to summarize the finances, 
    give pros and cons, and make a recommendation. You will explain the complex finances so that a 
    beginner without any financial knowledge can understand. You will always warn the user that they 
    need to do their own research, and that you are a guide to get started.
    """
    
    return base_prompt


def fetch_income_statement(symbol, api_key):
    url = f'https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={symbol}&apikey={api_key}'
    r = requests.get(url, verify= False)
    data = r.json()
    
    if "annualReports" not in data:
        st.error(f"Error fetching income statement for {symbol}")
        return None, None
    
    annual_reports = pd.DataFrame(data["annualReports"])
    quarterly_reports = pd.DataFrame(data["quarterlyReports"])
    
    numeric_columns = [
        'totalRevenue', 'grossProfit', 'operatingIncome', 'netIncome',
        'ebitda', 'operatingExpenses', 'costOfRevenue'
    ]
    
    for col in numeric_columns:
        if col in annual_reports.columns:
            annual_reports[col] = pd.to_numeric(annual_reports[col], errors='coerce')
        if col in quarterly_reports.columns:
            quarterly_reports[col] = pd.to_numeric(quarterly_reports[col], errors='coerce')
    
    return annual_reports, quarterly_reports


def json_serializer(obj):
    """Custom JSON serializer to handle non-serializable objects"""
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat() 
    if isinstance(obj, set):
        return list(obj)
    if pd.isna(obj):
        return None
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, (int, float)) and (pd.isna(obj) or pd.isinf(obj)):
        return None
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def process_dataframe_to_dict(df: Optional[pd.DataFrame]) -> Optional[list]:
    """Safely convert DataFrame to dictionary records"""
    if isinstance(df, pd.DataFrame) and not df.empty:
        return df.where(pd.notna(df), None).to_dict('records')
    return None

def calculate_yoy_metrics(annual_reports: Optional[pd.DataFrame]) -> dict:
    """Calculate year-over-year metrics safely"""
    yoy_metrics = {}
    if isinstance(annual_reports, pd.DataFrame) and len(annual_reports) >= 2:
        latest_annual = annual_reports.iloc[0]
        prev_annual = annual_reports.iloc[1]
        
        try:
            if pd.notna(latest_annual['totalRevenue']) and pd.notna(prev_annual['totalRevenue']):
                yoy_metrics["revenue_growth"] = (
                    (float(latest_annual['totalRevenue']) - float(prev_annual['totalRevenue'])) 
                    / float(prev_annual['totalRevenue']) * 100
                )
            
            if pd.notna(latest_annual['netIncome']) and pd.notna(prev_annual['netIncome']):
                yoy_metrics["net_income_growth"] = (
                    (float(latest_annual['netIncome']) - float(prev_annual['netIncome'])) 
                    / float(prev_annual['netIncome']) * 100
                )
        except (ValueError, TypeError, ZeroDivisionError) as e:
            print(f"Warning: Error calculating YoY metrics: {e}")
    
    return yoy_metrics

def get_recommendation_from_ollama(
    stock_data: Optional[pd.DataFrame],
    sentiment_data: Optional[pd.DataFrame],
    overview_data: Dict[str, Any],
    annual_reports: Optional[pd.DataFrame],
    quarterly_reports: Optional[pd.DataFrame]
) -> str:
    """Get investment recommendation from Ollama with proper error handling and data processing"""
    
    url = "http://localhost:11434/api/generate"
    
    # Process each dataset safely
    stock_data_dict = process_dataframe_to_dict(stock_data)
    sentiment_data_dict = process_dataframe_to_dict(sentiment_data)
    annual_reports_dict = process_dataframe_to_dict(annual_reports)
    quarterly_reports_dict = process_dataframe_to_dict(quarterly_reports)
    
    # Calculate YoY metrics
    yoy_metrics = calculate_yoy_metrics(annual_reports)
    
    # Prepare analysis data with proper null handling
    analysis_data = {
        "company_info": {
            "overview": overview_data
        },
        "financial_performance": {
            "annual_reports": annual_reports_dict,
            "quarterly_reports": quarterly_reports_dict,
            "yoy_growth": yoy_metrics
        },
        "market_data": {
            "stock_prices": stock_data_dict,
            "sentiment": sentiment_data_dict
        },
        "analysis_timestamp": datetime.now().isoformat()
    }

    # Create prompt
    system_prompt = get_system_prompt()  # Assuming this function exists
    analysis_prompt = f"""
    Please analyze the following financial data and provide a comprehensive investment analysis:

    1. Company Overview:
    - Provide a brief summary of the company and its business model
    - Analyze the company's market position

    2. Financial Analysis:
    - Review the income statement trends and key metrics
    - Analyze year-over-year growth in revenue and profits
    - Identify any concerning or promising financial trends

    3. Market Sentiment:
    - Analyze recent news sentiment
    - Consider market trends

    4. Investment Recommendation:
    - Provide a clear investment recommendation (Buy/Hold/Sell)
    - List key risks and opportunities
    - Suggest factors that investors should monitor

    Here is the detailed financial data to analyze:
    {json.loads(json.dumps(analysis_data, default=json_serializer))}
    {print(json.loads(json.dumps(analysis_data, default=json_serializer)))}
    """
    
    try:
        # Prepare the payload with custom JSON serialization
        payload = {
            "model": "llama3.1",
            "system": system_prompt,
            "prompt": analysis_prompt,
            "content": json.loads(json.dumps(analysis_data, default=json_serializer)),
            "stream": False
        }

        # Make the request with proper error handling
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        response_data = response.json()
        return response_data.get('response', 'No response generated')
        
    except requests.exceptions.RequestException as e:
        print(f"Error making request to Ollama: {e}")
        return f"Error connecting to Ollama: {str(e)}"
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response: {e}")
        return f"Error processing Ollama response: {str(e)}"
    except Exception as e:
        print(f"Unexpected error: {e}")
        return f"Unexpected error: {str(e)}"










# def process_dataframe_to_dict(df: Optional[pd.DataFrame]) -> Optional[Dict[str, Any]]:
#     """Convert DataFrame to a dictionary if not None."""
#     return df.to_dict(orient="records") if df is not None else None

# def calculate_yoy_metrics(annual_reports: Optional[pd.DataFrame]) -> Dict[str, Any]:
#     """Calculate year-over-year growth metrics."""
#     if annual_reports is None or annual_reports.empty:
#         return {}
#     # Example: Calculate YoY revenue growth (assuming a 'Revenue' column exists)
#     annual_reports_sorted = annual_reports.sort_values(by="Year", ascending=True)
#     annual_reports_sorted["YoY_Revenue_Growth"] = annual_reports_sorted["Revenue"].pct_change()
#     return annual_reports_sorted[["Year", "YoY_Revenue_Growth"]].dropna().to_dict(orient="records")

# def get_system_prompt() -> str:
#     """Return system-level instructions for OpenAI."""
#     return "You are a financial analyst providing investment recommendations."

def get_openai_recommendation(
    stock_data: Optional[pd.DataFrame],
    sentiment_data: Optional[pd.DataFrame],
    overview_data: Dict[str, Any],
    annual_reports: Optional[pd.DataFrame],
    quarterly_reports: Optional[pd.DataFrame],
    api_key
) -> str:
    """Get investment recommendation from OpenAI with proper error handling and data processing."""
    # api_key = "your-api-key-here"
    client = OpenAI(api_key=api_key)
    
    stock_data_dict = process_dataframe_to_dict(stock_data)
    sentiment_data_dict = process_dataframe_to_dict(sentiment_data)
    annual_reports_dict = process_dataframe_to_dict(annual_reports)
    quarterly_reports_dict = process_dataframe_to_dict(quarterly_reports)
    
    yoy_metrics = calculate_yoy_metrics(annual_reports)
    
    analysis_data = {
        "company_info": {"overview": overview_data},
        "financial_performance": {
            "annual_reports": annual_reports_dict,
            "quarterly_reports": quarterly_reports_dict,
            "yoy_growth": yoy_metrics
        },
        "market_data": {
            "stock_prices": stock_data_dict,
            "sentiment": sentiment_data_dict
        }
        #,
        #"analysis_timestamp": datetime.now().isoformat()
    }
    
    system_prompt = get_system_prompt()
    
    # analysis_data_json = json.dumps(analysis_data, indent=4)
    
    analysis_prompt = f"""
        Please analyze the following financial data and provide a comprehensive investment analysis:
        
        1. Company Overview:
        - Provide a brief summary of the company and its business model
        - Analyze the company's market position

        2. Financial Analysis:
        - Review the income statement trends and key metrics
        - Analyze year-over-year growth in revenue and profits
        - Identify any concerning or promising financial trends

        3. Market Sentiment:
        - Analyze recent news sentiment
        - Consider market trends

        4. Investment Recommendation:
        - Provide a clear investment recommendation (Buy/Hold/Sell)
        - List key risks and opportunities
        - Suggest factors that investors should monitor
        
        Here is the detailed financial data to analyze:
        {json.loads(json.dumps(analysis_data, default=json_serializer))}
        {print(json.loads(json.dumps(analysis_data, default=json_serializer)))}
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": analysis_prompt}
            ]
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error while processing OpenAI response: {str(e)}"





def get_claude_recommendation(
    stock_data: Optional[pd.DataFrame],
    sentiment_data: Optional[pd.DataFrame],
    overview_data: Dict[str, Any],
    annual_reports: Optional[pd.DataFrame],
    quarterly_reports: Optional[pd.DataFrame]
) -> str:
    """Get investment recommendation from Claude with proper error handling and data processing"""
    
    url = "https://api.anthropic.com/v1/messages"
    
    stock_data_dict = process_dataframe_to_dict(stock_data)
    sentiment_data_dict = process_dataframe_to_dict(sentiment_data)
    annual_reports_dict = process_dataframe_to_dict(annual_reports)
    quarterly_reports_dict = process_dataframe_to_dict(quarterly_reports)
    
    yoy_metrics = calculate_yoy_metrics(annual_reports)
    
    analysis_data = {
        "company_info": {"overview": overview_data},
        "financial_performance": {
            "annual_reports": annual_reports_dict,
            "quarterly_reports": quarterly_reports_dict,
            "yoy_growth": yoy_metrics
        },
        "market_data": {
            "stock_prices": stock_data_dict,
            "sentiment": sentiment_data_dict
        },
        "analysis_timestamp": datetime.now().isoformat()
    }
    
    system_prompt = get_system_prompt()
    analysis_prompt = f"""
        Please analyze the following financial data and provide a comprehensive investment analysis:

        1. Company Overview:
        - Provide a brief summary of the company and its business model
        - Analyze the company's market position

        2. Financial Analysis:
        - Review the income statement trends and key metrics
        - Analyze year-over-year growth in revenue and profits
        - Identify any concerning or promising financial trends

        3. Market Sentiment:
        - Analyze recent news sentiment
        - Consider market trends

        4. Investment Recommendation:
        - Provide a clear investment recommendation (Buy/Hold/Sell)
        - List key risks and opportunities
        - Suggest factors that investors should monitor

        Here is the detailed financial data to analyze:
        {json.loads(json.dumps(analysis_data, default=json_serializer))}
        {print(json.loads(json.dumps(analysis_data, default=json_serializer)))}
        """
    try:
        payload = {
            "model": "claude-3-opus-20240229",
            "max_tokens": 1000,
            "messages": [  #{"role":"system","content":system_prompt}
                 {"role": "user", "content": {json.loads(json.dumps(analysis_data, default=json_serializer))},"prompt":system_prompt}
                # {"role":"assitant","content":analysis_data}
            ]
                }
        
        headers = {"Authorization": f"Bearer {st.session_state.claude_api_key}"}
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        response_data = response.json()
        
        return response_data["content"][0]["text"]
    except requests.exceptions.RequestException as e:
        return f"Error connecting to Claude: {str(e)}"
    except json.JSONDecodeError as e:
        return f"Error processing Claude response: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"




# Creating a layout with two columns
col1, col2 = st.columns([2, 1])  # Left (Data) | Right (Workflow)

with col1:
    st.subheader("Agentic AI For Stock Analysis")
    # Main UI Workflow
    st.title("🚀 Financial Analysis Workflow")

    # Step 1: API Key Configuration
    st.header("Step 1: Configure API Keys")
    with st.expander("API Key Setup", expanded=not st.session_state.api_key_set):
        # Alpha Vantage API Key
        alpha_vantage_key = st.text_input("Enter Alpha Vantage API Key:", type="password")
        
        # AI Provider Selection
        ai_provider = st.radio(
            "Select AI Provider:",
            ["OpenAI", "Claude", "Ollama"],
            key="ai_provider_selection"
        )
        
        # Additional API keys based on selection
        if ai_provider == "OpenAI":
            openai_key = st.text_input("Enter OpenAI API Key:", type="password")
            st.session_state.openai_api_key = openai_key
        elif ai_provider == "Claude":
            claude_key = st.text_input("Enter Claude API Key:", type="password")
            st.session_state.claude_api_key = claude_key
        
        if st.button("Save API Keys"):
            if alpha_vantage_key:
                st.session_state.api_key = alpha_vantage_key
                st.session_state.api_key_set = True
                st.session_state.ai_provider = ai_provider.lower()
                st.success("API Keys saved successfully!")
            else:
                st.error("Please enter the Alpha Vantage API key")


    # Step 2: Stock Category and Selection
    if st.session_state.api_key_set:
        st.header("Step 2: Select Stocks")
        
        # Load stock data
        stock_df = load_stock_data()
        
        # Get categories
        categories = get_stock_categories(stock_df)
        
        # Category selection
        selected_category = st.selectbox(
            "Select a stock category:",
            options=[""] + list(categories),
            index=0,
            key="category_selector"
        )
        
        if selected_category:
            st.session_state.selected_category = selected_category
            
            # Get stocks in selected category
            category_stocks = get_stocks_in_category(stock_df, selected_category)
            
            # Create a searchable dropdown with stock symbols and names from selected category
            stock_options = [f"{row['Symbol']} - {row['Company Name']}" 
                            for _, row in category_stocks.iterrows()]
            
            selected_stocks = st.multiselect(
                f"Select stocks to analyze from {selected_category} category (max 2):",
                options=stock_options,
                max_selections=2
            )
            
            if selected_stocks:
                st.session_state.selected_stocks = [stock.split(" - ")[0] for stock in selected_stocks]
                
            # Display category statistics
            with st.expander(f"📊 {selected_category} Category Statistics"):
                st.write(f"Total stocks in category: {len(category_stocks)}")
                # Add more category-specific statistics as needed

    # [Rest of the code remains the same: Step 3 and 4, Footer]
    if st.session_state.api_key_set and st.session_state.selected_stocks:
        st.header("Step 3: Fetch Data and Analyze")
        
        if st.button("Run Analysis"):
            with st.spinner("Fetching data and generating analysis..."):
                analysis_results = {}
                
                for symbol in st.session_state.selected_stocks:
                    st.subheader(f"Analysis for {symbol}")
                    
                    # Company Overview Section
                    st.write("📊 Company Overview")
                    overview_data = fetch_company_overview(symbol, st.session_state.api_key)
                    if overview_data:
                        with st.expander("Company Description"):
                            st.write(overview_data.get("Description", "No description available"))
                        display_company_metrics(overview_data)
                    
                    # Stock Data and Sentiment Analysis
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("📈 Stock Price Data")
                        stock_data = fetch_stock_data(symbol, st.session_state.api_key)
                        if stock_data is not None:
                            st.dataframe(stock_data.head())
                    
                    with col2:
                        st.write("📰 News Sentiment")
                        sentiment_data = fetch_news_sentiment(symbol, st.session_state.api_key)
                        if sentiment_data is not None:
                            st.dataframe(sentiment_data.head())
                    
                    # Income Statement Section
                    st.write("📈 Income Statement Analysis")
                    annual_reports, quarterly_reports = fetch_income_statement(symbol, st.session_state.api_key)
                    
                    tabs = st.tabs(["Annual Reports", "Quarterly Reports"])
                    with tabs[0]:
                        st.subheader("Annual Financial Data")
                        if annual_reports is not None:
                            st.dataframe(annual_reports)
                            if len(annual_reports) >= 2:
                                latest = annual_reports.iloc[0]
                                previous = annual_reports.iloc[1]
                                revenue_growth = ((float(latest['totalRevenue']) - float(previous['totalRevenue'])) 
                                                / float(previous['totalRevenue']) * 100)
                                st.metric("Revenue Growth", f"{revenue_growth:.1f}%")
                    
                    with tabs[1]:
                        st.subheader("Quarterly Financial Data")
                        if quarterly_reports is not None:
                            st.dataframe(quarterly_reports)
                                        
                            
                    
                    # Store analysis data
                    analysis_results[symbol] = {
                                    "overview": overview_data,
                                    "stock_prices": stock_data,
                                    "sentiment": sentiment_data,
                                    "annual_reports": annual_reports,
                                    "quarterly_reports": quarterly_reports,
                                }
                    st.session_state.analysis_complete = True
                    print(analysis_results[symbol])
                    # Run AI-based Recommendation Analysis
                    if st.session_state.analysis_complete:
                        st.header("Step 4: AI-Powered Recommendation")

                        if ai_provider ==  'Ollama':

                        # Get AI recommendation from Ollama
                            with st.spinner("Generating AI recommendation..."):
                                recommendation = get_recommendation_from_ollama(
                                                            stock_data,
                                                            sentiment_data,
                                                            overview_data,
                                                            quarterly_reports,
                                                            f"Please analyze the stock data, news sentiment, and provide comprehensive investment recommendations.Would You recommend this stock ?")
                                                        
                                st.subheader("📌 AI Recommendation")
                                st.write(recommendation)
                        elif ai_provider == 'OpenAI':
                            with st.spinner("Generating AI recommendation..."):
                                recommendation = get_openai_recommendation(
                                                            stock_data,
                                                            sentiment_data,
                                                            overview_data,
                                                            quarterly_reports,
                                                            f"Please analyze the stock data, news sentiment, and provide comprehensive investment recommendations.Would You recommend this stock ? ",openai_key)
                                                    
                                st.subheader("📌 AI Recommendation")
                                st.write(recommendation)
                        elif ai_provider == 'Claude':
                            with st.spinner("Generating AI recommendation..."):
                                recommendation = get_claude_recommendation(
                                                            stock_data,
                                                            sentiment_data,
                                                            overview_data,
                                                            quarterly_reports,
                                                            f"Please analyze the stock data, news sentiment, and provide comprehensive investment recommendation Would You recommend this stock ?")
                                st.subheader("📌 AI Recommendation")
                                st.write(recommendation)
                    

# with col2:
#     st.subheader("Workflow")

#     st.graphviz_chart("""
#         digraph G {
#             rankdir=TB;
#             node [shape=rect style=rounded fontsize=14 fontname="Arial"];

#             API [label="🔗 API Fetch" style=filled fillcolor=lightblue];
#             AI [label="🤖 AI Processing" style=filled fillcolor=lightgreen];
#             Analysis [label="📊 Data Analysis" style=filled fillcolor=lightyellow];
#             Recommendation [label="💡 Recommendation" style=filled fillcolor=lightcoral];

#             API -> AI;
#             AI -> Analysis;
#             Analysis -> Recommendation;
#         }
#     """)


st.markdown("---")
st.markdown("*This is a financial analysis tool. Always conduct your own research before making investment decisions.*")