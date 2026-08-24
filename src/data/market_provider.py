import os
import datetime as dt
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import pandas as pd
import requests
import yfinance as yf


@dataclass
class QuoteData:
    symbol: str
    company_name: str
    price: float
    change: float
    change_percent: float
    currency: str
    market_status: str          # "Real-time", "Market closed", "Delayed market data", etc.
    is_market_open: bool
    last_updated: str           # Formatted string "HH:MM:SS" or "YYYY-MM-DD HH:MM:SS"
    timestamp_iso: str
    data_label: str             # "Real-time" vs "Latest available price" vs "Delayed market data"
    exchange: str


@dataclass
class HistoricalData:
    symbol: str
    period: str                 # "1M" or "5M"
    df: pd.DataFrame            # Index: Datetime/Date, Columns: [Open, High, Low, Close, Volume]
    trading_days_count: int
    start_date: str
    end_date: str


class BaseMarketProvider(ABC):
    """Abstract Base Class for Market Data Providers."""

    @abstractmethod
    def search_symbols(self, query: str) -> List[Dict[str, str]]:
        """Search for matching stock symbols."""
        pass

    @abstractmethod
    def get_latest_quote(self, symbol: str) -> QuoteData:
        """Fetch the latest/current market quote."""
        pass

    @abstractmethod
    def get_historical_data(self, symbol: str, period: str = "5M") -> HistoricalData:
        """Fetch recent historical OHLCV data for 1M or 5M."""
        pass


class YahooFinanceProvider(BaseMarketProvider):
    """Yahoo Finance Market Data Provider."""

    # Curated popular symbols for search auto-suggestions
    POPULAR_STOCKS = [
        {"symbol": "AAPL", "name": "Apple Inc.", "exchange": "NASDAQ", "country": "US"},
        {"symbol": "MSFT", "name": "Microsoft Corporation", "exchange": "NASDAQ", "country": "US"},
        {"symbol": "GOOGL", "name": "Alphabet Inc. (Class A)", "exchange": "NASDAQ", "country": "US"},
        {"symbol": "AMZN", "name": "Amazon.com Inc.", "exchange": "NASDAQ", "country": "US"},
        {"symbol": "NVDA", "name": "NVIDIA Corporation", "exchange": "NASDAQ", "country": "US"},
        {"symbol": "TSLA", "name": "Tesla Inc.", "exchange": "NASDAQ", "country": "US"},
        {"symbol": "META", "name": "Meta Platforms Inc.", "exchange": "NASDAQ", "country": "US"},
        {"symbol": "JPM", "name": "JPMorgan Chase & Co.", "exchange": "NYSE", "country": "US"},
        {"symbol": "RELIANCE.NS", "name": "Reliance Industries Ltd.", "exchange": "NSE", "country": "IN"},
        {"symbol": "TCS.NS", "name": "Tata Consultancy Services Ltd.", "exchange": "NSE", "country": "IN"},
        {"symbol": "INFY.NS", "name": "Infosys Ltd.", "exchange": "NSE", "country": "IN"},
        {"symbol": "HDFCBANK.NS", "name": "HDFC Bank Ltd.", "exchange": "NSE", "country": "IN"},
        {"symbol": "ICICIBANK.NS", "name": "ICICI Bank Ltd.", "exchange": "NSE", "country": "IN"},
        {"symbol": "TATAMOTORS.NS", "name": "Tata Motors Ltd.", "exchange": "NSE", "country": "IN"},
    ]

    def search_symbols(self, query: str) -> List[Dict[str, str]]:
        query_upper = query.strip().upper()
        if not query_upper:
            return self.POPULAR_STOCKS[:6]

        results: List[Dict[str, str]] = []
        seen = set()

        # Check local curated list first
        for stock in self.POPULAR_STOCKS:
            if (query_upper in stock["symbol"].upper() or 
                query_upper in stock["name"].upper()):
                results.append(stock)
                seen.add(stock["symbol"])

        # Also search via Yahoo Finance API
        try:
            url = f"https://query2.finance.yahoo.com/v1/finance/search?q={requests.utils.quote(query)}&quotesCount=8&newsCount=0"
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            resp = requests.get(url, headers=headers, timeout=4)
            if resp.status_code == 200:
                data = resp.json()
                for q in data.get("quotes", []):
                    quote_type = q.get("quoteType", "")
                    if quote_type in ("EQUITY", "ETF"):
                        sym = q.get("symbol", "")
                        if sym and sym not in seen:
                            results.append({
                                "symbol": sym,
                                "name": q.get("shortname") or q.get("longname") or sym,
                                "exchange": q.get("exchange", "Unknown"),
                                "country": "IN" if sym.endswith(".NS") or sym.endswith(".BO") else "US"
                            })
                            seen.add(sym)
        except Exception:
            # Fallback gracefully to curated matches
            pass

        # Handle Indian ticker without suffix (e.g. user typed "RELIANCE" -> suggest "RELIANCE.NS")
        if not query_upper.endswith(".NS") and not query_upper.endswith(".BO"):
            ns_sym = f"{query_upper}.NS"
            if ns_sym not in seen:
                for stock in self.POPULAR_STOCKS:
                    if stock["symbol"] == ns_sym:
                        results.append(stock)
                        break

        return results[:10]

    def get_latest_quote(self, symbol: str) -> QuoteData:
        symbol = symbol.strip().upper()
        ticker = yf.Ticker(symbol)
        
        info = {}
        try:
            info = ticker.info or {}
        except Exception:
            info = {}

        # Fallback price fetch from fast_info or recent 1d history
        current_price = None
        previous_close = None
        currency = info.get("currency", "USD")
        company_name = info.get("shortName") or info.get("longName") or symbol
        exchange = info.get("exchange", "Unknown")
        market_state = info.get("marketState", "CLOSED").upper()

        if "regularMarketPrice" in info and info["regularMarketPrice"] is not None:
            current_price = float(info["regularMarketPrice"])
            previous_close = float(info.get("regularMarketPreviousClose") or info.get("previousClose") or current_price)
        elif hasattr(ticker, "fast_info") and ticker.fast_info is not None:
            try:
                fast = ticker.fast_info
                current_price = float(fast.last_price)
                previous_close = float(fast.previous_close or current_price)
                currency = fast.currency or currency
                exchange = fast.exchange or exchange
            except Exception:
                pass

        if current_price is None or previous_close is None:
            # Fetch 5-day 1-minute or daily data to find latest price
            hist = ticker.history(period="5d")
            if hist.empty:
                raise ValueError(f"Unable to retrieve market price for symbol '{symbol}'. Verify the ticker name.")
            current_price = float(hist["Close"].iloc[-1])
            if len(hist) > 1:
                previous_close = float(hist["Close"].iloc[-2])
            else:
                previous_close = current_price

        change = round(current_price - previous_close, 2)
        change_pct = round((change / previous_close) * 100, 2) if previous_close else 0.0

        # Market Open/Closed and Data labeling
        is_open = market_state in ("REGULAR", "OPEN")
        if is_open:
            market_status = "Market Open"
            data_label = "Real-time"
        else:
            market_status = "Market Closed"
            data_label = "Latest available price"

        now = dt.datetime.now()
        timestamp_iso = now.isoformat()
        last_updated = now.strftime("%I:%M:%S %p")

        return QuoteData(
            symbol=symbol,
            company_name=company_name,
            price=round(current_price, 2),
            change=change,
            change_percent=change_pct,
            currency=currency,
            market_status=market_status,
            is_market_open=is_open,
            last_updated=last_updated,
            timestamp_iso=timestamp_iso,
            data_label=data_label,
            exchange=exchange
        )

    def get_historical_data(self, symbol: str, period: str = "5M") -> HistoricalData:
        symbol = symbol.strip().upper()
        ticker = yf.Ticker(symbol)
        
        # 1 Month = ~21 trading days (fetch 1mo)
        # 5 Months = ~105 trading days (fetch 5mo or 6mo for safety)
        yf_period = "1mo" if period.upper() == "1M" else "5mo"
        
        df = ticker.history(period=yf_period, interval="1d")
        if df.empty:
            # Try date-based fallback
            days = 32 if period.upper() == "1M" else 155
            start = dt.datetime.now() - dt.timedelta(days=days)
            df = yf.download(symbol, start=start, end=dt.datetime.now(), progress=False)

        if df.empty:
            raise ValueError(f"No historical market data returned for '{symbol}' over {period} period.")

        # Ensure standard OHLCV columns and drop multi-index levels if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Historical data missing essential column '{col}' for symbol '{symbol}'.")

        df = df[required_cols].copy()
        df.dropna(inplace=True)
        df.sort_index(inplace=True)

        trading_days = len(df)
        start_date = df.index[0].strftime("%Y-%m-%d")
        end_date = df.index[-1].strftime("%Y-%m-%d")

        return HistoricalData(
            symbol=symbol,
            period="1M" if period.upper() == "1M" else "5M",
            df=df,
            trading_days_count=trading_days,
            start_date=start_date,
            end_date=end_date
        )


class AlphaVantageProvider(BaseMarketProvider):
    """Alpha Vantage Market Data Provider (Configurable via .env)."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("MARKET_DATA_API_KEY", "")
        self._fallback = YahooFinanceProvider()

    def search_symbols(self, query: str) -> List[Dict[str, str]]:
        if not self.api_key:
            return self._fallback.search_symbols(query)
        try:
            url = f"https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords={query}&apikey={self.api_key}"
            resp = requests.get(url, timeout=5).json()
            matches = resp.get("bestMatches", [])
            results = []
            for m in matches:
                results.append({
                    "symbol": m.get("1. symbol"),
                    "name": m.get("2. name"),
                    "exchange": m.get("4. region", "US"),
                    "country": "US"
                })
            return results if results else self._fallback.search_symbols(query)
        except Exception:
            return self._fallback.search_symbols(query)

    def get_latest_quote(self, symbol: str) -> QuoteData:
        if not self.api_key:
            return self._fallback.get_latest_quote(symbol)
        try:
            url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={self.api_key}"
            resp = requests.get(url, timeout=5).json()
            quote = resp.get("Global Quote", {})
            if not quote or "05. price" not in quote:
                return self._fallback.get_latest_quote(symbol)
            price = float(quote["05. price"])
            change = float(quote.get("09. change", 0.0))
            change_pct = float(quote.get("10. change percent", "0%").replace("%", ""))
            now = dt.datetime.now()
            return QuoteData(
                symbol=symbol,
                company_name=symbol,
                price=round(price, 2),
                change=round(change, 2),
                change_percent=round(change_pct, 2),
                currency="USD",
                market_status="Delayed market data",
                is_market_open=False,
                last_updated=now.strftime("%I:%M:%S %p"),
                timestamp_iso=now.isoformat(),
                data_label="Delayed market data",
                exchange="Global"
            )
        except Exception:
            return self._fallback.get_latest_quote(symbol)

    def get_historical_data(self, symbol: str, period: str = "5M") -> HistoricalData:
        # Fallback to YahooFinance for robust structured historical series
        return self._fallback.get_historical_data(symbol, period)


def get_market_provider() -> BaseMarketProvider:
    """Factory function returning configured market data provider."""
    provider_type = os.getenv("MARKET_DATA_PROVIDER", "yfinance").lower()
    api_key = os.getenv("MARKET_DATA_API_KEY", "")

    if provider_type == "alphavantage" and api_key:
        return AlphaVantageProvider(api_key=api_key)
    return YahooFinanceProvider()
