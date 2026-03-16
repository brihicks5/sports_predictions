"""Polymarket CLOB client for browsing markets and placing trades.

Wraps the official py-clob-client library with a simplified interface
for searching markets, checking prices, and buying/selling positions.

Setup:
    pip install py-clob-client
    # Add to .env:
    #   POLYMARKET_PRIVATE_KEY=0x...    (Settings > Cash > Export Private Key)
    #   POLYMARKET_FUNDER_ADDRESS=0x... (proxy wallet address from profile)

Usage:
    client = PolymarketClient()
    markets = client.search_markets("Duke Blue Devils")
    price = client.get_price(market["token_id"])
    client.buy(token_id, price=0.55, size=20)       # limit order
    client.buy(token_id, amount=10.0)                # market order
    client.get_balance()
    client.get_positions()
"""

LOW_BALANCE_THRESHOLD = 5.00  # USD — warn when balance drops below this


class PolymarketClient:
    """Client for interacting with Polymarket's CLOB.

    Connects using a Polygon proxy wallet (signature type 1), which is
    the default for accounts created on polymarket.com and funded via
    bank account.

    Methods:
        search_markets(query) -> list[dict]
            Search Gamma API for markets matching a query string.
            Returns list of market dicts with token_id, question,
            outcomes, and current prices.

        get_price(token_id) -> dict
            Get current best bid/ask/midpoint for a token.
            Returns {"bid": float, "ask": float, "mid": float}.

        get_order_book(token_id) -> dict
            Get full order book (bids and asks) for a token.

        buy(token_id, price=None, size=None, amount=None, order_type="GTC") -> dict
            Place a buy order. Two modes:

            Limit order (default):
                buy(token_id, price=0.55, size=20)
                Places a GTC limit order for 20 shares at $0.55.
                Sits on the book until filled or cancelled.

            Market order:
                buy(token_id, amount=10.0)
                Spends $10 immediately at the current best ask.
                Uses FOK (fill-or-kill) — fails if not enough
                liquidity.

            Returns order confirmation dict.
            Warns if balance drops below $5 after the trade.

        sell(token_id, price=None, size=None, amount=None, order_type="GTC") -> dict
            Place a sell order. Same modes as buy().

        get_balance() -> float
            Get available USDC.e balance in the funder wallet.
            Prints a warning if below $5.

        get_positions() -> list[dict]
            Get all open positions. Returns list of dicts with
            token_id, market question, size, avg_price, and
            current_value.

        cancel_order(order_id) -> bool
            Cancel an open order. Returns True if successful.

        cancel_all_orders() -> int
            Cancel all open orders. Returns count cancelled.
    """

    pass
