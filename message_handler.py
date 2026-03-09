import json
import logging
from collections import deque
from best_quote_calculator import calculate_best_quote

stats_history_order_book = deque(maxlen=1000)


def handle_depth_message(_, message, local_book=None):
    data = json.loads(message)

    # Skip the subscription confirmation message (only has 'id' and 'result')
    if "result" in data and "b" not in data:
        logging.info("WebSocket subscription confirmed.")
        return
    # SYNC LOGIC: Ignore updates that are older than our snapshot
    if data["u"] <= local_book["lastUpdateId"]:
        return
    # 1. Logic to sync with lastUpdateId goes here
    # 2. Update local_book dictionary — bids ('b') and asks ('a')
    for price, qty in data.get("b", []):
        if float(qty) == 0:
            local_book["bids"].pop(price, None)
        else:
            local_book["bids"][price] = qty

    for price, qty in data.get("a", []):
        if float(qty) == 0:
            local_book["asks"].pop(price, None)
        else:
            local_book["asks"][price] = qty

    stats_history_order_book.append(
        {
            "timestamp": data["E"],
            "lastUpdateId": data["u"],
            "best_bids": max(local_book["bids"].keys(), key=float),
            "best_asks": min(local_book["asks"].keys(), key=float),
        }
    )

    # Now your strategy logic can always read from 'local_book'
    # which is updated in real-time (no 1-second lag!)
    calculate_best_quote(local_book)
