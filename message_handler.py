from best_quote_calculator import calculate_best_quote
import json
import logging


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

    # Now your strategy logic can always read from 'local_book'
    # which is updated in real-time (no 1-second lag!)
    calculate_best_quote()
