def calculate_best_quote(local_book: dict = None):
    """Print the current best bid and best ask from the local order book.

    Called by ``MessageHandler.handle_depth_message`` every
    ``QUOTE_EVERY_N_TICKS`` ticks to display the live spread on stdout
    without flooding the console on every 100 ms tick.

    Args:
        local_book (dict): The live order book dict with ``"bids"`` and
            ``"asks"`` sub-dicts keyed by price string.

    Raises:
        ValueError: If *local_book* is ``None``.
    """
    if local_book is not None:
        # Get top of book without a network call
        if not local_book["bids"] or not local_book["asks"]:
            return

        # Get the highest Buy price and lowest Sell price
        best_bid = max(local_book["bids"].keys(), key=float)
        best_ask = min(local_book["asks"].keys(), key=float)
        print(f"Spread: {best_bid} | {best_ask}", end="\r")
    else:
        # Fallback to REST API call if local_book is not provided
        raise ValueError(
            "No local book data available. Please provide the local_book dictionary."
        )
