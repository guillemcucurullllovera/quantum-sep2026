"""
Sentiment-based feature computations for quantum signal generation.

Computes net sentiment as normalized difference between bullish and bearish
positions across base and quote assets, and tracks sentiment momentum.
"""


def compute_net_sentiment(
    base_bull: int,
    base_bear: int,
    base_total: int,
    quote_bull: int,
    quote_bear: int,
    quote_total: int,
) -> float:
    """
    Compute net sentiment as differential bullish/bearish ratio across assets.

    Calculates sentiment for base and quote assets separately, then computes
    the difference to capture relative positioning bias.

    Args:
        base_bull: Number of bullish positions in base asset
        base_bear: Number of bearish positions in base asset
        base_total: Total positions in base asset
        quote_bull: Number of bullish positions in quote asset
        quote_bear: Number of bearish positions in quote asset
        quote_total: Total positions in quote asset

    Returns:
        Net sentiment in range [-1, 1].
        Positive values indicate net bullish bias, negative indicates bearish.
        Returns clipped value in [-1, 1].

    Examples:
        >>> compute_net_sentiment(60, 40, 100, 50, 50, 100)
        0.1
        >>> compute_net_sentiment(40, 60, 100, 70, 30, 100)
        -0.6
    """
    # Calculate sentiment for each asset
    base_sentiment = (base_bull - base_bear) / max(base_total, 1)
    quote_sentiment = (quote_bull - quote_bear) / max(quote_total, 1)

    # Compute net difference
    net_sentiment = base_sentiment - quote_sentiment

    # Clip to [-1, 1]
    return max(-1.0, min(1.0, net_sentiment))


def compute_sentiment_momentum(
    current_sentiment: float, sentiment_3d_ago: float | None = None
) -> float:
    """
    Compute sentiment momentum as first difference of sentiment values.

    Tracks whether sentiment is shifting bullish (positive) or bearish (negative).

    Args:
        current_sentiment: Current net sentiment value (typically in [-1, 1])
        sentiment_3d_ago: Sentiment value from 3 days ago, or None for initial observation

    Returns:
        Sentiment momentum clipped to [-1, 1].
        Returns 0.0 if sentiment_3d_ago is None.

    Examples:
        >>> compute_sentiment_momentum(0.3, 0.1)
        0.2
        >>> compute_sentiment_momentum(-0.2, 0.1)
        -0.3
        >>> compute_sentiment_momentum(0.5, None)
        0.0
    """
    if sentiment_3d_ago is None:
        return 0.0

    momentum = current_sentiment - sentiment_3d_ago
    return max(-1.0, min(1.0, momentum))
