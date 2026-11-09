"""Item key for the reported-item gate and the plan. The Item dataclass itself lives in core/items.py."""

import json

from weekly_analyst.core.items import Item


def item_key(item: Item) -> str:
    """Stable key for an item (metric, period, segment), used by the reported-item gate."""
    return json.dumps(
        [item.metric, item.period, [list(pair) for pair in item.segment]], separators=(",", ":")
    )
