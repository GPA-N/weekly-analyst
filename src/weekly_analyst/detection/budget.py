"""Weekly alert budget and gating: alerts a run may still send, and which ranked items pass (spec 6.6)."""

from weekly_analyst.core.config import Settings
from weekly_analyst.core.items import Item
from weekly_analyst.core.store import StateStore
from weekly_analyst.detection.items import item_key


class AlertBudget:
    """Owns the weekly alert budget K and the per-item gate (minimum change, reported-item gate).

    `store` is needed only for remaining(); `settings` only for remaining() and insufficient_history().
    """

    def __init__(self, store: StateStore | None, settings: Settings | None) -> None:
        self.store = store
        self.settings = settings

    def remaining(self, iso_week: str) -> int:
        """Alerts a run may still send in this ISO week: K minus alerts already sent (spec 6.6)."""
        return max(0, self.settings.alert_budget_k - self.store.alerts_sent_in_week(iso_week))

    def insufficient_history(self, n_periods: int) -> bool:
        """True when there is too little history to alert (cold start, spec 6.6)."""
        return n_periods < self.settings.cold_start_periods

    def select(
        self,
        items: list[Item],
        remaining: int,
        min_change: dict[str, float],
        already_reported: set[str],
    ) -> list[Item]:
        """Pick alerts from rank-ordered items (spec 6.6), keeping input order.

        An item needs detail["abs_change"] >= min_change[metric]; a NaN change fails this gate.
        An item whose key is already reported is skipped unless detail["material_change"] is set.
        At most `remaining` items are returned.
        """
        if remaining <= 0:
            return []
        selected: list[Item] = []
        for item in items:
            if not (item.detail["abs_change"] >= min_change[item.metric]):
                continue
            if item_key(item) in already_reported and not item.detail.get("material_change", False):
                continue
            selected.append(item)
            if len(selected) == remaining:
                break
        return selected
