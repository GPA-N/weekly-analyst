"""The run's states: one system class per state, the fixed order SYSTEMS, and the helpers they share.

The order is spec 6.2: extract, validate, detect, plan, investigate, draft, check, outbox, remember. The
scheduler runs them in this order; the model never chooses it.
"""

from weekly_analyst.systems.check import CheckSystem
from weekly_analyst.systems.detect import DetectSystem
from weekly_analyst.systems.draft import DraftSystem
from weekly_analyst.systems.extract import ExtractSystem
from weekly_analyst.systems.investigate import InvestigateSystem
from weekly_analyst.systems.outbox import OutboxSystem
from weekly_analyst.systems.plan import PlanSystem
from weekly_analyst.systems.remember import RememberSystem
from weekly_analyst.systems.validate import ValidateSystem

SYSTEMS = (
    ExtractSystem(), ValidateSystem(), DetectSystem(), PlanSystem(), InvestigateSystem(),
    DraftSystem(), CheckSystem(), OutboxSystem(), RememberSystem(),
)
