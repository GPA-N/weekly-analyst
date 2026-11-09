"""Report model: the input the fixed PDF template renders, built from the draft and the run's components (spec 6.8)."""

import polars as pl

from weekly_analyst.analysis.draft import DraftFields
from weekly_analyst.analysis.plan import _item_key as plan_item_key
from weekly_analyst.core.components import EvidenceC, SourceC, ValidationC
from weekly_analyst.core.context import RunContext
from weekly_analyst.core.entity import Entity
from weekly_analyst.core.items import Item
from weekly_analyst.core.world import World
from weekly_analyst.data.metrics_calc import compute
from weekly_analyst.report.model import DataTable, Finding, Kpi, MetricDefinition, ReportModel, ShownNumber
from weekly_analyst.systems.common import frame, metric_map, prev_period, public_item, raw_items
from weekly_analyst.systems.report_findings import change_text, finding_for, level_text


def build_report(world: World, entity: Entity, ctx: RunContext, draft: DraftFields) -> ReportModel:
    df = frame(world, entity, ctx)
    validation = world.get(entity, ValidationC)
    period = validation.period
    prev = prev_period(period)
    items = raw_items(world, entity)
    texts = {f.item_key: f for f in draft.findings}
    findings: list[Finding] = []
    tables: list[DataTable] = []
    for item in items:
        found, table = finding_for(ctx, df, item, texts[plan_item_key(public_item(item))])
        findings.append(found)
        tables.append(table)
    snapshot_date = df[ctx.metrics.date_column].max().isoformat()
    risks = list(draft.caveats) + [
        f"{f['kind'].replace('_', ' ')}: {f['detail']}" for f in validation.findings
    ]
    failed = [r for r in world.get(entity, EvidenceC).results if "error" in r]
    if failed:
        risks.append(f"{len(failed)} planned follow-up check(s) could not be run and are not reflected in this report")
    if validation.cold_start:
        risks.append(
            f"insufficient history: fewer than {ctx.settings.cold_start_periods} prior weeks, "
            "so no change alerts were sent"
        )
    next_checks = [f"Check the source rows behind {plan_item_key(public_item(item))}" for item in items]
    if any(f["kind"] == "schema_change" for f in validation.findings):
        next_checks.append("Confirm the schema change with the owner of the source file")
    return ReportModel(
        report_name="Weekly analysis",
        reporting_period=period,
        prior_period=prev,
        data_as_of=snapshot_date,
        prepared_for=", ".join(ctx.recipients) or "Recipients not set",
        report_version="1",
        run_id=entity.run_id,
        snapshot_date=snapshot_date,
        snapshot_hash=world.get(entity, SourceC).snapshot_hash,
        headline=draft.headline,
        recommended_action=draft.recommended_action,
        kpis=_kpis(ctx, df, period, prev, items),
        findings=findings,
        risks_caveats=risks,
        next_checks=next_checks,
        metric_definitions=[
            MetricDefinition(
                name=m.name,
                definition=m.definition,
                owner=m.owner,
                includes=m.inclusions,
                excludes=m.exclusions,
                version=str(m.version),
                checksum=ctx.metrics.checksum,
            )
            for m in ctx.metrics.metrics
        ],
        data_tables=tables,
    )


def _kpis(ctx: RunContext, df: pl.DataFrame, period: str, prev: str, items: list[Item]) -> list[Kpi]:
    kpis = []
    by_name = metric_map(ctx)
    for name in dict.fromkeys(item.metric for item in items):
        metric = by_name[name]
        now = compute(df, metric, None, period)
        before = compute(df, metric, None, prev)
        if now is None or before is None:
            continue
        kpis.append(
            Kpi(
                label=metric.name,
                value=ShownNumber(level_text(now), metric.name, None, period),
                vs_prior=ShownNumber(change_text(now - before), metric.name, None, period),
            )
        )
    return kpis
