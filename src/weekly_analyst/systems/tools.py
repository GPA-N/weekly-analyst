"""Investigate tools: the only calls a plan may dispatch in the investigate state (spec 6.2 step 6, 6.4, 6.6)."""

from pathlib import Path

from weekly_analyst.analysis.analyst import attribute_change, concentration, wilson
from weekly_analyst.analysis.sqlgrammar import SQL_SEED, SQL_TEMPERATURE, build_select_grammar, build_sql_prompt
from weekly_analyst.core.components import SnapshotC, ValidationC
from weekly_analyst.core.context import RunContext
from weekly_analyst.core.entity import Entity
from weekly_analyst.core.world import World
from weekly_analyst.data import sanitize
from weekly_analyst.data.named_queries import build_sql
from weekly_analyst.data.query import TABLE_NAME, LocalQuery
from weekly_analyst.systems.common import frame, metric_map, prev_period, week

ROWS_SHOWN = 50  # query rows passed back to the model; the full count is passed too


class ToolRunner:
    """Dispatches one investigate tool call against this run's snapshot."""

    def __init__(self, world: World, entity: Entity, ctx: RunContext) -> None:
        self.world = world
        self.entity = entity
        self.ctx = ctx

    def call_tool(self, tool: str, args: dict) -> dict:
        ctx = self.ctx
        if tool == "local_query":
            return _shown(self._query().run(self._free_sql(args)))
        if tool == "named_query":
            return _shown(self._query().run(build_sql(ctx.metrics, args.get("name"), args.get("params"))))
        if tool == "attribute_change":
            metric = metric_map(ctx)[args["metric"]]
            segment_col = args["segment_col"]
            if segment_col not in metric.segment_columns:
                raise ValueError(f"{segment_col!r} is not a segment of {metric.name}")
            period = args.get("period", self.world.get(self.entity, ValidationC).period)
            col = ctx.metrics.date_column
            df = frame(self.world, self.entity, ctx)
            table = attribute_change(
                week(df, col, prev_period(period)), week(df, col, period), metric, segment_col
            )
            return {
                "segments": [
                    {"segment": _public_value(r["segment"]), "delta": r["delta"], "share": r["share"]}
                    for r in table.iter_rows(named=True)
                ]
            }
        if tool == "wilson":
            low, high = wilson(int(args["successes"]), int(args["n"]))
            return {"low": low, "high": high}
        if tool == "concentration":
            return {"share": concentration([float(v) for v in args["values"]])}
        raise ValueError(f"unknown tool {tool!r}")

    def _free_sql(self, args: dict) -> str:
        """The model writes one SELECT under the grammar. One call, no repair: LocalQuery decides."""
        ctx = self.ctx
        allowed = ctx.metrics.columns()
        schema = {
            name: str(dtype)
            for name, dtype in frame(self.world, self.entity, ctx).schema.items()
            if name in allowed
        }
        prompt = build_sql_prompt(schema, TABLE_NAME, args["question"], args["required_columns"])
        grammar = build_select_grammar(allowed, TABLE_NAME)
        raw = ctx.client.complete(prompt, {}, temperature=SQL_TEMPERATURE, seed=SQL_SEED, grammar=grammar)
        return raw.strip()

    def _query(self) -> LocalQuery:
        if "query" not in self.ctx.cache:
            self.ctx.cache["query"] = LocalQuery(
                Path(self.world.get(self.entity, SnapshotC).parquet),
                self.ctx.metrics.columns(),
                self.ctx.settings,
            )
        return self.ctx.cache["query"]


def _shown(rows: list[dict]) -> dict:
    shown = [{key: _public_value(value) for key, value in row.items()} for row in rows[:ROWS_SHOWN]]
    return {"row_count": len(rows), "rows": shown}


def _public_value(value):
    return sanitize.label(value) if isinstance(value, str) else value
