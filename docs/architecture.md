# Architecture (target layout, OOP and ECS)

Behavior does not change in this refactor. The existing test suite is the safety net: every stage ends with the full suite green, and a stage that needs a test edit only changes import paths.

## ECS mapping

The fixed state machine in design section 6.2 maps onto an entity-component-system:

- **Entity:** a run. It is an id and nothing else (`Entity(run_id)`). Item candidates stay plain data inside components, because they never act on their own.
- **Components:** plain frozen dataclasses, one per piece of run data, JSON-serialisable: `SnapshotC` (hash, parquet path, rows), `ValidationC`, `ItemsC` (ranked items, material items), `PlanC`, `EvidenceC`, `DraftC`, `ReportC` (pdf path, shown numbers, structural numbers), `CheckC` (failures), `OutboxC`, `HaltC` (reason). A run's progress is exactly the set of components it holds, so resume means "load the entity's components and continue with the first system whose outputs are missing".
- **World:** the component store for all runs. `World.add(entity, component)`, `World.get(entity, ComponentType)`, `World.has(entity, ComponentType)`, `World.save(entity)` and `World.load(run_id)` (JSON progress file per run, atomic write). It replaces the hand-written progress file in `runner.py`.
- **Systems:** one class per state, each declaring `requires: tuple[type, ...]` and `produces: tuple[type, ...]` and implementing `run(world, entity, ctx) -> None`. Systems: `ExtractSystem`, `ValidateSystem`, `DetectSystem`, `PlanSystem`, `InvestigateSystem`, `DraftSystem`, `CheckSystem`, `OutboxSystem`, `RememberSystem`.
- **Scheduler:** the fixed order is a tuple of systems. It runs each system whose `produces` components are missing, saves after each, checks the snapshot hash and wall time before each (restart or halt), and stops on `HaltC`. The model still never chooses the order.
- **Context (`RunContext`):** the non-component services a system needs: settings, metrics file, model client, state store, outbox, clock. Passed in, never global.

## OOP seams

- `Detector` abstract base class with `score(ctx) -> dict[key, float]`; concrete `ZScoreDetector`, `DriftDetector`, `CusumDetector`, `ValidationDetector`. A `RankBlend` class combines them; `AlertBudget` class owns the weekly budget and gating.
- `ModelClient` Protocol (already exists): `LlamaCppClient`, `ScriptedClient`.
- `Renderer` Protocol with `ReportLabRenderer` (the chosen renderer can change after the spike without touching callers).
- `Sender` Protocol for delivery; `Outbox` class owns approve, stale guard and purge.
- `MetricsFile`, `Settings`, `StateStore`, `LocalQuery` stay classes.

## Layout

```
src/weekly_analyst/
  core/        config.py  store.py  items.py  text.py  pdfscan.py  errors.py  entity.py  components.py  world.py  system.py  scheduler.py  context.py
  data/        metrics_file.py  extract.py  validate.py  metrics_calc.py  recompute.py  query.py  sanitize.py
  detection/   base.py  zscore.py  drift.py  cusum.py  validation.py  blend.py  budget.py  items.py
  analysis/    analyst.py  plan.py  draft.py  model.py  checks.py
  report/      model.py  charts.py  renderer.py
  delivery/    outbox.py  watch.py
  systems/     extract.py  validate.py  detect.py  plan.py  investigate.py  draft.py  check.py  outbox.py  remember.py
  runner.py    (thin facade: run(), resume(), RunConfig, RunOutcome, RunLocked, lock)
  cli.py
  sim/         generate.py  events.py
eval/          scoring.py  controls.py
```

## Rules

- Files stay under about 250 lines; `runner.py` (759), `report.py` (513) and `detect.py` (348) are split.
- No module imports "upward": `core` imports nothing from the other packages; `data`, `detection`, `analysis`, `report`, `delivery` import `core` and `data` only as needed; `systems` import everything below; `runner` and `cli` import `systems`.
- Public names keep their current spelling so tests only change import paths. During a stage, a moved module may keep a one-line re-export at the old path until the stage's last step, then the shim is deleted.
- Every package has an `__init__.py` with a short docstring. Docstrings on classes say what they own.

Rulings after stage 1 (ticket 039): pdfscan, store, Item and the numeral helpers live in core/ because delivery, plan and checks import them and the dependency rule forbids cross-package imports between the sibling packages.
