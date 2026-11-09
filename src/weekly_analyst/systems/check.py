"""Check state: render the draft PDF, check its extracted text, and redraft at most twice (spec 6.3)."""

from pathlib import Path

import pypdf

from weekly_analyst.analysis.checks import check_report
from weekly_analyst.analysis.draft import DraftFields
from weekly_analyst.core.components import CheckC, DraftC, ItemsC, ReportC, SnapshotC, SourceC, ValidationC
from weekly_analyst.core.context import RunContext
from weekly_analyst.core.entity import Entity
from weekly_analyst.core.system import Halt, System
from weekly_analyst.core.world import World
from weekly_analyst.report import render_pdf
from weekly_analyst.report.model import ReportTooLong
from weekly_analyst.systems.draft import request_draft
from weekly_analyst.systems.report_model import build_report

REDRAFT_ROUNDS = 2  # spec 6.3: failed checks return the draft to the generator at most twice


class CheckSystem(System):
    name = "check"
    requires = (DraftC, SnapshotC, SourceC, ValidationC, ItemsC)
    produces = (CheckC, ReportC)

    def run(self, world: World, entity: Entity, ctx: RunContext) -> None:
        parquet = Path(world.get(entity, SnapshotC).parquet)
        round_no = world.get(entity, CheckC).round_no if world.has(entity, CheckC) else 0
        while True:
            draft = DraftFields.model_validate(world.get(entity, DraftC).draft)
            pdf = world.run_dir(entity) / f"draft-{round_no}.pdf"
            try:
                model = build_report(world, entity, ctx, draft)
                rendered = render_pdf(model, pdf, status="draft", paper=ctx.paper)
            except (ReportTooLong, ValueError) as err:
                raise Halt(f"report: {err}") from None
            model_text = "\n".join(
                [draft.headline, draft.recommended_action, *(f.so_what for f in draft.findings), *draft.caveats]
            )
            failures = check_report(
                pdf_text(pdf),
                rendered.shown_numbers,
                ctx.metrics,
                parquet,
                exempt=rendered.structural_numbers,
                model_text=model_text,
            )
            if not failures:
                world.add(entity, CheckC(round_no=round_no, failures=[]))
                # Taken after render_pdf, which sets the prior-period deltas on the shown numbers.
                world.add(
                    entity,
                    ReportC(
                        pdf=str(pdf),
                        pages=rendered.pages,
                        model=model.to_dict(),
                        shown_numbers=[shown.to_dict() for shown in rendered.shown_numbers],
                        structural_numbers=rendered.structural_numbers,
                        model_text=model_text,
                        paper=ctx.paper,
                    ),
                )
                return
            described = [f"{f.kind}: {f.detail}" for f in failures]
            if round_no >= REDRAFT_ROUNDS:
                raise Halt(f"check failed after {REDRAFT_ROUNDS} redraft rounds: " + "; ".join(described))
            round_no += 1
            world.add(entity, CheckC(round_no=round_no, failures=described))
            redraft = request_draft(world, entity, ctx, described)
            world.add(entity, DraftC(draft=redraft.model_dump(mode="json")))
            world.save(entity)  # a resumed run continues from this redraft round


def pdf_text(path: Path) -> str:
    return "\n".join(page.extract_text() or "" for page in pypdf.PdfReader(str(path)).pages)
