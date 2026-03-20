from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
import re
from typing import Any, MutableMapping

from .knowledge import format_public_sources, retrieve_knowledge
from .safety import find_forbidden_language


PORTFOLIO_SESSION_KEY = "portfolio_registry"
PORTFOLIO_EXPORT_SESSION_KEY = "portfolio_last_export"
ARTIFACT_TYPES = ("final_plan", "internal_comm", "executive_update")
DEFAULT_STATUS_LABEL = "Draft"
DEFAULT_DRAFT_NOTE = "Simulation draft prepared with AI assistance; review before external sharing."

ARTIFACT_DISPLAY_NAMES = {
    "final_plan": "Final plan",
    "internal_comm": "Internal communication",
    "executive_update": "Executive update",
}

ARTIFACT_NAMESPACE_MAP = {
    "final_plan": ["ceo", "chro", "regional"],
    "internal_comm": ["regional", "chro"],
    "executive_update": ["ceo", "chro", "regional"],
}

SECTION_TITLES = {
    "cover": "Portfolio Pack",
    "final_plan": "Final Plan",
    "internal_comm": "Internal Communications",
    "executive_update": "Executive Update",
    "sources": "Sources And Notes",
}


class PortfolioError(ValueError):
    pass


@dataclass
class PortfolioArtifact:
    artifact_type: str
    title: str
    body_markdown: str
    status_label: str
    created_at: str
    source_notes: list[str] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PortfolioPack:
    thread_id: str
    exported_at: str
    final_plan: PortfolioArtifact
    internal_comms: list[PortfolioArtifact]
    executive_update: PortfolioArtifact
    draft_note: str = DEFAULT_DRAFT_NOTE


@dataclass
class ExportedPortfolioPack:
    file_name: str
    file_path: Path
    pdf_bytes: bytes
    pack: PortfolioPack


def _utc_timestamp(created_at: datetime | None = None) -> str:
    current = created_at or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    current = current.astimezone(timezone.utc).replace(microsecond=0)
    return current.isoformat().replace("+00:00", "Z")


def _file_timestamp(created_at: datetime | None = None) -> str:
    current = created_at or datetime.now(timezone.utc)
    if current.tzinfo is not None:
        current = current.astimezone(timezone.utc).replace(tzinfo=None)
    return current.strftime("%Y%m%d-%H%M%S")


def _split_source_notes(source_notes: str | list[str] | None) -> list[str]:
    if source_notes is None:
        return []
    if isinstance(source_notes, str):
        parts = [part.strip(" -") for part in re.split(r"[\n;]+", source_notes)]
        return [part for part in parts if part]
    return [str(note).strip() for note in source_notes if str(note).strip()]


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _artifact_display_name(artifact_type: str) -> str:
    try:
        return ARTIFACT_DISPLAY_NAMES[artifact_type]
    except KeyError as exc:
        raise PortfolioError(f"Unsupported artifact_type '{artifact_type}'.") from exc


def _default_title(artifact_type: str, metadata: dict[str, str] | None = None) -> str:
    metadata = metadata or {}
    if artifact_type == "internal_comm":
        comm_type = metadata.get("comm_type", "post").replace("_", " ").title()
        return f"{comm_type} Draft"
    return _artifact_display_name(artifact_type)


def _registry_template() -> dict[str, Any]:
    return {
        "final_plan": None,
        "internal_comm": [],
        "executive_update": None,
    }


def _ensure_registry(session_state: MutableMapping[str, Any]) -> dict[str, Any]:
    registry = session_state.get(PORTFOLIO_SESSION_KEY)
    if not isinstance(registry, dict):
        registry = _registry_template()
        session_state[PORTFOLIO_SESSION_KEY] = registry
        return registry

    if "final_plan" not in registry:
        registry["final_plan"] = None
    if "internal_comm" not in registry or not isinstance(registry["internal_comm"], list):
        registry["internal_comm"] = []
    if "executive_update" not in registry:
        registry["executive_update"] = None
    return registry


def _normalize_notes(artifact_type: str, body_markdown: str, source_notes: str | list[str] | None) -> list[str]:
    notes: list[str] = []
    chunks = retrieve_knowledge(
        body_markdown,
        namespaces=ARTIFACT_NAMESPACE_MAP.get(artifact_type, ["ceo", "chro", "regional"]),
        top_k=3,
    )
    public_sources = format_public_sources(chunks)
    if public_sources:
        notes.extend(line.replace("- ", "", 1).strip() for line in public_sources.splitlines() if line.strip())
    notes.extend(_split_source_notes(source_notes))
    return _dedupe_preserve_order(notes)


def save_portfolio_artifact(
    session_state: MutableMapping[str, Any],
    *,
    artifact_type: str,
    body_markdown: str,
    title: str | None = None,
    source_notes: str | list[str] | None = None,
    metadata: dict[str, str] | None = None,
    created_at: datetime | None = None,
    status_label: str = DEFAULT_STATUS_LABEL,
) -> PortfolioArtifact:
    if artifact_type not in ARTIFACT_TYPES:
        raise PortfolioError(f"Unsupported artifact_type '{artifact_type}'.")
    if not body_markdown.strip():
        raise PortfolioError("Cannot save an empty portfolio artifact.")

    registry = _ensure_registry(session_state)
    metadata = dict(metadata or {})
    artifact = PortfolioArtifact(
        artifact_type=artifact_type,
        title=(title or "").strip() or _default_title(artifact_type, metadata),
        body_markdown=body_markdown.strip(),
        status_label=status_label.strip() or DEFAULT_STATUS_LABEL,
        created_at=_utc_timestamp(created_at),
        source_notes=_normalize_notes(artifact_type, body_markdown, source_notes),
        metadata=metadata,
    )

    if artifact_type == "internal_comm":
        registry["internal_comm"] = list(registry.get("internal_comm", [])) + [artifact]
    else:
        registry[artifact_type] = artifact
    return artifact


def get_portfolio_status(session_state: MutableMapping[str, Any]) -> dict[str, Any]:
    registry = _ensure_registry(session_state)
    final_plan = registry.get("final_plan")
    internal_comms = list(registry.get("internal_comm", []))
    executive_update = registry.get("executive_update")

    missing: list[str] = []
    if final_plan is None:
        missing.append("final_plan")
    if not internal_comms:
        missing.append("internal_comm")
    if executive_update is None:
        missing.append("executive_update")

    return {
        "is_ready": not missing,
        "missing": missing,
        "counts": {
            "final_plan": int(final_plan is not None),
            "internal_comm": len(internal_comms),
            "executive_update": int(executive_update is not None),
        },
        "titles": {
            "final_plan": getattr(final_plan, "title", None),
            "internal_comm": [artifact.title for artifact in internal_comms],
            "executive_update": getattr(executive_update, "title", None),
        },
    }


def _build_pack(session_state: MutableMapping[str, Any], thread_id: str, created_at: datetime | None = None) -> PortfolioPack:
    registry = _ensure_registry(session_state)
    status = get_portfolio_status(session_state)
    if not status["is_ready"]:
        missing = ", ".join(_artifact_display_name(name) for name in status["missing"])
        raise PortfolioError(f"Portfolio Pack is incomplete. Missing: {missing}.")

    final_plan = registry["final_plan"]
    internal_comms = list(registry["internal_comm"])
    executive_update = registry["executive_update"]
    artifacts = [final_plan, *internal_comms, executive_update]

    violations: list[str] = []
    for artifact in artifacts:
        matches = find_forbidden_language(artifact.body_markdown)
        if matches:
            violations.append(f"{artifact.title}: {', '.join(matches)}")
    if violations:
        raise PortfolioError(
            "Portfolio Pack cannot be exported until forbidden language is removed from: "
            + "; ".join(violations)
        )

    return PortfolioPack(
        thread_id=thread_id,
        exported_at=_utc_timestamp(created_at),
        final_plan=final_plan,
        internal_comms=internal_comms,
        executive_update=executive_update,
    )


def _filename(thread_id: str, created_at: datetime | None = None) -> str:
    safe_thread_id = re.sub(r"[^a-zA-Z0-9_-]+", "-", thread_id).strip("-") or "session"
    return f"portfolio-pack-{safe_thread_id}-{_file_timestamp(created_at)}.pdf"


def _markdown_to_blocks(text: str) -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            blocks.append(("spacer", ""))
            continue
        if line.startswith(("- ", "* ")):
            blocks.append(("bullet", line[2:].strip()))
            continue
        if re.match(r"^\d+\.\s+", line):
            blocks.append(("bullet", re.sub(r"^\d+\.\s+", "", line)))
            continue
        if line.startswith("### "):
            blocks.append(("h3", line[4:].strip()))
            continue
        if line.startswith("## "):
            blocks.append(("h2", line[3:].strip()))
            continue
        if line.startswith("# "):
            blocks.append(("h1", line[2:].strip()))
            continue
        blocks.append(("p", line))
    return blocks


def build_portfolio_pdf(pack: PortfolioPack) -> bytes:
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_LEFT
        from reportlab.lib.pagesizes import LETTER
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import HRFlowable, PageBreak, Paragraph, SimpleDocTemplate, Spacer
    except ImportError as exc:
        raise RuntimeError("ReportLab is required to build Portfolio Pack PDFs.") from exc

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=LETTER,
        leftMargin=0.8 * inch,
        rightMargin=0.8 * inch,
        topMargin=0.9 * inch,
        bottomMargin=0.75 * inch,
        pageCompression=0,
        title="Portfolio Pack",
        author="Edtronaut AI Co-worker",
    )
    styles = getSampleStyleSheet()
    brand = colors.HexColor("#1f3a5f")
    accent = colors.HexColor("#7d8ca3")
    text = colors.HexColor("#222222")

    styles.add(
        ParagraphStyle(
            name="PortfolioTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=28,
            textColor=brand,
            spaceAfter=14,
            alignment=TA_LEFT,
        )
    )
    styles.add(
        ParagraphStyle(
            name="PortfolioSection",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=20,
            textColor=brand,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="PortfolioSubhead",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            textColor=text,
            spaceBefore=8,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="PortfolioBody",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            textColor=text,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="PortfolioMeta",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            textColor=accent,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="PortfolioNote",
            parent=styles["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=8.5,
            leading=11,
            textColor=accent,
            spaceAfter=4,
        )
    )

    def page_chrome(canvas: Any, document: Any) -> None:
        canvas.saveState()
        canvas.setStrokeColor(accent)
        canvas.setLineWidth(0.5)
        canvas.line(doc.leftMargin, LETTER[1] - 42, LETTER[0] - doc.rightMargin, LETTER[1] - 42)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(accent)
        canvas.drawString(doc.leftMargin, 24, "Edtronaut Portfolio Pack")
        canvas.drawRightString(LETTER[0] - doc.rightMargin, 24, f"Page {document.page}")
        canvas.restoreState()

    story: list[Any] = []
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph(SECTION_TITLES["cover"], styles["PortfolioTitle"]))
    story.append(
        Paragraph(
            "Employer-readable simulation outputs prepared for review and portfolio use.",
            styles["PortfolioBody"],
        )
    )
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(f"Session ID: {pack.thread_id}", styles["PortfolioMeta"]))
    story.append(Paragraph(f"Exported: {pack.exported_at}", styles["PortfolioMeta"]))
    story.append(Paragraph(pack.draft_note, styles["PortfolioNote"]))
    story.append(Spacer(1, 0.2 * inch))
    story.append(HRFlowable(width="100%", thickness=1, color=accent))

    def append_artifact_section(section_title: str, artifact: PortfolioArtifact) -> None:
        story.append(PageBreak())
        story.append(Paragraph(section_title, styles["PortfolioSection"]))
        story.append(Paragraph(artifact.title, styles["PortfolioSubhead"]))
        story.append(
            Paragraph(
                f"Status: {artifact.status_label} | Created: {artifact.created_at}",
                styles["PortfolioMeta"],
            )
        )
        if artifact.metadata:
            detail = " | ".join(f"{key.replace('_', ' ').title()}: {value}" for key, value in artifact.metadata.items())
            story.append(Paragraph(detail, styles["PortfolioMeta"]))
        story.append(Paragraph(pack.draft_note, styles["PortfolioNote"]))
        story.append(Spacer(1, 0.05 * inch))
        for block_type, block_text in _markdown_to_blocks(artifact.body_markdown):
            if block_type == "spacer":
                story.append(Spacer(1, 0.07 * inch))
            elif block_type == "h1":
                story.append(Paragraph(block_text, styles["PortfolioSection"]))
            elif block_type == "h2":
                story.append(Paragraph(block_text, styles["PortfolioSubhead"]))
            elif block_type == "h3":
                story.append(Paragraph(block_text, styles["PortfolioSubhead"]))
            elif block_type == "bullet":
                story.append(Paragraph(f"• {block_text}", styles["PortfolioBody"]))
            else:
                story.append(Paragraph(block_text, styles["PortfolioBody"]))
        if artifact.source_notes:
            story.append(Spacer(1, 0.05 * inch))
            story.append(Paragraph("Source Notes", styles["PortfolioSubhead"]))
            for note in artifact.source_notes:
                story.append(Paragraph(f"• {note}", styles["PortfolioBody"]))

    append_artifact_section(SECTION_TITLES["final_plan"], pack.final_plan)
    story.append(PageBreak())
    story.append(Paragraph(SECTION_TITLES["internal_comm"], styles["PortfolioSection"]))
    story.append(Paragraph(pack.draft_note, styles["PortfolioNote"]))
    for artifact in pack.internal_comms:
        story.append(Paragraph(artifact.title, styles["PortfolioSubhead"]))
        story.append(
            Paragraph(
                f"Type: {artifact.metadata.get('comm_type', 'post').title()} | "
                f"Status: {artifact.status_label} | Created: {artifact.created_at}",
                styles["PortfolioMeta"],
            )
        )
        if artifact.metadata.get("audience"):
            story.append(Paragraph(f"Audience: {artifact.metadata['audience']}", styles["PortfolioMeta"]))
        for block_type, block_text in _markdown_to_blocks(artifact.body_markdown):
            if block_type == "spacer":
                story.append(Spacer(1, 0.06 * inch))
            elif block_type == "bullet":
                story.append(Paragraph(f"• {block_text}", styles["PortfolioBody"]))
            elif block_type in {"h1", "h2", "h3"}:
                story.append(Paragraph(block_text, styles["PortfolioSubhead"]))
            else:
                story.append(Paragraph(block_text, styles["PortfolioBody"]))
        if artifact.source_notes:
            story.append(Paragraph("Source Notes: " + "; ".join(artifact.source_notes), styles["PortfolioNote"]))
        story.append(Spacer(1, 0.12 * inch))

    append_artifact_section(SECTION_TITLES["executive_update"], pack.executive_update)
    story.append(PageBreak())
    story.append(Paragraph(SECTION_TITLES["sources"], styles["PortfolioSection"]))
    story.append(Paragraph("Each section below records the sources or notes attached during export.", styles["PortfolioBody"]))
    for artifact in [pack.final_plan, *pack.internal_comms, pack.executive_update]:
        story.append(Paragraph(artifact.title, styles["PortfolioSubhead"]))
        if artifact.source_notes:
            for note in artifact.source_notes:
                story.append(Paragraph(f"• {note}", styles["PortfolioBody"]))
        else:
            story.append(Paragraph("• No source notes captured.", styles["PortfolioBody"]))

    doc.build(story, onFirstPage=page_chrome, onLaterPages=page_chrome)
    return buffer.getvalue()


def export_portfolio_pack(
    session_state: MutableMapping[str, Any],
    *,
    thread_id: str,
    export_root: str | Path,
    created_at: datetime | None = None,
) -> ExportedPortfolioPack:
    pack = _build_pack(session_state, thread_id=thread_id, created_at=created_at)
    pdf_bytes = build_portfolio_pdf(pack)
    export_root_path = Path(export_root)
    file_name = _filename(thread_id=thread_id, created_at=created_at)
    target_dir = export_root_path / thread_id
    target_dir.mkdir(parents=True, exist_ok=True)
    file_path = target_dir / file_name
    file_path.write_bytes(pdf_bytes)

    exported = ExportedPortfolioPack(
        file_name=file_name,
        file_path=file_path,
        pdf_bytes=pdf_bytes,
        pack=pack,
    )
    session_state[PORTFOLIO_EXPORT_SESSION_KEY] = {
        "file_name": file_name,
        "file_path": str(file_path),
        "pdf_bytes": pdf_bytes,
        "exported_at": pack.exported_at,
    }
    return exported
