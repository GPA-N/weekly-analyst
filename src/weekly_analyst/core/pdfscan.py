"""Static hazard scan of a PDF before it is sent. Opens read-only, never renders or fetches.

Empty list means clean. A file that cannot be parsed is a finding, never clean.
"""

import getpass
import re
import socket
from pathlib import Path

import pikepdf

METADATA_KEYS = ("/Author", "/Creator", "/Producer", "/Title", "/Subject", "/Keywords")
REMOTE_RE = re.compile(r"^\s*(https?|ftp)://", re.IGNORECASE)
PATH_RE = re.compile(
    r"(?<![A-Za-z])[A-Za-z]:[\\/]"  # Windows drive path, C:\ or C:/
    r"|\\\\[^\\\s]+\\"  # UNC path, host in first segment
    r"|(?<![\w.:/])/[\w.\-]+/[\w.\-]+"  # Unix absolute path with two segments
)
MIN_IDENTITY_LEN = 3


def scan_pdf(path: Path) -> list[str]:
    try:
        pdf = pikepdf.open(path)
    except (pikepdf.PdfError, pikepdf.PasswordError):
        return ["unreadable PDF: cannot parse, so not verified clean"]
    with pdf:
        dicts: list[pikepdf.Dictionary] = []
        _collect_dicts(pdf.trailer, dicts)
        for obj in pdf.objects:
            _collect_dicts(obj, dicts)

        found: set[str] = set()
        for d in dicts:
            _check_dict(d, found)

        identities = _local_identities()
        for key in METADATA_KEYS:
            if key in pdf.docinfo:
                text = str(pdf.docinfo[key])
                if PATH_RE.search(text) or _mentions(text, identities):
                    found.add(f"metadata {key} holds a path, user name or host name")
    return sorted(found)


def _collect_dicts(node, out: list) -> None:
    """Collect dictionaries reachable without crossing an indirect reference (those are visited on their own)."""
    if isinstance(node, pikepdf.Stream):
        out.append(node.stream_dict)
        children = node.stream_dict.values()
    elif isinstance(node, pikepdf.Dictionary):
        out.append(node)
        children = node.values()
    elif isinstance(node, pikepdf.Array):
        children = node
    else:
        return
    for child in children:
        if isinstance(child, (pikepdf.Dictionary, pikepdf.Array)) and not child.is_indirect:
            _collect_dicts(child, out)


def _check_dict(d, found: set[str]) -> None:
    kind = str(d.get("/S", ""))
    if kind == "/URI":
        found.add("URI action")
    if kind == "/Launch":
        found.add("Launch action")
    if kind == "/JavaScript" or "/JS" in d or "/JavaScript" in d:
        found.add("JavaScript")
    if "/EmbeddedFiles" in d or "/EF" in d or str(d.get("/Type", "")) == "/EmbeddedFile":
        found.add("embedded file")
    is_xobject = str(d.get("/Type", "")) == "/XObject" or str(d.get("/Subtype", "")) in ("/Image", "/Form")
    if is_xobject and _points_remote(d):
        found.add("remote image or XObject reference (http, https or ftp)")


def _points_remote(d) -> bool:
    for key in ("/F", "/Ref"):
        value = d.get(key)
        if isinstance(value, pikepdf.String):
            texts = [str(value)]
        elif isinstance(value, pikepdf.Dictionary):
            texts = [str(v) for v in value.values() if isinstance(v, pikepdf.String)]
        else:
            texts = []
        if any(REMOTE_RE.match(t) for t in texts):
            return True
    return False


def _local_identities() -> list[str]:
    names = (getpass.getuser(), socket.gethostname())
    return [n for n in names if len(n) >= MIN_IDENTITY_LEN]


def _mentions(text: str, identities: list[str]) -> bool:
    return any(re.search(rf"(?<!\w){re.escape(n)}(?!\w)", text, re.IGNORECASE) for n in identities)
