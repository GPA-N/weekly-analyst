import getpass
import socket

import pikepdf
import pytest
from pikepdf import Dictionary, Name, String

from weekly_analyst.core.pdfscan import scan_pdf


def _clean_pdf():
    pdf = pikepdf.new()
    pdf.add_blank_page(page_size=(200, 200))
    pdf.docinfo["/Title"] = "Weekly report"
    pdf.docinfo["/Producer"] = "weekly-analyst tests"
    return pdf


def _uri(pdf):
    link = Dictionary(
        Type=Name.Annot,
        Subtype=Name.Link,
        Rect=[0, 0, 50, 50],
        A=Dictionary(S=Name.URI, URI=String("https://example.com/")),
    )
    pdf.pages[0].obj.Annots = pikepdf.Array([link])


def _javascript(pdf):
    pdf.Root.OpenAction = Dictionary(S=Name.JavaScript, JS=String("app.alert(1)"))


def _launch(pdf):
    pdf.Root.OpenAction = Dictionary(S=Name.Launch, F=Dictionary(F=String("cmd.exe")))


def _embedded_file(pdf):
    pdf.attachments["note.txt"] = pikepdf.AttachedFileSpec(pdf, b"hello", filename="note.txt")


def _remote_image(pdf):
    image = pdf.make_stream(
        b"\x00",
        Type=Name.XObject,
        Subtype=Name.Image,
        Width=1,
        Height=1,
        ColorSpace=Name.DeviceGray,
        BitsPerComponent=8,
        F=Dictionary(FS=Name.URL, F=String("https://example.com/logo.png")),
    )
    pdf.pages[0].obj.Resources = Dictionary(XObject=Dictionary(Im1=image))


def _metadata_path(pdf):
    pdf.docinfo["/Author"] = r"C:\Users\alice\Documents\report.pdf"


HAZARDS = [
    pytest.param(_uri, "uri", id="uri-action"),
    pytest.param(_javascript, "javascript", id="javascript"),
    pytest.param(_launch, "launch", id="launch-action"),
    pytest.param(_embedded_file, "embedded", id="embedded-file"),
    pytest.param(_remote_image, "remote", id="remote-image"),
    pytest.param(_metadata_path, "metadata", id="metadata-path"),
]


@pytest.mark.parametrize("plant, keyword", HAZARDS)
def test_scan_finds_each_planted_object(tmp_path, plant, keyword):
    pdf = _clean_pdf()
    plant(pdf)
    path = tmp_path / "hazard.pdf"
    pdf.save(path)

    findings = scan_pdf(path)

    assert len(findings) == 1
    assert keyword in findings[0].lower()


def test_clean_report_scans_empty(tmp_path):
    path = tmp_path / "clean.pdf"
    _clean_pdf().save(path)

    assert scan_pdf(path) == []


@pytest.mark.parametrize(
    "value",
    [
        r"C:\Users\alice\Documents\report.pdf",
        "/home/alice/reports/q3.pdf",
        r"\\FILESRV01\finance\q3.pdf",
    ],
)
def test_metadata_path_forms_are_flagged(tmp_path, value):
    pdf = _clean_pdf()
    pdf.docinfo["/Subject"] = value
    path = tmp_path / "paths.pdf"
    pdf.save(path)

    findings = scan_pdf(path)

    assert len(findings) == 1
    assert "metadata" in findings[0]


@pytest.mark.parametrize("name", [socket.gethostname(), getpass.getuser()])
def test_metadata_with_local_host_or_user_name_is_flagged(tmp_path, name):
    pdf = _clean_pdf()
    pdf.docinfo["/Creator"] = f"Exported from {name}"
    path = tmp_path / "identity.pdf"
    pdf.save(path)

    findings = scan_pdf(path)

    assert len(findings) == 1
    assert "metadata" in findings[0]


def test_unreadable_file_is_a_finding_not_clean(tmp_path):
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"this is not a pdf")

    findings = scan_pdf(path)

    assert len(findings) == 1
    assert "unreadable" in findings[0]
