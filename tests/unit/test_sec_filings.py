from __future__ import annotations

from src.tools.equity import sec_filings


SEC_INDEX_HTML = """
<table class="tableFile" summary="Document Format Files">
<tr><th>Seq</th><th>Description</th><th>Document</th><th>Type</th><th>Size</th></tr>
<tr>
  <td>1</td>
  <td>FORM 10-K</td>
  <td><a href="/ix?doc=/Archives/edgar/data/34088/000003408826000045/xom-20251231.htm">xom-20251231.htm iXBRL</a></td>
  <td>10-K</td>
  <td>5591068</td>
</tr>
<tr>
  <td>2</td>
  <td>EXHIBIT</td>
  <td><a href="/Archives/edgar/data/34088/000003408826000045/exhibit.htm">exhibit.htm</a></td>
  <td>EX-10</td>
  <td>123</td>
</tr>
</table>
"""


def test_resolve_sec_primary_document_url_handles_ixbrl_index(monkeypatch):
    captured = {}

    def fake_request_text(url, **kwargs):
        captured["url"] = url
        return SEC_INDEX_HTML

    monkeypatch.setattr(sec_filings, "request_text", fake_request_text)

    resolved = sec_filings.resolve_sec_primary_document_url(
        "https://www.sec.gov/Archives/edgar/data/34088/000003408826000045/0000034088-26-000045-index.htm"
    )

    assert captured["url"].endswith("-index.htm")
    assert resolved == "https://www.sec.gov/Archives/edgar/data/34088/000003408826000045/xom-20251231.htm"


def test_fetch_filing_text_resolves_index_before_fetch(monkeypatch):
    fetched = {}

    monkeypatch.setattr(sec_filings, "resolve_sec_primary_document_url", lambda url: "https://www.sec.gov/primary.htm")

    def fake_fetch_url(url):
        fetched["url"] = url
        return "<html><body><p>Official filing text</p></body></html>"

    monkeypatch.setattr(sec_filings.trafilatura, "fetch_url", fake_fetch_url)
    monkeypatch.setattr(sec_filings.trafilatura, "extract", lambda html, **kwargs: "Official filing text")
    monkeypatch.setattr(sec_filings, "cached_retry_call", lambda _name, _payload, fn, **_kwargs: fn())

    text = sec_filings.fetch_filing_text("https://www.sec.gov/example-index.htm")

    assert fetched["url"] == "https://www.sec.gov/primary.htm"
    assert text == "Official filing text"


def test_fetch_filing_text_falls_back_to_sec_user_agent_request(monkeypatch):
    fetched = {}

    monkeypatch.setattr(sec_filings, "resolve_sec_primary_document_url", lambda url: "https://www.sec.gov/primary.htm")
    monkeypatch.setattr(sec_filings.trafilatura, "fetch_url", lambda url: None)
    monkeypatch.setattr(sec_filings.trafilatura, "extract", lambda html, **kwargs: "FORM 10-K\nOfficial filing text")

    def fake_request_text(url, **kwargs):
        fetched["url"] = url
        fetched["headers"] = kwargs.get("headers")
        return "<html>filing</html>"

    monkeypatch.setattr(sec_filings, "request_text", fake_request_text)
    monkeypatch.setattr(sec_filings, "cached_retry_call", lambda _name, _payload, fn, **_kwargs: fn())

    text = sec_filings.fetch_filing_text("https://www.sec.gov/example-index.htm")

    assert fetched["url"] == "https://www.sec.gov/primary.htm"
    assert fetched["headers"]["User-Agent"].startswith("AlphaSeeker Research")
    assert text == "FORM 10-K\nOfficial filing text"
