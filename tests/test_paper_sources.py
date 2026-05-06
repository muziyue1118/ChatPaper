import pathlib
import sys
import unittest
from datetime import date, timedelta


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import paper_sources  # noqa: E402
from paper_sources import (  # noqa: E402
    SearchCandidate,
    configure_arxiv_https_endpoint,
    dedupe_candidates,
    filter_candidates,
    normalize_day_window,
    partition_candidates,
    render_reading_list_markdown,
)


class PaperSourcesTest(unittest.TestCase):
    def with_fake_arxiv_client(self, query_url_format):
        class FakeClient:
            pass

        class FakeArxiv:
            Client = FakeClient

        FakeClient.query_url_format = query_url_format
        return FakeArxiv

    def make_candidate(self, **overrides):
        payload = {
            "source": "ieee",
            "provider_id": "id-1",
            "title": "EEG Emotion Recognition with Graph Transformers",
            "authors": ["Alice", "Bob"],
            "abstract": "We study EEG emotion recognition with graph transformers.",
            "published": "2026-01-01",
            "doi": "10.1000/test",
            "landing_url": "https://example.com/paper",
            "pdf_url": "",
            "oa_status": "closed",
            "downloadable": False,
        }
        payload.update(overrides)
        return SearchCandidate(**payload)

    def test_configure_arxiv_https_endpoint_upgrades_http_endpoint(self):
        original_arxiv = paper_sources.arxiv
        fake_arxiv = self.with_fake_arxiv_client("http://export.arxiv.org/api/query?{}")
        try:
            paper_sources.arxiv = fake_arxiv
            configure_arxiv_https_endpoint()
            self.assertEqual(fake_arxiv.Client.query_url_format, "https://export.arxiv.org/api/query?{}")
        finally:
            paper_sources.arxiv = original_arxiv

    def test_configure_arxiv_https_endpoint_keeps_https_endpoint(self):
        original_arxiv = paper_sources.arxiv
        fake_arxiv = self.with_fake_arxiv_client("https://export.arxiv.org/api/query?{}")
        try:
            paper_sources.arxiv = fake_arxiv
            configure_arxiv_https_endpoint()
            self.assertEqual(fake_arxiv.Client.query_url_format, "https://export.arxiv.org/api/query?{}")
        finally:
            paper_sources.arxiv = original_arxiv

    def test_filter_candidates_uses_title_and_abstract(self):
        candidates = [
            self.make_candidate(title="Brain signals", abstract="emotion recognition with eeg"),
            self.make_candidate(provider_id="id-2", title="Vision model", abstract="image classification"),
        ]
        filtered = filter_candidates(candidates, "emotion eeg", max_results=5)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].provider_id, "id-1")

    def test_partition_candidates_respects_download_policy(self):
        open_candidate = self.make_candidate(provider_id="id-open", oa_status="open", downloadable=True, pdf_url="https://example.com/open.pdf")
        closed_candidate = self.make_candidate(provider_id="id-closed")
        downloadable, reading = partition_candidates([open_candidate, closed_candidate], "oa_only")
        self.assertEqual([candidate.provider_id for candidate in downloadable], ["id-open"])
        self.assertEqual([candidate.provider_id for candidate in reading], ["id-closed"])

        downloadable, reading = partition_candidates([open_candidate, closed_candidate], "metadata_only")
        self.assertEqual(downloadable, [])
        self.assertEqual([candidate.provider_id for candidate in reading], ["id-open", "id-closed"])

    def test_filter_candidates_respects_day_window(self):
        recent_candidate = self.make_candidate(provider_id="recent", published=(date.today() - timedelta(days=2)).isoformat())
        older_candidate = self.make_candidate(provider_id="older", published=(date.today() - timedelta(days=20)).isoformat())
        filtered = filter_candidates([recent_candidate, older_candidate], "", max_results=10, days_from=0, days_to=7)
        self.assertEqual([candidate.provider_id for candidate in filtered], ["recent"])

    def test_filter_candidates_paginates_after_filtering(self):
        candidates = [
            self.make_candidate(provider_id="id-1", title="EEG emotion recognition"),
            self.make_candidate(provider_id="id-2", title="EEG emotion decoding"),
            self.make_candidate(provider_id="id-3", title="EEG emotion transfer"),
        ]
        filtered = filter_candidates(candidates, "EEG emotion", max_results=1, page_num=2)
        self.assertEqual([candidate.provider_id for candidate in filtered], ["id-2"])

    def test_normalize_day_window_swaps_inverted_range(self):
        days_from, days_to = normalize_day_window(30, 7)
        self.assertEqual(days_from, 7)
        self.assertEqual(days_to, 30)

    def test_dedupe_candidates_prefers_sciencedirect(self):
        scopus = self.make_candidate(source="scopus", provider_id="scopus-1", doi="10.1000/dup", landing_url="https://scopus.example.com")
        sciencedirect = self.make_candidate(
            source="sciencedirect",
            provider_id="sd-1",
            doi="10.1000/dup",
            landing_url="https://sciencedirect.example.com",
            pdf_url="https://sciencedirect.example.com/open.pdf",
            oa_status="open",
            downloadable=True,
        )
        deduped = dedupe_candidates([scopus, sciencedirect])
        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped[0].source, "sciencedirect")
        self.assertEqual(deduped[0].pdf_url, "https://sciencedirect.example.com/open.pdf")
        self.assertTrue(deduped[0].downloadable)

    def test_render_reading_list_markdown_contains_core_fields(self):
        candidate = self.make_candidate(
            title="Open Access EEG Paper",
            doi="10.1000/open",
            landing_url="https://example.com/open",
            oa_status="unknown",
        )
        markdown = render_reading_list_markdown([candidate], errors=["IEEE request failed with HTTP 429"], heading="Reading List")
        self.assertIn("# Reading List", markdown)
        self.assertIn("Provider Errors", markdown)
        self.assertIn("Open Access EEG Paper", markdown)
        self.assertIn("10.1000/open", markdown)
        self.assertIn("https://example.com/open", markdown)
        self.assertIn("Suggested Filename", markdown)


if __name__ == "__main__":
    unittest.main()
