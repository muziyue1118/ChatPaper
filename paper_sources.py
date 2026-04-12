import datetime
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import arxiv
except ImportError:
    arxiv = None
import requests


IEEE_SEARCH_URL = "https://ieeexploreapi.ieee.org/api/v1/search/articles"
ELSEVIER_SCOPUS_URL = "https://api.elsevier.com/content/search/scopus"
ELSEVIER_SCIENCEDIRECT_URL = "https://api.elsevier.com/content/search/sciencedirect"


@dataclass
class SearchCandidate:
    source: str
    provider_id: str
    title: str
    authors: List[str] = field(default_factory=list)
    abstract: str = ""
    published: str = ""
    doi: str = ""
    landing_url: str = ""
    pdf_url: str = ""
    oa_status: str = "unknown"
    downloadable: bool = False
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResponse:
    source: str
    candidates: List[SearchCandidate] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def sanitize_filename(text: str, default: str = "paper") -> str:
    cleaned = re.sub(r"[\/\\:\*\?\"<>\|]+", "_", (text or "").strip())
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ._")
    return cleaned or default


def normalize_title_for_dedupe(title: str) -> str:
    text = (title or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def match_filter_keys(candidate: SearchCandidate, filter_keys: str) -> bool:
    normalized_keys = [key.strip().lower() for key in (filter_keys or "").split() if key.strip()]
    if not normalized_keys:
        return True
    haystack = f"{candidate.title} {candidate.abstract}".lower()
    return all(key in haystack for key in normalized_keys)


def filter_candidates(
    candidates: Iterable[SearchCandidate],
    filter_keys: str,
    max_results: int,
    days_from: int = 0,
    days_to: Optional[int] = None,
    page_num: int = 1,
) -> List[SearchCandidate]:
    normalized_days_from, normalized_days_to = normalize_day_window(days_from, days_to)
    matched = [
        candidate
        for candidate in candidates
        if match_filter_keys(candidate, filter_keys)
        and match_day_window(candidate, normalized_days_from, normalized_days_to)
    ]
    return paginate_candidates(matched, max_results=max_results, page_num=page_num)


def paginate_candidates(candidates: Iterable[SearchCandidate], max_results: int, page_num: int = 1) -> List[SearchCandidate]:
    page_size = max(1, int(max_results or 1))
    normalized_page_num = normalize_page_num(page_num)
    start = (normalized_page_num - 1) * page_size
    end = start + page_size
    candidate_list = list(candidates)
    return candidate_list[start:end]


def normalize_page_num(page_num: int = 1) -> int:
    return max(1, int(page_num or 1))


def calculate_prefetch_limit(max_results: int, page_num: int = 1, oversample_factor: int = 3) -> int:
    page_size = max(1, int(max_results or 1))
    normalized_page_num = normalize_page_num(page_num)
    normalized_oversample = max(1, int(oversample_factor or 1))
    return page_size * normalized_page_num * normalized_oversample


def normalize_day_window(days_from: int = 0, days_to: Optional[int] = None) -> Tuple[int, Optional[int]]:
    normalized_from = max(0, int(days_from or 0))
    normalized_to = None if days_to is None else max(0, int(days_to))
    if normalized_to is not None and normalized_from > normalized_to:
        normalized_from, normalized_to = normalized_to, normalized_from
    return normalized_from, normalized_to


def match_day_window(candidate: SearchCandidate, days_from: int = 0, days_to: Optional[int] = None) -> bool:
    if days_to is None and days_from <= 0:
        return True
    published_date = parse_published_date(candidate.published)
    if published_date is None:
        return False
    age_days = (datetime.date.today() - published_date).days
    if age_days < 0:
        return False
    if age_days < days_from:
        return False
    if days_to is not None and age_days > days_to:
        return False
    return True


def parse_published_date(value: Any) -> Optional[datetime.date]:
    if isinstance(value, datetime.datetime):
        return value.date()
    if isinstance(value, datetime.date):
        return value
    if value in (None, ""):
        return None
    text = str(value).strip()
    if not text:
        return None
    normalized = text.replace("/", "-")
    patterns = (
        "%Y-%m-%d",
        "%Y-%m",
        "%Y",
        "%d %B %Y",
        "%B %d, %Y",
        "%d %b %Y",
        "%b %d, %Y",
    )
    for pattern in patterns:
        try:
            parsed = datetime.datetime.strptime(normalized, pattern)
            return parsed.date()
        except ValueError:
            continue
    return None


def dedupe_candidates(candidates: Iterable[SearchCandidate]) -> List[SearchCandidate]:
    source_priority = {
        "sciencedirect": 0,
        "elsevier": 1,
        "scopus": 2,
        "ieee": 3,
        "arxiv": 4,
    }
    merged: Dict[str, SearchCandidate] = {}
    for candidate in candidates:
        key = f"doi:{candidate.doi.lower()}" if candidate.doi else f"title:{normalize_title_for_dedupe(candidate.title)}"
        if not key:
            continue
        if key not in merged:
            merged[key] = candidate
            continue
        current = merged[key]
        preferred, fallback = current, candidate
        if source_priority.get(candidate.source, 99) < source_priority.get(current.source, 99):
            preferred, fallback = candidate, current
        merged[key] = merge_candidates(preferred, fallback)
    return list(merged.values())


def merge_candidates(primary: SearchCandidate, secondary: SearchCandidate) -> SearchCandidate:
    merged = SearchCandidate(**primary.__dict__)
    for attr in ("provider_id", "title", "abstract", "published", "doi", "landing_url", "pdf_url"):
        if not getattr(merged, attr) and getattr(secondary, attr):
            setattr(merged, attr, getattr(secondary, attr))
    if not merged.authors and secondary.authors:
        merged.authors = list(secondary.authors)
    if merged.oa_status == "unknown" and secondary.oa_status != "unknown":
        merged.oa_status = secondary.oa_status
    merged.downloadable = merged.downloadable or secondary.downloadable
    raw = dict(secondary.raw)
    raw.update(merged.raw)
    merged.raw = raw
    return merged


def partition_candidates(candidates: Iterable[SearchCandidate], download_policy: str) -> Tuple[List[SearchCandidate], List[SearchCandidate]]:
    if download_policy == "metadata_only":
        return [], list(candidates)
    downloadable = [candidate for candidate in candidates if candidate.downloadable]
    reading_list = [candidate for candidate in candidates if not candidate.downloadable]
    return downloadable, reading_list


def render_reading_list_markdown(
    candidates: Iterable[SearchCandidate],
    errors: Optional[Iterable[str]] = None,
    heading: str = "Reading List",
) -> str:
    lines = [f"# {heading}", ""]
    error_list = [error for error in (errors or []) if error]
    if error_list:
        lines.append("## Provider Errors")
        lines.append("")
        for error in error_list:
            lines.append(f"- {error}")
        lines.append("")

    candidates = list(candidates)
    if not candidates:
        lines.append("No metadata-only papers were generated in this run.")
        lines.append("")
        return "\n".join(lines)

    for index, candidate in enumerate(candidates, start=1):
        lines.append(f"## {index}. {candidate.title or 'Untitled'}")
        lines.append("")
        lines.append(f"- Source: {candidate.source}")
        lines.append(f"- Authors: {', '.join(candidate.authors) if candidate.authors else 'None'}")
        lines.append(f"- Published: {candidate.published or 'Unknown'}")
        lines.append(f"- DOI: {candidate.doi or 'None'}")
        lines.append(f"- Landing URL: {candidate.landing_url or 'None'}")
        lines.append(f"- PDF URL: {candidate.pdf_url or 'None'}")
        lines.append(f"- OA Status: {candidate.oa_status}")
        lines.append(f"- Suggested Filename: {suggested_pdf_name(candidate)}")
        lines.append("")
        lines.append("### Abstract")
        lines.append("")
        lines.append(candidate.abstract or "No abstract available.")
        lines.append("")
    return "\n".join(lines)


def suggested_pdf_name(candidate: SearchCandidate) -> str:
    stem = sanitize_filename(candidate.title or candidate.provider_id or "paper")
    return stem[:120] + ".pdf"


def search_papers(
    source: str,
    query: str,
    max_results: int,
    filter_keys: str = "",
    days_from: int = 0,
    days_to: Optional[int] = None,
    page_num: int = 1,
    sort: Optional[Any] = None,
    ieee_api_key: str = "",
    elsevier_api_key: str = "",
    timeout: int = 20,
) -> SearchResponse:
    source = (source or "arxiv").lower()
    normalized_page_num = normalize_page_num(page_num)
    prefetch_limit = calculate_prefetch_limit(max_results, normalized_page_num)
    if source == "arxiv":
        if arxiv is None:
            response = SearchResponse(source="arxiv", errors=["The 'arxiv' package is not installed. Install requirements.txt to enable arXiv search."])
        else:
            response = SearchResponse(source="arxiv", candidates=search_arxiv(query, prefetch_limit, sort=sort))
    elif source == "ieee":
        response = search_ieee(query, prefetch_limit, ieee_api_key=ieee_api_key, timeout=timeout)
    elif source == "scopus":
        response = search_scopus(query, prefetch_limit, elsevier_api_key=elsevier_api_key, timeout=timeout)
    elif source == "sciencedirect":
        response = search_sciencedirect(query, prefetch_limit, elsevier_api_key=elsevier_api_key, timeout=timeout)
    elif source == "elsevier":
        response = search_elsevier(query, prefetch_limit, elsevier_api_key=elsevier_api_key, timeout=timeout)
    else:
        response = SearchResponse(source=source, errors=[f"Unsupported source: {source}"])
    response.candidates = filter_candidates(
        response.candidates,
        filter_keys,
        max_results,
        days_from=days_from,
        days_to=days_to,
        page_num=normalized_page_num,
    )
    return response


def search_arxiv(query: str, max_results: int, sort: Optional[Any] = None) -> List[SearchCandidate]:
    if arxiv is None:
        raise RuntimeError("The 'arxiv' package is not installed.")
    search = arxiv.Search(
        query=query,
        max_results=max(max_results * 3, max_results),
        sort_by=sort or arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending,
    )
    candidates: List[SearchCandidate] = []
    for result in search.results():
        published = ""
        if getattr(result, "updated", None):
            published = result.updated.date().isoformat()
        pdf_url = getattr(result, "pdf_url", "") or ""
        candidates.append(
            SearchCandidate(
                source="arxiv",
                provider_id=result.entry_id,
                title=result.title,
                authors=[str(author) for author in getattr(result, "authors", [])],
                abstract=_clean_text(result.summary),
                published=published,
                doi=getattr(result, "doi", "") or "",
                landing_url=result.entry_id,
                pdf_url=pdf_url,
                oa_status="open",
                downloadable=bool(pdf_url),
                raw={"entry_id": result.entry_id},
            )
        )
    return candidates


def search_ieee(query: str, max_results: int, ieee_api_key: str, timeout: int = 20) -> SearchResponse:
    if not ieee_api_key:
        return SearchResponse(source="ieee", errors=["IEEE API key is missing. Set [IEEE] API_KEY or IEEE_API_KEY."])
    fetch_limit = max(1, int(max_results or 1))
    candidates: List[SearchCandidate] = []
    errors: List[str] = []
    start_record = 1
    while len(candidates) < fetch_limit:
        batch_size = min(200, fetch_limit - len(candidates))
        params = {
            "apikey": ieee_api_key,
            "format": "json",
            "querytext": query,
            "max_records": batch_size,
            "start_record": start_record,
        }
        try:
            response = requests.get(IEEE_SEARCH_URL, params=params, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
        except requests.HTTPError as exc:
            errors.append(_format_http_error("IEEE", exc))
            break
        except requests.RequestException as exc:
            errors.append(f"IEEE request failed: {exc}")
            break
        except ValueError as exc:
            errors.append(f"IEEE response parsing failed: {exc}")
            break

        articles = payload.get("articles", [])
        if not articles:
            break
        for article in articles:
            pdf_url = article.get("pdf_url") or article.get("access_pdf_url") or ""
            landing_url = article.get("html_url") or article.get("document_url") or article.get("article_url") or ""
            oa_status = _normalize_oa_status(
                article.get("open_access")
                or article.get("open_access_flag")
                or article.get("is_open_access")
                or article.get("access_type")
            )
            candidates.append(
                SearchCandidate(
                    source="ieee",
                    provider_id=str(article.get("article_number") or article.get("doi") or article.get("title") or ""),
                    title=article.get("title", ""),
                    authors=_extract_ieee_authors(article),
                    abstract=_clean_text(article.get("abstract", "")),
                    published=_format_date(article.get("publication_date") or article.get("publication_year")),
                    doi=article.get("doi", "") or "",
                    landing_url=landing_url,
                    pdf_url=pdf_url,
                    oa_status=oa_status,
                    downloadable=oa_status == "open" and bool(pdf_url),
                    raw=article,
                )
            )
        start_record += len(articles)
        if len(articles) < batch_size:
            break
    return SearchResponse(source="ieee", candidates=candidates[:fetch_limit], errors=errors)


def search_scopus(query: str, max_results: int, elsevier_api_key: str, timeout: int = 20) -> SearchResponse:
    if not elsevier_api_key:
        return SearchResponse(source="scopus", errors=["Elsevier API key is missing. Set [Elsevier] API_KEY or ELSEVIER_API_KEY."])
    fetch_limit = max(1, int(max_results or 1))
    headers = {
        "X-ELS-APIKey": elsevier_api_key,
        "Accept": "application/json",
    }
    candidates: List[SearchCandidate] = []
    errors: List[str] = []
    start = 0
    while len(candidates) < fetch_limit:
        batch_size = min(200, fetch_limit - len(candidates))
        params = {
            "query": query,
            "count": batch_size,
            "start": start,
            "view": "STANDARD",
        }
        try:
            response = requests.get(ELSEVIER_SCOPUS_URL, headers=headers, params=params, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
        except requests.HTTPError as exc:
            errors.append(_format_http_error("Scopus", exc))
            break
        except requests.RequestException as exc:
            errors.append(f"Scopus request failed: {exc}")
            break
        except ValueError as exc:
            errors.append(f"Scopus response parsing failed: {exc}")
            break

        entries = payload.get("search-results", {}).get("entry", [])
        if not entries:
            break
        candidates.extend(_scopus_entry_to_candidate(entry) for entry in entries)
        start += len(entries)
        if len(entries) < batch_size:
            break
    return SearchResponse(source="scopus", candidates=candidates[:fetch_limit], errors=errors)


def search_sciencedirect(query: str, max_results: int, elsevier_api_key: str, timeout: int = 20) -> SearchResponse:
    if not elsevier_api_key:
        return SearchResponse(source="sciencedirect", errors=["Elsevier API key is missing. Set [Elsevier] API_KEY or ELSEVIER_API_KEY."])
    fetch_limit = max(1, int(max_results or 1))
    headers = {
        "X-ELS-APIKey": elsevier_api_key,
        "Accept": "application/json",
    }
    candidates: List[SearchCandidate] = []
    errors: List[str] = []
    start = 0
    while len(candidates) < fetch_limit:
        batch_size = min(200, fetch_limit - len(candidates))
        params = {
            "query": query,
            "count": batch_size,
            "start": start,
        }
        try:
            response = requests.get(ELSEVIER_SCIENCEDIRECT_URL, headers=headers, params=params, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
        except requests.HTTPError as exc:
            errors.append(_format_http_error("ScienceDirect", exc))
            break
        except requests.RequestException as exc:
            errors.append(f"ScienceDirect request failed: {exc}")
            break
        except ValueError as exc:
            errors.append(f"ScienceDirect response parsing failed: {exc}")
            break

        entries = _extract_sciencedirect_entries(payload)
        if not entries:
            break
        candidates.extend(_sciencedirect_entry_to_candidate(entry) for entry in entries)
        start += len(entries)
        if len(entries) < batch_size:
            break
    return SearchResponse(source="sciencedirect", candidates=candidates[:fetch_limit], errors=errors)


def search_elsevier(query: str, max_results: int, elsevier_api_key: str, timeout: int = 20) -> SearchResponse:
    scidir = search_sciencedirect(query, max_results, elsevier_api_key=elsevier_api_key, timeout=timeout)
    scopus = search_scopus(query, max_results, elsevier_api_key=elsevier_api_key, timeout=timeout)
    combined = dedupe_candidates(scidir.candidates + scopus.candidates)
    return SearchResponse(
        source="elsevier",
        candidates=combined,
        errors=scidir.errors + scopus.errors,
    )


def download_candidate_pdf(candidate: SearchCandidate, output_dir: str, timeout: int = 30) -> str:
    if not candidate.pdf_url:
        raise ValueError("Candidate does not have a downloadable PDF URL.")
    response = requests.get(candidate.pdf_url, timeout=timeout, allow_redirects=True)
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "").lower()
    if "pdf" not in content_type and not candidate.pdf_url.lower().endswith(".pdf"):
        raise ValueError(f"Expected a PDF response, got Content-Type={content_type or 'unknown'}.")
    file_name = suggested_pdf_name(candidate)
    file_path = os.path.join(output_dir, file_name)
    with open(file_path, "wb") as output:
        output.write(response.content)
    return file_path


def _scopus_entry_to_candidate(entry: Dict[str, Any]) -> SearchCandidate:
    landing_url = _extract_elsevier_link(entry, ("scopus", "self", "record", "abstract"))
    pdf_url = _extract_elsevier_link(entry, ("pdf", "full-text-pdf", "full-text"))
    oa_status = _normalize_oa_status(entry.get("openaccess") or entry.get("openaccessFlag"))
    return SearchCandidate(
        source="scopus",
        provider_id=str(entry.get("dc:identifier") or entry.get("eid") or entry.get("prism:doi") or ""),
        title=entry.get("dc:title", "") or "",
        authors=_extract_scopus_authors(entry),
        abstract=_clean_text(entry.get("dc:description", "") or entry.get("description", "")),
        published=_format_date(entry.get("prism:coverDate") or entry.get("prism:coverDisplayDate")),
        doi=entry.get("prism:doi", "") or "",
        landing_url=landing_url,
        pdf_url=pdf_url,
        oa_status=oa_status,
        downloadable=oa_status == "open" and bool(pdf_url),
        raw=entry,
    )


def _sciencedirect_entry_to_candidate(entry: Dict[str, Any]) -> SearchCandidate:
    title = _first_non_empty(
        entry.get("title"),
        entry.get("dc:title"),
        entry.get("articleTitle"),
    )
    abstract = _first_non_empty(
        entry.get("abstract"),
        entry.get("dc:description"),
        entry.get("description"),
        entry.get("summary"),
    )
    landing_url = _first_non_empty(
        entry.get("link"),
        entry.get("url"),
        entry.get("uri"),
        _extract_elsevier_link(entry, ("sciencedirect", "self", "record", "abstract", "full-text")),
    )
    pdf_url = _first_non_empty(
        entry.get("pdfUrl"),
        entry.get("pdf_url"),
        entry.get("downloadLink"),
        _extract_elsevier_link(entry, ("pdf", "full-text-pdf")),
    )
    oa_status = _normalize_oa_status(
        entry.get("openAccess")
        or entry.get("openaccess")
        or entry.get("openaccessArticle")
        or entry.get("openaccessFlag")
    )
    return SearchCandidate(
        source="sciencedirect",
        provider_id=str(_first_non_empty(entry.get("pii"), entry.get("doi"), title)),
        title=title or "",
        authors=_extract_generic_authors(entry),
        abstract=_clean_text(abstract or ""),
        published=_format_date(_first_non_empty(entry.get("publicationDate"), entry.get("prism:coverDate"), entry.get("coverDate"))),
        doi=_first_non_empty(entry.get("doi"), entry.get("prism:doi")) or "",
        landing_url=landing_url or "",
        pdf_url=pdf_url or "",
        oa_status=oa_status,
        downloadable=oa_status == "open" and bool(pdf_url),
        raw=entry,
    )


def _extract_ieee_authors(article: Dict[str, Any]) -> List[str]:
    author_block = article.get("authors") or {}
    authors = author_block.get("authors") if isinstance(author_block, dict) else author_block
    if not isinstance(authors, list):
        return []
    names = []
    for author in authors:
        if isinstance(author, dict):
            full_name = author.get("full_name") or author.get("name")
            if full_name:
                names.append(full_name)
        elif author:
            names.append(str(author))
    return names


def _extract_scopus_authors(entry: Dict[str, Any]) -> List[str]:
    authors = entry.get("author")
    if isinstance(authors, list):
        result = []
        for author in authors:
            if isinstance(author, dict):
                name = author.get("authname") or author.get("ce:indexed-name") or author.get("preferred-name", {}).get("ce:indexed-name")
                if name:
                    result.append(name)
        if result:
            return result
    creator = entry.get("dc:creator")
    return [creator] if creator else []


def _extract_generic_authors(entry: Dict[str, Any]) -> List[str]:
    authors = []
    for key in ("authors", "author", "creator"):
        value = entry.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    name = _first_non_empty(item.get("name"), item.get("full-name"), item.get("given-name"), item.get("surname"))
                    if name:
                        authors.append(name)
                elif item:
                    authors.append(str(item))
        elif isinstance(value, dict):
            name = _first_non_empty(value.get("name"), value.get("full-name"))
            if name:
                authors.append(name)
        elif isinstance(value, str) and value.strip():
            authors.append(value.strip())
    return authors


def _extract_sciencedirect_entries(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    if isinstance(payload.get("results"), list):
        return payload["results"]
    search_results = payload.get("search-results")
    if isinstance(search_results, dict):
        if isinstance(search_results.get("entry"), list):
            return search_results["entry"]
        if isinstance(search_results.get("results"), list):
            return search_results["results"]
    for value in payload.values():
        if isinstance(value, list) and value and isinstance(value[0], dict):
            return value
    return []


def _extract_elsevier_link(entry: Dict[str, Any], preferred_refs: Iterable[str]) -> str:
    links = entry.get("link") or entry.get("links") or []
    if isinstance(links, str):
        return links
    if not isinstance(links, list):
        return ""
    preferred_refs = tuple(ref.lower() for ref in preferred_refs)
    for link in links:
        if not isinstance(link, dict):
            continue
        ref = str(link.get("@ref") or link.get("ref") or "").lower()
        href = link.get("@href") or link.get("href") or link.get("$") or ""
        if ref in preferred_refs and href:
            return href
    for link in links:
        if isinstance(link, dict):
            href = link.get("@href") or link.get("href") or link.get("$") or ""
            if href:
                return href
    return ""


def _format_http_error(provider: str, exc: requests.HTTPError) -> str:
    status_code = getattr(exc.response, "status_code", "unknown")
    return f"{provider} request failed with HTTP {status_code}: {exc}"


def _clean_text(text: str) -> str:
    return (text or "").replace("-\n", "-").replace("\n", " ").strip()


def _format_date(value: Any) -> str:
    if isinstance(value, datetime.date):
        return value.isoformat()
    if value is None:
        return ""
    return str(value)


def _first_non_empty(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
        if value not in (None, "", []):
            return str(value)
    return ""


def _normalize_oa_status(value: Any) -> str:
    if isinstance(value, bool):
        return "open" if value else "closed"
    if value is None:
        return "unknown"
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "open", "open access", "oa"}:
        return "open"
    if normalized in {"0", "false", "no", "n", "closed", "subscription", "restricted"}:
        return "closed"
    return "unknown"
