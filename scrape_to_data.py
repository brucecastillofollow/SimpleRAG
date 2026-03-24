from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup


def normalize_lines(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


def slugify(value: str, max_len: int = 80) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    if not value:
        value = "document"
    return value[:max_len].strip("-")


def url_to_filename(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc.replace("www.", "")
    path = parsed.path.strip("/")
    raw = f"{host}-{path}" if path else host
    return f"{slugify(raw)}.txt"


def extract_main_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # Keep scripts for JSON-LD extraction below, remove layout/noise tags.
    for tag in soup(["style", "noscript", "header", "footer", "nav", "aside", "form", "svg"]):
        tag.decompose()

    parts: list[str] = []

    # 1) Main page body text.
    root = soup.find("main") or soup.find("article") or soup.find(attrs={"role": "main"}) or soup.body or soup
    body_text = normalize_lines(root.get_text(separator="\n"))
    if body_text:
        parts.append(body_text)

    # 2) Useful metadata tags often present even on JS-heavy pages.
    meta_keys = {
        "description",
        "keywords",
        "author",
        "og:title",
        "og:description",
        "twitter:title",
        "twitter:description",
    }
    head_lines: list[str] = []
    if soup.title and soup.title.get_text(strip=True):
        head_lines.append(f"Title: {soup.title.get_text(strip=True)}")
    for tag in soup.find_all("meta"):
        key = (tag.get("name") or tag.get("property") or "").strip().lower()
        val = (tag.get("content") or "").strip()
        if key in meta_keys and val:
            head_lines.append(f"{key}: {val}")
    if head_lines:
        parts.append("\n".join(dict.fromkeys(head_lines)))

    # 3) JSON-LD (schema.org) often stores business/service text.
    jsonld_lines: list[str] = []
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = (script.string or script.get_text() or "").strip()
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except Exception:
            continue

        def collect(obj: object) -> None:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    # Keep human-readable business fields.
                    if k in {"name", "description", "headline", "slogan", "text"} and isinstance(v, str):
                        txt = v.strip()
                        if txt:
                            jsonld_lines.append(f"{k}: {txt}")
                    else:
                        collect(v)
            elif isinstance(obj, list):
                for item in obj:
                    collect(item)

        collect(data)
    if jsonld_lines:
        parts.append("\n".join(dict.fromkeys(jsonld_lines)))

    # De-duplicate repeated lines across sections.
    merged = "\n\n".join(parts)
    return "\n".join(dict.fromkeys(merged.splitlines()))


def fetch_url(url: str, timeout: int = 20) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36"
        )
    }
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def save_document(output_dir: Path, url: str, text: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / url_to_filename(url)

    # Avoid accidental overwrite if same name appears.
    if file_path.exists():
        stem = file_path.stem
        suffix = file_path.suffix
        idx = 2
        while True:
            candidate = output_dir / f"{stem}-{idx}{suffix}"
            if not candidate.exists():
                file_path = candidate
                break
            idx += 1

    header = f"Source URL: {url}\n\n"
    file_path.write_text(header + text, encoding="utf-8")
    return file_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape public webpages into local text docs for RAG.")
    parser.add_argument(
        "--url",
        action="append",
        default=[],
        help="Single URL to scrape. Repeat the flag for multiple URLs.",
    )
    parser.add_argument(
        "--url-file",
        type=Path,
        help="Path to a text file with one URL per line.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory to write scraped documents.",
    )
    return parser.parse_args()


def gather_urls(args: argparse.Namespace) -> list[str]:
    urls = list(args.url)
    if args.url_file:
        if not args.url_file.exists():
            raise FileNotFoundError(f"URL file not found: {args.url_file}")
        for line in args.url_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            urls.append(line)
    # De-duplicate while preserving order
    unique_urls = []
    seen = set()
    for u in urls:
        if u not in seen:
            unique_urls.append(u)
            seen.add(u)
    return unique_urls


def main() -> None:
    args = parse_args()
    urls = gather_urls(args)
    if not urls:
        print("[info] No URLs provided. Use --url or --url-file.")
        return

    for url in urls:
        try:
            print(f"[info] Fetching {url}")
            html = fetch_url(url)
            text = extract_main_text(html)
            if not text.strip():
                print(f"[warn] No text extracted from: {url}")
                continue
            out_path = save_document(args.output_dir, url, text)
            print(f"[ok] Saved -> {out_path}")
        except Exception as exc:
            print(f"[error] Failed: {url}\n        {exc}")


if __name__ == "__main__":
    main()
