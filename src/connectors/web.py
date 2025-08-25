# # src/connectors/web.py
import argparse, json, time, re
from collections import deque, defaultdict
from pathlib import Path
from urllib.parse import urljoin, urlparse, unquote
import urllib.robotparser as urobot
import requests
from bs4 import BeautifulSoup
import trafilatura
import yaml

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36; DocPilotRAG/0.1 (+https://example.local)"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}

# Locale path detector: /Top10/fr/... /Top10/pt-BR/... /Top10/zh-TW/...
LOCALE_RE = re.compile(r"^/Top10/[A-Za-z]{2}(?:-[A-Za-z]{2,})?/", re.IGNORECASE)

def load_sources(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def is_core_top10(url: str) -> bool:
    u = urlparse(url)
    return u.netloc.lower() == "owasp.org" and u.path.startswith("/Top10/")

def is_https(url: str) -> bool:
    return urlparse(url).scheme.lower() == "https"

def canonicalize(url: str) -> str:
    """Normalize without breaking file URLs (no trailing slash on .pdf/.html)."""
    u = urlparse(url)
    path = unquote(u.path)
    path = re.sub(r"/index\.html?$", "/", path, flags=re.IGNORECASE)
    path = re.sub(r"/{2,}", "/", path)
    if not path:
        path = "/"
    # only append slash for directory-like paths (not files with extensions)
    if not path.endswith("/") and not re.search(r"\.[A-Za-z0-9]{1,6}$", path):
        path += "/"
    return f"{u.scheme}://{u.netloc}{path}{('?' + u.query) if u.query else ''}"

def clean_url(base, href):
    if not href:
        return None
    href = href.strip()
    if href.startswith(("#", "mailto:", "javascript:")):
        return None
    return canonicalize(urljoin(base, href))

def extract_readable(html, url):
    try:
        extracted = trafilatura.extract(html, include_comments=False, include_tables=False, url=url)
        if extracted and len(extracted.split()) > 50:
            return extracted
    except Exception:
        pass
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["nav", "aside", "footer", "script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text("\n")
    return re.sub(r"\n{3,}", "\n\n", text)

def collect_references_links(soup: BeautifulSoup, base_url: str, section_regex: re.Pattern) -> set[str]:
    """
    Return absolute, canonicalized hrefs that occur inside the 'References' section(s).
    We:
      - find any <h1..h6> whose text OR id matches section_regex (e.g., 'references'),
      - walk forward until the next heading of <= the same level,
      - collect all <a href> within that slice.
    This avoids relying on fragile classes and works across all A0X pages.
    """
    refs = set()
    # find matching headings by text OR id
    matches: list[tuple[BeautifulSoup, int]] = []
    for level_tag in ("h1","h2","h3","h4","h5","h6"):
        level = int(level_tag[1])
        for h in soup.find_all(level_tag):
            heading_text = h.get_text(" ", strip=True)
            heading_id = (h.get("id") or "").strip()
            if section_regex.search(heading_text) or section_regex.search(heading_id):
                matches.append((h, level))

    for h, level in matches:
        for sib in h.find_all_next():
            # stop when we hit a same- or higher-level heading
            if sib.name in ("h1","h2","h3","h4","h5","h6"):
                if int(sib.name[1]) <= level:
                    break
                else:
                    continue
            # within the section content, collect anchors
            if getattr(sib, "find_all", None):
                for a in sib.find_all("a", href=True):
                    nu = clean_url(base_url, a["href"])
                    if nu:
                        refs.add(nu)
    return refs

def crawl_web(config_path: str, out_path: str):
    cfg = load_sources(config_path)
    web = cfg.get("web") or {}

    # Treat OWASP + Cheat Sheets as "internal" based on YAML
    allowed_domains = {d.lower() for d in (web.get("allowed_domains") or [])}

    seeds = [canonicalize(s) for s in (web.get("seed_urls", []) or [])]
    max_pages = int(web.get("max_pages", 200))
    rate = float(web.get("rate_limit_seconds", 1.0))
    respect_robots = bool(web.get("respect_robots_txt", True))
    min_words = int(web.get("min_words", 20))

    # Core Top10 path filters
    allowed_path_regex = re.compile(web["allowed_path_regex"], re.IGNORECASE) if web.get("allowed_path_regex") else None
    disallow_path_regex = re.compile(web["disallow_path_regex"], re.IGNORECASE) if web.get("disallow_path_regex") else None

    # External expansion knobs
    max_external_hops = int(web.get("max_external_hops", 1))              # hops from seeds
    external_require_https = bool(web.get("external_require_https", True)) # https only
    # If not provided, default to 'references'
    section_pat = web.get("external_from_sections_regex") or "references"
    section_re = re.compile(section_pat, re.IGNORECASE)
    external_max_pages_total = int(web.get("external_max_pages_total", 0)) # 0 = no cap
    external_per_domain_cap = int(web.get("external_per_domain_cap", 0))   # 0 = no cap

    out = Path(out_path); out.parent.mkdir(parents=True, exist_ok=True)

    # robots.txt cache
    rp_cache = {}
    def allowed_by_robots(url):
        if not respect_robots:
            return True
        parsed = urlparse(url)
        root = f"{parsed.scheme}://{parsed.netloc}"
        if root not in rp_cache:
            rp = urobot.RobotFileParser()
            try:
                rp.set_url(urljoin(root, "/robots.txt"))
                rp.read()
            except Exception:
                rp = None
            rp_cache[root] = rp
        rp = rp_cache[root]
        return True if (rp is None) else rp.can_fetch(HEADERS["User-Agent"], url)

    # State
    seen = set()
    saved = 0
    external_pages_total = 0
    per_domain_counts = defaultdict(int)

    # Queue holds (url, external_hops)
    q = deque((s, 0) for s in seeds)

    with open(out, "w", encoding="utf-8") as fout:
        while q and saved < max_pages:
            url, ext_hops = q.popleft()
            url = canonicalize(url)
            if url in seen:
                continue

            u = urlparse(url)
            # Internal = any page on allowed_domains (Top10 + Cheat Sheets). Path allowlist still applies below.
            internal_core = (u.netloc.lower() in allowed_domains)

            # PRE-FETCH FILTERS (no network yet)
            if internal_core:
                if LOCALE_RE.search(u.path):
                    continue
                if allowed_path_regex and not allowed_path_regex.search(u.path):
                    if url not in seeds:
                        continue
                if disallow_path_regex and disallow_path_regex.search(u.path):
                    continue
            else:
                if ext_hops > max_external_hops:
                    continue
                if external_require_https and not is_https(url):
                    continue
                if external_max_pages_total and external_pages_total >= external_max_pages_total:
                    continue
                if external_per_domain_cap:
                    dom = u.netloc.lower()
                    if per_domain_counts[dom] >= external_per_domain_cap:
                        continue

            if not allowed_by_robots(url):
                continue

            seen.add(url)

            try:
                resp = requests.get(url, headers=HEADERS, timeout=20)
                print(f"[fetch] {resp.status_code} {url}")
                ctype = resp.headers.get("Content-Type", "").lower()
                if "text/html" not in ctype:
                    continue

                html = resp.text
                text = extract_readable(html, url)
                soup = BeautifulSoup(html, "lxml")
                title = soup.title.string.strip() if soup.title and soup.title.string else url

                # Collect the EXACT set of external reference links on this internal page
                refs_set = set()
                if internal_core:
                    refs_set = collect_references_links(soup, url, section_re)

                if text and len(text.split()) >= min_words:
                    fout.write(json.dumps({"url": url, "title": title, "text": text}, ensure_ascii=False) + "\n")
                    saved += 1
                    if not internal_core:
                        external_pages_total += 1
                        per_domain_counts[u.netloc.lower()] += 1

                # ENQUEUE LINKS (filter BEFORE enqueue)
                for a in soup.find_all("a", href=True):
                    nu = clean_url(url, a["href"])
                    if not nu or nu in seen:
                        continue
                    nu_u = urlparse(nu)

                    # INTERNAL LINK: stay inside OWASP/CheatSheets without consuming hop budget
                    if nu_u.netloc.lower() in allowed_domains:
                        # internal Top10: English-only filters
                        if LOCALE_RE.search(nu_u.path):
                            continue
                        if allowed_path_regex and not allowed_path_regex.search(nu_u.path):
                            continue
                        if disallow_path_regex and disallow_path_regex.search(nu_u.path):
                            continue
                        q.append((nu, ext_hops))  # internal: same hop depth
                    else:
                        # Only allow going external FROM an internal Top10 page
                        if not internal_core:
                            continue
                        if ext_hops >= max_external_hops:
                            continue
                        if external_require_https and not is_https(nu):
                            continue
                        # Strict: only external links that were inside the 'References' slice
                        if refs_set and nu not in refs_set:
                            continue
                        if external_max_pages_total and external_pages_total >= external_max_pages_total:
                            continue
                        if external_per_domain_cap:
                            dom = nu_u.netloc.lower()
                            if per_domain_counts[dom] >= external_per_domain_cap:
                                continue
                        q.append((nu, ext_hops + 1))  # external: one hop deeper

            except requests.RequestException:
                pass

            time.sleep(rate)

    print(f"[crawl] saved {saved} pages to {out}")
    internal_saved = saved - external_pages_total
    print(f"[stats] internal={internal_saved}, external={external_pages_total}, total={saved}")
    if per_domain_counts:
        top = sorted(per_domain_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
        print("[stats] top external domains: " + ", ".join(f"{d}: {c}" for d, c in top))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sources.yaml")
    ap.add_argument("--out", default="data/raw/crawl.jsonl")
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    crawl_web(args.config, args.out)
