# # src/connectors/web.py
# import argparse, json, time, re
# from collections import deque, defaultdict
# from pathlib import Path
# from urllib.parse import urljoin, urlparse, unquote
# import urllib.robotparser as urobot

# import requests
# from bs4 import BeautifulSoup
# import trafilatura
# import yaml

# # Browser-like UA to avoid being blocked
# HEADERS = {
#     "User-Agent": (
#         "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
#         "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36; DocPilotRAG/0.1 (+https://example.local)"
#     )
# }

# # Locale path detector: /Top10/fr/... /Top10/pt-BR/... /Top10/zh-TW/...
# LOCALE_RE = re.compile(r"^/Top10/[A-Za-z]{2}(?:-[A-Za-z]{2,})?/", re.IGNORECASE)

# def load_sources(config_path: str):
#     with open(config_path, "r", encoding="utf-8") as f:
#         return yaml.safe_load(f)

# def same_domain(url, allowed_domains):
#     host = urlparse(url).netloc.lower()
#     return any(host == d.lower() or host.endswith("." + d.lower()) for d in allowed_domains)

# def is_internal(url: str) -> bool:
#     return urlparse(url).netloc.lower().endswith("owasp.org")

# def is_https(url: str) -> bool:
#     return urlparse(url).scheme.lower() == "https"

# def canonicalize(url: str) -> str:
#     """Normalize minor variants so we don't crawl duplicates."""
#     u = urlparse(url)
#     # decode %28SSRF%29 -> (SSRF), strip index.html, collapse //, ensure trailing /
#     path = unquote(u.path)
#     path = re.sub(r"/index\.html?$", "/", path)
#     path = re.sub(r"/{2,}", "/", path)
#     # Only append a trailing slash for directory-like paths.  If the path appears to
#     # end with a file extension (e.g. .html, .pdf), don't add a slash because many
#     # sites treat that as a different URL (often 404).  See issue where
#     # cheat‑sheet links were canonicalized to `.html/` and returned 404.
#     if not path.endswith("/") and not re.search(r"\.[A-Za-z0-9]{1,5}$", path):
#         path = path + "/"
#     return f"{u.scheme}://{u.netloc}{path}{('?' + u.query) if u.query else ''}"

# def clean_url(base, href):
#     if not href:
#         return None
#     href = href.strip()
#     if href.startswith("#") or href.startswith("mailto:") or href.startswith("javascript:"):
#         return None
#     return canonicalize(urljoin(base, href))

# def extract_readable(html, url):
#     # prefer trafilatura; fallback to simple bs4 text
#     try:
#         extracted = trafilatura.extract(html, include_comments=False, include_tables=False, url=url)
#         if extracted and len(extracted.split()) > 50:
#             return extracted
#     except Exception:
#         pass
#     soup = BeautifulSoup(html, "lxml")
#     for tag in soup(["nav", "aside", "footer", "script", "style", "noscript"]):
#         tag.decompose()
#     text = soup.get_text("\n")
#     text = re.sub(r"\n{3,}", "\n\n", text)
#     return text

# def crawl_web(config_path: str, out_path: str):
#     cfg = load_sources(config_path) or {}
#     web = cfg.get("web") or {}

#     allowed = web.get("allowed_domains", [])
#     seeds = [canonicalize(s) for s in (web.get("seed_urls", []) or [])]
#     max_pages = int(web.get("max_pages", 100))
#     rate = float(web.get("rate_limit_seconds", 1.0))
#     respect_robots = bool(web.get("respect_robots_txt", True))
#     min_words = int(web.get("min_words", 20))

#     # Internal path filters (case-insensitive)
#     allowed_path_regex = None
#     if web.get("allowed_path_regex"):
#         try:
#             allowed_path_regex = re.compile(web["allowed_path_regex"], re.IGNORECASE)
#         except re.error:
#             allowed_path_regex = None
#     disallow_path_regex = None
#     if web.get("disallow_path_regex"):
#         try:
#             disallow_path_regex = re.compile(web["disallow_path_regex"], re.IGNORECASE)
#         except re.error:
#             disallow_path_regex = None

#     # External expansion knobs (no domain allowlist)
#     max_external_hops = int(web.get("max_external_hops", 1))              # hops from seeds
#     external_require_https = bool(web.get("external_require_https", True)) # https-only
#     external_from_sections_regex = web.get("external_from_sections_regex") # e.g. 'references|cheat|prevention'
#     headings_re = re.compile(external_from_sections_regex, re.IGNORECASE) if external_from_sections_regex else None
#     external_max_pages_total = int(web.get("external_max_pages_total", 0)) # 0 = no cap
#     external_per_domain_cap = int(web.get("external_per_domain_cap", 0))   # 0 = no cap

#     out = Path(out_path)
#     out.parent.mkdir(parents=True, exist_ok=True)

#     # robots.txt cache
#     rp_cache = {}
#     def allowed_by_robots(url):
#         if not respect_robots:
#             return True
#         parsed = urlparse(url)
#         root = f"{parsed.scheme}://{parsed.netloc}"
#         if root not in rp_cache:
#             rp = urobot.RobotFileParser()
#             try:
#                 rp.set_url(urljoin(root, "/robots.txt"))
#                 rp.read()
#             except Exception:
#                 rp = None
#             rp_cache[root] = rp
#         rp = rp_cache[root]
#         return True if (rp is None) else rp.can_fetch(HEADERS["User-Agent"], url)

#     # State
#     seen = set()
#     saved = 0
#     external_pages_total = 0
#     per_domain_counts = defaultdict(int)

#     # Queue holds (url, external_hops)
#     q = deque((s, 0) for s in seeds)

#     def link_under_ok_heading(a_tag) -> bool:
#         if headings_re is None:
#             return True
#         h = a_tag.find_previous(["h1","h2","h3","h4","h5","h6"])
#         return bool(h and headings_re.search(h.get_text(" ").strip()))

#     with open(out, "w", encoding="utf-8") as fout:
#         while q and saved < max_pages:
#             url, ext_hops = q.popleft()
#             url = canonicalize(url)

#             # Skip if already seen
#             if url in seen:
#                 continue

#             internal = is_internal(url)
#             path_cur = urlparse(url).path

#             # PRE-FETCH FILTERS (no network yet)
#             if internal:
#                 # only crawl internal pages under owasp.org and Top10 filters
#                 if not same_domain(url, allowed):
#                     continue
#                 if LOCALE_RE.search(path_cur):
#                     continue
#                 if allowed_path_regex and not allowed_path_regex.search(path_cur):
#                     if url not in seeds:  # allow seeds even if not matching
#                         continue
#                 if disallow_path_regex and disallow_path_regex.search(path_cur):
#                     continue
#             else:
#                 # external page: honor hop/https/caps
#                 if ext_hops > max_external_hops:
#                     continue
#                 if external_require_https and not is_https(url):
#                     continue
#                 if external_max_pages_total and external_pages_total >= external_max_pages_total:
#                     continue
#                 if external_per_domain_cap:
#                     dom = urlparse(url).netloc.lower()
#                     if per_domain_counts[dom] >= external_per_domain_cap:
#                         continue

#             # robots.txt after path/hop checks
#             if not allowed_by_robots(url):
#                 continue

#             seen.add(url)

#             try:
#                 resp = requests.get(url, headers=HEADERS, timeout=20)
#                 print(f"[fetch] {resp.status_code} {url}")
#                 ctype = resp.headers.get("Content-Type", "").lower()
#                 if "text/html" not in ctype:
#                     continue

#                 html = resp.text
#                 text = extract_readable(html, url)
#                 soup = BeautifulSoup(html, "lxml")
#                 title = soup.title.string.strip() if soup.title and soup.title.string else url

#                 if text and len(text.split()) >= min_words:
#                     rec = {"url": url, "title": title, "text": text}
#                     fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
#                     saved += 1
#                     if not internal:
#                         external_pages_total += 1
#                         per_domain_counts[urlparse(url).netloc.lower()] += 1

#                 # ENQUEUE LINKS (filter BEFORE enqueue)
#                 for a in soup.find_all("a", href=True):
#                     nu = clean_url(url, a["href"])
#                     if not nu or nu in seen:
#                         continue

#                     if is_internal(nu):
#                         # internal link: enforce English-only Top10 filters
#                         if not same_domain(nu, allowed):
#                             continue
#                         path_next = urlparse(nu).path
#                         if LOCALE_RE.search(path_next):
#                             continue
#                         if allowed_path_regex and not allowed_path_regex.search(path_next):
#                             continue
#                         if disallow_path_regex and disallow_path_regex.search(path_next):
#                             continue
#                         # internal links do not increase external hops
#                         q.append((nu, ext_hops))
#                     else:
#                         # Only allow going external FROM an internal OWASP page
#                         if not internal:
#                             continue
#                         if ext_hops >= max_external_hops:
#                             continue
#                         if external_require_https and not is_https(nu):
#                             continue
#                         # keep only links that appear under permitted headings (References, etc.)
#                         if not link_under_ok_heading(a):
#                             continue
#                         # caps before enqueue
#                         if external_max_pages_total and external_pages_total >= external_max_pages_total:
#                             continue
#                         if external_per_domain_cap:
#                             dom = urlparse(nu).netloc.lower()
#                             if per_domain_counts[dom] >= external_per_domain_cap:
#                                 continue
#                         # enqueue external as one hop deeper
#                         q.append((nu, ext_hops + 1))

#             except requests.RequestException:
#                 pass

#             time.sleep(rate)

#         print(f"[crawl] saved {saved} pages to {out}")

#         # Print crawl statistics: internal vs external saved and top domains
#         internal_saved = saved - external_pages_total
#         print(f"[stats] internal={internal_saved}, external={external_pages_total}, total={saved}")
#         if per_domain_counts:
#             # show top 10 domains for external pages
#             top = sorted(per_domain_counts.items(), key=lambda item: item[1], reverse=True)[:10]
#             summary = ", ".join([f"{d}: {c}" for d, c in top])
#             print(f"[stats] top external domains: {summary}")

# if __name__ == "__main__":
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--config", default="configs/sources.yaml")
#     ap.add_argument("--out", default="data/raw/crawl.jsonl")
#     args = ap.parse_args()
#     Path(args.out).parent.mkdir(parents=True, exist_ok=True)
#     crawl_web(args.config, args.out)


# src/connectors/web.py
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
            internal_core = is_core_top10(url)

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

                    if is_core_top10(nu):
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
