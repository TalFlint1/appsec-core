# src/connectors/web.py
import argparse, json, time, re
from collections import deque
from pathlib import Path
from urllib.parse import urljoin, urlparse
import urllib.robotparser as urobot
import requests
from bs4 import BeautifulSoup
import trafilatura
import yaml

# Use a more browser‑like User‑Agent to reduce blocking by sites. Some servers
# aggressively block unknown bots; a familiar UA string helps ensure pages are
# served. The project and contact info are still included so owners can reach us.
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; DocPilotRAG/0.1; +https://example.local)"
}

def load_sources(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def same_domain(url, allowed_domains):
    host = urlparse(url).netloc.lower()
    # Accept subdomains by matching the end of the hostname. This allows
    # e.g. `cheatsheetseries.owasp.org` when `owasp.org` is allowed.
    return any(host == d.lower() or host.endswith("." + d.lower()) for d in allowed_domains)

def clean_url(base, href):
    if not href: return None
    href = href.strip()
    if href.startswith("#") or href.startswith("mailto:") or href.startswith("javascript:"):
        return None
    return urljoin(base, href)

def extract_readable(html, url):
    # prefer trafilatura; fallback to simple bs4 text
    try:
        downloaded = trafilatura.extract(html, include_comments=False, include_tables=False, url=url)
        if downloaded and len(downloaded.split()) > 50:
            return downloaded
    except Exception:
        pass
    soup = BeautifulSoup(html, "lxml")
    # drop nav/aside/footer
    for tag in soup(["nav","aside","footer","script","style","noscript"]):
        tag.decompose()
    text = soup.get_text("\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text

def crawl_web(config_path: str, out_path: str):
    cfg = load_sources(config_path)
    web = (cfg or {}).get("web") or {}
    allowed = web.get("allowed_domains", [])
    seeds = web.get("seed_urls", [])
    max_pages = int(web.get("max_pages", 100))
    rate = float(web.get("rate_limit_seconds", 1.0))
    respect_robots = bool(web.get("respect_robots_txt", True))

    out = Path(out_path); out.parent.mkdir(parents=True, exist_ok=True)

    rp_cache = {}
    def allowed_by_robots(url):
        if not respect_robots: return True
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

    seen, q, saved = set(), deque(seeds), 0
    with open(out, "w", encoding="utf-8") as fout:
        # Determine a minimum number of words required to save a page. Lowering this
        # threshold makes it easier to collect small pages such as section summaries.
        min_words = int(web.get("min_words", 20))
        # Compile optional path regex if provided in config to limit crawling scope.
        allowed_path_regex = None
        if web.get("allowed_path_regex"):
            try:
                allowed_path_regex = re.compile(web["allowed_path_regex"], re.IGNORECASE)
            except re.error:
                allowed_path_regex = None
        # Compile optional disallow regex to exclude paths (e.g., non-English locales)
        disallow_path_regex = None
        if web.get("disallow_path_regex"):
            try:
                disallow_path_regex = re.compile(web["disallow_path_regex"], re.IGNORECASE)
            except re.error:
                disallow_path_regex = None
        while q and saved < max_pages:
            url = q.popleft()

            # Skip URLs we've already visited or that fall outside the allowed domain set.
            if url in seen or not same_domain(url, allowed):
                continue
            seen.add(url)
            if not allowed_by_robots(url):
                continue
            
            # Path-level allow/disallow checks for the current URL
            path_cur = urlparse(url).path
            if allowed_path_regex and not allowed_path_regex.search(path_cur):
                # allow the seed/landing page even if it doesn't match; otherwise skip
                if url not in seeds:
                    continue
            if disallow_path_regex and disallow_path_regex.search(path_cur):
                continue
            try:
                resp = requests.get(url, headers=HEADERS, timeout=20)
                # Log the HTTP status to aid debugging; disabled by default in config.
                print(f"[fetch] {resp.status_code} {url}")
                # Skip non‑HTML content types. Normalise the header to lower case
                # before checking so it matches e.g. 'Text/HTML; charset=utf-8'.
                ctype = resp.headers.get("Content-Type", "").lower()
                if "text/html" not in ctype:
                    continue
                html = resp.text
                text = extract_readable(html, url)
                # Derive page title, falling back to the URL if missing.
                soup = BeautifulSoup(html, "lxml")
                t_tag = soup.title.string.strip() if soup.title and soup.title.string else url
                if text and len(text.split()) >= min_words:
                    rec = {"url": url, "title": t_tag, "text": text}
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    saved += 1
                # enqueue links on the same domain, filtered by path allow/disallow
                for a in soup.find_all("a", href=True):
                    nu = clean_url(url, a["href"])
                    if not nu or nu in seen:
                        continue
                    if not same_domain(nu, allowed):
                        continue
                    path_next = urlparse(nu).path
                    if allowed_path_regex and not allowed_path_regex.search(path_next):
                        continue
                    if disallow_path_regex and disallow_path_regex.search(path_next):
                        continue
                    q.append(nu)

            except requests.RequestException:
                # Ignore network errors and continue crawling
                pass
            time.sleep(rate)

    print(f"[crawl] saved {saved} pages to {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sources.yaml")
    ap.add_argument("--out", default="data/raw/crawl.jsonl")
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    crawl_web(args.config, args.out)
