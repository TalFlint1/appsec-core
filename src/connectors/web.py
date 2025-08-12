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

HEADERS = {"User-Agent": "DocPilotRAG/0.1 (+https://example.local)"}

def load_sources(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def same_domain(url, allowed_domains):
    host = urlparse(url).netloc.lower()
    return any(host.endswith(d.lower()) for d in allowed_domains)

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
        if downloaded and len(downloaded.split()) > 80:
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
        while q and saved < max_pages:
            url = q.popleft()
            if url in seen or not same_domain(url, allowed):
                continue
            seen.add(url)
            if not allowed_by_robots(url): continue

            try:
                r = requests.get(url, headers=HEADERS, timeout=20)
                if "text/html" not in r.headers.get("Content-Type",""):
                    continue
                text = extract_readable(r.text, url)
                title = BeautifulSoup(r.text, "lxml").title.string.strip() if BeautifulSoup(r.text, "lxml").title else url
                if text and len(text.split()) > 50:
                    rec = {"url": url, "title": title, "text": text}
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    saved += 1
                # enqueue links
                soup = BeautifulSoup(r.text, "lxml")
                for a in soup.find_all("a", href=True):
                    nu = clean_url(url, a["href"])
                    if nu and same_domain(nu, allowed) and nu not in seen:
                        q.append(nu)
            except requests.RequestException:
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
