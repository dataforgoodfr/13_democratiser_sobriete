"""
Utilities to download PDF files with standard HTTP request, with fallback to Selenium.
"""

from contextlib import contextmanager
import time
import logging
import requests
from requests.adapters import HTTPAdapter, Retry
from pathlib import Path
import shutil
import tempfile

from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


@contextmanager
def get_session(max_retries=3):
    """Context manager for requests session with proper cleanup."""
    session = make_session(max_retries)
    try:
        yield session
    finally:
        session.close()


# Shared session with retries
def make_session(max_retries=3):
    session = requests.Session()
    retries = Retry(
        total=max_retries,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["HEAD", "GET", "OPTIONS"]),
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.mount("http://", HTTPAdapter(max_retries=retries))
    session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; PDF-Downloader/1.0)"})
    return session


def probe_url_headers(url: str, session: requests.Session, timeout: int = 10):
    """
    Determines if url is a standard PDF file.
    Try HEAD first, fallback to GET (streamed) if HEAD blocked.
    Returns a dict: { 'content_type': str|None, 'content_length': int|None, 'headers': dict, 'first_chunk': bytes|None }
    'first_chunk' will be None unless GET was used and we read one chunk (small).
    """
    info = {"content_type": None, "content_length": None, "headers": {}, "first_chunk": None}
    try:
        resp = session.head(url, allow_redirects=True, timeout=timeout)
        info["headers"] = resp.headers
        info["content_type"] = resp.headers.get("Content-Type")
        cl = resp.headers.get("Content-Length")
        info["content_length"] = int(cl) if cl and cl.isdigit() else None

        # Some servers respond 405 to HEAD; treat non-2xx as HEAD failure
        if resp.status_code >= 400:
            raise Exception(f"HEAD status {resp.status_code}")
        return info
    except Exception:
        # HEAD failed or unreliable — do a streamed GET and read just first chunk
        try:
            resp = session.get(url, allow_redirects=True, stream=True, timeout=timeout)
            info["headers"] = resp.headers
            info["content_type"] = resp.headers.get("Content-Type")
            cl = resp.headers.get("Content-Length")
            info["content_length"] = int(cl) if cl and cl.isdigit() else None

            # Read first chunk without draining whole response
            iter_chunks = resp.iter_content(chunk_size=8192)
            first = next(iter_chunks, b"")
            info["first_chunk"] = first
            # Important: we must close response (we will re-open streaming download if we proceed)
            resp.close()
            return info
        except Exception:
            return info


def looks_like_pdf_from_headers(info: dict) -> bool:
    ct = (info.get("content_type") or "").lower()
    if "pdf" in ct:
        return True
    cd = (info.get("headers") or {}).get("Content-Disposition", "")
    if ".pdf" in cd.lower():
        return True
    return False


def looks_like_pdf_from_bytes(first_chunk: bytes) -> bool:
    return bool(first_chunk and first_chunk.startswith(b"%PDF"))


def download_pdf(
    url: str,
    output_path: str,
    webdriver: webdriver.Chrome | None = None,
    timeout: int = 30,
) -> tuple[str | None, bool]:
    """
    1) Probe URL (HEAD/GET small).
    2) If probe says PDF or first bytes indicate PDF, stream-download to a temp file and atomically move.
    3) Otherwise fallback to Selenium.

    Return output path (or None) and bool indicating if Selenium was used.
    """
    with get_session() as session:
        info = probe_url_headers(url, session, timeout=min(10, timeout))

        # Heuristics
        if looks_like_pdf_from_headers(info) or looks_like_pdf_from_bytes(
            info.get("first_chunk", b"")
        ):
            return download_pdf_with_request(session, url, output_path, info, timeout), False
        elif webdriver:
            return download_pdf_with_webdriver(webdriver, url, output_path, timeout), True
        else:
            return None, False


def download_pdf_with_request(
    session: requests.Session,
    url: str,
    output_path: str,
    info: dict,
    timeout: int = 30,
    min_size_bytes: int = 1024,
) -> str | None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with session.get(url, stream=True, timeout=timeout, allow_redirects=True) as r:
        r.raise_for_status()
        # write to temp file in target dir
        with tempfile.NamedTemporaryFile(delete=False, dir=str(output_path.parent)) as tmpf:
            tmp_path = Path(tmpf.name)
            # If we already read first_chunk in probe and it's non-empty, write it first.
            first = info.get("first_chunk")
            if first:
                tmpf.write(first)
            bytes_written = len(first) if first else 0
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    tmpf.write(chunk)
                    bytes_written += len(chunk)
        # Basic validation
        if bytes_written < min_size_bytes:
            tmp_path.unlink(missing_ok=True)
            return None
        with open(tmp_path, "rb") as fh:
            if not fh.read(4) == b"%PDF":
                # not a PDF (delete and bail)
                tmp_path.unlink(missing_ok=True)
                return None
        shutil.move(str(tmp_path), str(output_path))
        return str(output_path)


def download_pdf_with_webdriver(
    driver: webdriver.Chrome, url: str, output_path: str, timeout: int = 30
) -> str | None:
    """Download PDF from URL with proper file handling"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    driver.get(url)
    time.sleep(2)

    wait_time = 0
    download_started = False

    while wait_time < timeout:
        temp_files = list(Path(tempfile.gettempdir()).glob("*.pdf"))

        if temp_files and not download_started:
            download_started = True

        for temp_file in temp_files:
            if temp_file.suffix.lower() == ".pdf" and temp_file.stat().st_size > 0:
                try:
                    shutil.move(str(temp_file), str(output_path))
                    if output_path.exists():
                        return str(output_path)
                    else:
                        logging.error(f"File move failed: {output_path}")
                        return None
                except Exception as e:
                    logging.error(f"Error moving file: {e}")
                    return None

        time.sleep(1)
        wait_time += 1

    logging.warning(f"Download timeout after {timeout}s")
    return None


def start_webdriver(download_dir: str) -> webdriver.Chrome:
    """Start Selenium webdriver with proper download configuration"""
    chrome_options = webdriver.ChromeOptions()

    # Configure download preferences FIRST (before creating driver)
    prefs = {
        "download.default_directory": str(Path(download_dir).absolute()),
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "plugins.always_open_pdf_externally": True,
        "profile.default_content_settings.popups": 0,
        "profile.default_content_setting_values.automatic_downloads": 1,
    }
    chrome_options.add_experimental_option("prefs", prefs)

    # Add arguments for better PDF handling
    chrome_options.add_argument("--disable-popup-blocking")
    chrome_options.add_argument("--disable-web-security")
    chrome_options.add_argument("--allow-running-insecure-content")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-default-apps")
    chrome_options.add_argument("--no-first-run")

    # Enable logging for debugging downloads
    chrome_options.add_argument("--enable-logging")
    chrome_options.add_argument("--v=1")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    logging.info("Chrome webdriver started")

    driver.set_page_load_timeout(60)

    # Verify download directory is working
    logging.info(f"Chrome configured to download to: {Path(download_dir).absolute()}")
    return driver
