from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import shutil

import logging
import re
import requests
import tempfile
log = logging.getLogger(__file__)
log.setLevel(logging.INFO)

from tqdm import tqdm as _tqdm
cache_root = os.path.expanduser(os.path.join("~", ".bert-test"))

class Tqdm:
    # These defaults are the same as the argument defaults in tqdm.
    default_mininterval= 0.1

    @staticmethod
    def set_default_mininterval(value: float) -> None:
        Tqdm.default_mininterval = value

    @staticmethod
    def set_slower_interval(use_slower_interval: bool) -> None:
        """
        If ``use_slower_interval`` is ``True``, we will dramatically slow down ``tqdm's`` default
        output rate.  ``tqdm's`` default output rate is great for interactively watching progress,
        but it is not great for log files.  You might want to set this if you are primarily going
        to be looking at output through log files, not the terminal.
        """
        if use_slower_interval:
            Tqdm.default_mininterval = 10.0
        else:
            Tqdm.default_mininterval = 0.1

    @staticmethod
    def tqdm(*args, **kwargs):
        new_kwargs = {"mininterval": Tqdm.default_mininterval, **kwargs}

        return _tqdm(*args, **new_kwargs)

def setupLog():
    """
    Reset logging: to make the logging.basicConfig() work again.

    Step 1: remove all the root.handlers
       logging.basicConfig() does nothing if the root logger already has handlers
            configured for it.
    Step 2: setup logging format
        logging.basicConfig() will create and add a default handler to the root logger.
    """
    if logging.root:
        logging.root.handlers = []
    fmt = "%(levelname).3s[%(asctime)s %(filename)s:%(lineno)d] %(message)s"
    dtfmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(format=fmt, datefmt=dtfmt)
    return


def cached_path(url_or_filename: str, cache_dir: Path) -> Path:
    dataset_cache = Path(cache_root) / cache_dir

    parsed = urlparse(url_or_filename)

    if parsed.scheme in ("http", "https"):
        # URL, so get it from the cache (downloading if necessary)
        return get_from_cache(url_or_filename, dataset_cache)
    elif parsed.scheme == "" and Path(url_or_filename).exists():
        # File, and it exists.
        return Path(url_or_filename)
    elif parsed.scheme == "":
        # File, but it doesn't exist.
        raise FileNotFoundError("file {} not found".format(url_or_filename))
    else:
        # Something unknown
        raise ValueError(
            "unable to parse {} as a URL or as a local path".format(url_or_filename)
        )

def get_from_cache(url: str, cache_dir: Path = None) -> Path:
    """
    Given a URL, look for the corresponding dataset in the local cache.
    If it's not there, download it. Then return the path to the cached file.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    filename = re.sub(r".+/", "", url)
    # get cache path to put the file
    cache_path = cache_dir / filename
    if cache_path.exists():
        return cache_path

    # make HEAD request to check ETag
    response = requests.head(url, headers={"User-Agent": "Flair"})
    if response.status_code != 200:
        raise IOError(
            "HEAD request failed for url {url} with status code {response.status_code}."
        )

    # add ETag to filename if it exists
    # etag = response.headers.get("ETag")

    if not cache_path.exists():
        # Download to temporary file, then copy to cache dir once finished.
        # Otherwise you get corrupt cache entries if the download gets interrupted.
        fd, temp_filename = tempfile.mkstemp()
        

        # GET file object
        req = requests.get(url, stream=True, headers={"User-Agent": "Flair"})
        content_length = req.headers.get("Content-Length")
        total = int(content_length) if content_length is not None else None
        progress = Tqdm.tqdm(unit="B", total=total)
        with open(temp_filename, "wb") as temp_file:
            for chunk in req.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive new chunks
                    progress.update(len(chunk))
                    temp_file.write(chunk)

        progress.close()

        
        shutil.copyfile(temp_filename, str(cache_path))
        os.close(fd)
        os.remove(temp_filename)

    return cache_path