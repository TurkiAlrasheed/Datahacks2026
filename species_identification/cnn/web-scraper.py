"""
iNaturalist image scraper — pulls more training images per species.

Given your existing `data/<species_name>/...` folder structure, this script:
  1. Reads the species names from your existing folders.
  2. Queries the iNaturalist API for research-grade observations of each
     species, optionally filtered to a geographic bounding box.
  3. Downloads up to N photos per species, skipping duplicates.

Run on your laptop in the project folder:
    python scrape_inaturalist.py

No API key required. Uses the iNaturalist v1 REST API.
Respects their rate limit (60 req/min) with conservative pacing.
"""

import hashlib
import json
import pathlib
import time
import urllib.parse
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR         = "../ucsd-data"                 # folder-per-species root
TARGET_PER_CLASS = 100                    # download until each class has this many total

# La Jolla Cove bounding box (roughly). Expand generously — tight bboxes
# miss good photos taken nearby. Set to None to disable geo filtering
# and pull from anywhere. For a marine dataset, I'd keep a loose bbox
# so you get photos taken along the Southern California coast.
BBOX = {
    "swlat": 32.60,   # south
    "swlng": -117.50, # west
    "nelat": 33.10,   # north
    "nelng": -117.00, # east
}
# BBOX = None

# Photo size. iNaturalist offers: square (75), small (240), medium (500),
# large (1024), original. "medium" is a good balance — big enough for
# 224-input training, small enough to fetch quickly.
PHOTO_SIZE = "medium"

# API pacing. iNat asks for <=60 req/min; we stay well under.
REQUEST_DELAY_S  = 1.1    # between metadata page requests (~55/min)
DOWNLOAD_WORKERS = 4      # parallel image downloads (separate from metadata)
PER_PAGE         = 100    # max allowed by iNat
USER_AGENT       = "la-jolla-species-classifier/1.0 (training dataset, contact: you@example.com)"

# Safety: skip any image larger than this (bytes) — corrupt downloads
# occasionally come through as multi-MB HTML error pages.
MAX_IMAGE_BYTES  = 8_000_000


# ---------------------------------------------------------------------------
API_BASE = "https://api.inaturalist.org/v1/observations"


def http_get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def http_get_bytes(url, max_bytes=MAX_IMAGE_BYTES):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = r.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise IOError(f"Oversized response ({len(data)} bytes)")
        return data


def build_query(taxon_name, page):
    params = {
        "taxon_name": taxon_name,
        "quality_grade": "research",
        "photos": "true",           # require at least one photo
        "per_page": PER_PAGE,
        "page": page,
        "order": "desc",
        "order_by": "created_at",
    }
    if BBOX is not None:
        params.update(BBOX)
    return f"{API_BASE}?{urllib.parse.urlencode(params)}"


def extract_photo_urls(obs_json):
    """Pull medium-size image URLs out of an observation JSON."""
    urls = []
    for obs in obs_json.get("results", []):
        for photo in obs.get("photos", []):
            # Photo URLs look like:
            #   https://inaturalist-open-data.s3.amazonaws.com/photos/12345/square.jpg
            # Swap the size segment to get the version we want.
            raw = photo.get("url")
            if not raw:
                continue
            for size in ("square", "small", "medium", "large", "original"):
                raw = raw.replace(f"/{size}.", f"/{PHOTO_SIZE}.")
            urls.append(raw)
    return urls


def fetch_species_urls(species_name, needed, existing_stems):
    """Paginate through the API until we have `needed` new photo URLs."""
    collected = []
    page = 1
    while len(collected) < needed and page <= 10:   # cap at 1000 results
        url = build_query(species_name, page)
        try:
            data = http_get_json(url)
        except urllib.error.HTTPError as e:
            print(f"  [{species_name}] API error page {page}: {e}")
            break
        photo_urls = extract_photo_urls(data)
        if not photo_urls:
            break
        for u in photo_urls:
            # Skip anything we already have, using URL hash as filename stem.
            stem = hashlib.md5(u.encode()).hexdigest()[:16]
            if stem in existing_stems:
                continue
            collected.append((stem, u))
            existing_stems.add(stem)
            if len(collected) >= needed:
                break
        if data.get("total_results", 0) <= page * PER_PAGE:
            break
        page += 1
        time.sleep(REQUEST_DELAY_S)
    return collected


def download_one(url_and_stem, out_dir):
    stem, url = url_and_stem
    ext = pathlib.Path(urllib.parse.urlparse(url).path).suffix.lower() or ".jpg"
    out_path = out_dir / f"inat_{stem}{ext}"
    try:
        data = http_get_bytes(url)
    except Exception as e:
        return (stem, False, str(e))
    if len(data) < 1000:   # suspiciously tiny — probably a redirect/error
        return (stem, False, f"tiny file ({len(data)} bytes)")
    out_path.write_bytes(data)
    return (stem, True, None)


def _find_species_dirs(data_root):
    """Locate species folders, handling two layouts:
        (A) data/<species>/*.jpg                    (flat)
        (B) data/{train,val,test}/<species>/*.jpg   (pre-split)
    For (B) we return the `train/<species>` folders — new images should
    go into the training split, not val or test.
    """
    top = sorted([d for d in data_root.iterdir() if d.is_dir()])
    names = {d.name.lower() for d in top}
    split_names = {"train", "val", "valid", "validation", "test"}
    if names & split_names:
        # Find the train split (or the biggest one if 'train' isn't named).
        train_candidates = [d for d in top
                            if d.name.lower() in {"train", "training"}]
        if train_candidates:
            train_dir = train_candidates[0]
        else:
            train_dir = max((d for d in top if d.name.lower() in split_names),
                            key=lambda d: sum(1 for _ in d.rglob("*")))
        print(f"Pre-split layout detected. Adding images to '{train_dir.name}/'")
        return sorted([c for c in train_dir.iterdir() if c.is_dir()])
    return top


def main():
    data_root = pathlib.Path(DATA_DIR)
    if not data_root.exists():
        raise FileNotFoundError(
            f"'{DATA_DIR}/' not found. Run this in the project folder."
        )

    species_dirs = _find_species_dirs(data_root)
    print(f"Found {len(species_dirs)} species folders\n")

    # If we're in a pre-split layout, also peek at val/test folders so we
    # don't redownload images that live there, and so we count the true
    # current total when deciding how many more to fetch.
    def _peer_splits(species_name):
        peers = []
        for split in ("val", "valid", "validation", "test"):
            p = data_root / split / species_name
            if p.exists() and p.is_dir():
                peers.append(p)
        return peers

    total_added = 0
    for sdir in species_dirs:
        species = sdir.name
        # Count existing images in this species' train folder AND in its
        # val/test peers (if pre-split), so dedup hashes cover all of them.
        peer_dirs = _peer_splits(species)
        all_existing = list(sdir.glob("*"))
        for pd in peer_dirs:
            all_existing.extend(pd.glob("*"))
        existing_stems = {p.stem.replace("inat_", "") for p in all_existing}
        have = len(all_existing)
        need = TARGET_PER_CLASS - have
        if need <= 0:
            print(f"  {species}: already have {have} (across splits), skipping")
            continue

        # The taxon_name parameter accepts common or scientific names but
        # underscores confuse it — swap to spaces.
        query_name = species.replace("_", " ")
        print(f"  {species}: have {have}, fetching up to {need} more ...")

        urls = fetch_species_urls(query_name, need, existing_stems)
        if not urls:
            print(f"    no additional photos found on iNaturalist")
            continue

        # Parallel downloads, capped to DOWNLOAD_WORKERS.
        # Always write to sdir (the train folder) — new data belongs there.
        added = 0
        errors = 0
        with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as ex:
            futures = [ex.submit(download_one, u, sdir) for u in urls]
            for fut in as_completed(futures):
                stem, ok, err = fut.result()
                if ok:
                    added += 1
                else:
                    errors += 1
        print(f"    +{added} images (errors: {errors})")
        total_added += added

    print(f"\nDone. Added {total_added} images total across all species.")
    print(f"Now re-run the training script to train on the expanded dataset.")


if __name__ == "__main__":
    main()