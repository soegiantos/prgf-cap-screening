#!/usr/bin/env python3
"""NOMAD continuation harvest — Ca+P entries beyond the first 2000.

First run (2026-11-17) capped at 40 pages x 50 = 2000 entries; NOMAD holds
27,469 Ca+P entries, so ~25,469 remain. This run resumes with the CORRECT
pagination: request field is `page_after_value`, response cursor is
`next_page_after_value` (the first script sent the cursor under the wrong
key, which is why the first run repeated the same 50 entries 40x before
the fix). Cursor is saved after every page to nomad_cont_cursor.json so a
later run can resume exactly where this one stops.

Writes only NEW entry_ids (dedup against the existing 2000 + this file) to
data_raw/nomad_ca_p_cont.jsonl, compact fields only (entry_id, formula,
elements). Cap: 8000 new entries per run. Run with nohup; progress to stdout.
"""
import json, os, time
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "data_raw")
os.makedirs(OUT_DIR, exist_ok=True)

UA = {"User-Agent": "Muse-PRGF-research/1.0 (academic, Universitas Airlangga)"}
URL = "https://nomad-lab.eu/prod/v1/api/v1/entries/query"
CONT_PATH = os.path.join(OUT_DIR, "nomad_ca_p_cont.jsonl")
CURSOR_PATH = os.path.join(OUT_DIR, "nomad_cont_cursor.json")
CAP_NEW = 8000
PAGE_SIZE = 50


def load_seen():
    seen = set()
    for path in (os.path.join(OUT_DIR, "nomad_ca_p.jsonl"), CONT_PATH):
        try:
            with open(path) as f:
                for line in f:
                    try:
                        seen.add(json.loads(line).get("entry_id"))
                    except Exception:
                        pass
        except FileNotFoundError:
            pass
    return seen


def main():
    seen = load_seen()
    print(f"seen entry_ids already: {len(seen)}", flush=True)
    after = None
    if os.path.exists(CURSOR_PATH):
        try:
            after = json.load(open(CURSOR_PATH)).get("page_after_value")
            print(f"resuming from saved cursor: {str(after)[:60]}", flush=True)
        except Exception as e:
            print(f"cursor file unreadable ({e}), starting from page 1", flush=True)
            after = None
    new_n = 0
    pages = 0
    with open(CONT_PATH, "a") as f:
        while new_n < CAP_NEW:
            p = {"query": {"results.material.elements": {"all": ["Ca", "P"]}},
                 "pagination": {"page_size": PAGE_SIZE}}
            if after:
                p["pagination"]["page_after_value"] = after
            try:
                rr = requests.post(URL, json=p, timeout=60, headers=UA)
            except Exception as e:
                print(f"NOMAD page {pages}: ERROR {e}, stopping (cursor saved)", flush=True)
                break
            if rr.status_code != 200:
                print(f"NOMAD page {pages}: HTTP {rr.status_code}, stopping", flush=True)
                break
            d = rr.json()
            data = d.get("data", [])
            if not data:
                print("NOMAD: no more data", flush=True)
                break
            for e in data:
                eid = e.get("entry_id")
                if eid in seen:
                    continue
                compact = {"entry_id": eid,
                           "formula": (e.get("results") or {}).get("material", {}).get("formula"),
                           "elements": (e.get("results") or {}).get("material", {}).get("elements")}
                f.write(json.dumps(compact) + "\n")
                seen.add(eid)
                new_n += 1
            after = d.get("pagination", {}).get("next_page_after_value")
            pages += 1
            with open(CURSOR_PATH, "w") as cf:
                json.dump({"page_after_value": after, "pages_done": pages,
                           "new_entries": new_n}, cf)
            if pages % 10 == 0 or new_n >= CAP_NEW:
                print(f"NOMAD cont: page {pages} -> +{new_n} new, total in DB "
                      f"{d.get('pagination', {}).get('total')}", flush=True)
            if not after:
                print("NOMAD: pagination exhausted", flush=True)
                break
            time.sleep(0.5)
    print(f"NOMAD CONT DONE: +{new_n} new entries in {pages} pages", flush=True)


if __name__ == "__main__":
    main()
