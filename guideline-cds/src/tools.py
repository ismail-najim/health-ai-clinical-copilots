"""Live lookups against public authorities, with offline-safe fallbacks.

Two helpers:
  - `uspstf_lookup`   hits the USPSTF Prevention TaskForce API for grade-attributed
                      screening / prevention recommendations.
  - `openfda_label_lookup`  hits openFDA for FDA drug-label sections (indications,
                      dosage, boxed warning, contraindications, interactions).

Both take an `offline` switch (and both fall back automatically if `requests` is
missing or the network call fails), returning a small built-in example payload so
the test suite never needs the network. The USPSTF API needs a free key; openFDA is
open with an optional key that only raises the rate limit. Every returned record
carries a source id, authority, license, and url so the citation gate can treat it
exactly like a corpus snippet.
"""
from __future__ import annotations

import os
from typing import Optional

USPSTF_BASE = "https://data.uspreventiveservicestaskforce.org/api/json"
OPENFDA_BASE = "https://api.fda.gov/drug/label.json"


# --- offline example payloads ------------------------------------------------

_USPSTF_OFFLINE = [
    {"id": "U-off-1", "title": "Colorectal cancer screening",
     "authority": "USPSTF (offline example)", "grade": "A",
     "license": "public-domain-example",
     "url": "https://www.uspreventiveservicestaskforce.org/",
     "text": ("USPSTF Grade A: screening for colorectal cancer in adults aged 50 "
              "to 75 years.")},
    {"id": "U-off-2", "title": "Lung cancer screening",
     "authority": "USPSTF (offline example)", "grade": "B",
     "license": "public-domain-example",
     "url": "https://www.uspreventiveservicestaskforce.org/",
     "text": ("USPSTF Grade B: annual low-dose CT screening for adults 50 to 80 "
              "with a 20 pack-year smoking history who currently smoke or quit "
              "within 15 years.")},
]

_OPENFDA_OFFLINE = {
    "metformin": [
        {"id": "F-off-metformin-boxed", "title": "metformin — boxed_warning",
         "authority": "FDA drug label (offline example)", "section": "boxed_warning",
         "license": "public-domain-example", "url": "https://open.fda.gov/",
         "text": ("Boxed warning: lactic acidosis is a rare but serious metabolic "
                  "complication; risk rises with renal impairment.")},
    ],
    "warfarin": [
        {"id": "F-off-warfarin-boxed", "title": "warfarin — boxed_warning",
         "authority": "FDA drug label (offline example)", "section": "boxed_warning",
         "license": "public-domain-example", "url": "https://open.fda.gov/",
         "text": ("Boxed warning: warfarin can cause major or fatal bleeding; "
                  "regular INR monitoring is required.")},
    ],
}


def _want_offline(offline: Optional[bool]) -> bool:
    if offline is not None:
        return offline
    # Auto: go offline if requests is unavailable.
    try:
        import requests  # noqa: F401
        return False
    except Exception:
        return True


# --- USPSTF ------------------------------------------------------------------

def uspstf_lookup(age: Optional[int] = None, sex: Optional[str] = None,
                  keyword: Optional[str] = None,
                  offline: Optional[bool] = None) -> list[dict]:
    """Return USPSTF preventive-service recommendations as citation-ready records.

    Grades: A/B (offer), C (selective), D (advise against), I (insufficient).
    Falls back to a built-in example set when offline or on any error.
    """
    if _want_offline(offline):
        recs = _USPSTF_OFFLINE
    else:
        try:
            import requests
            params: dict = {}
            key = os.environ.get("USPSTF_KEY")
            if key:
                params["key"] = key
            if age:
                params["age"] = age
            if sex:
                params["sex"] = sex
            resp = requests.get(USPSTF_BASE, params=params, timeout=30)
            resp.raise_for_status()
            raw = resp.json().get("specificRecommendations", [])
            recs = []
            for i, rec in enumerate(raw):
                grade = rec.get("grade", "")
                title = rec.get("title", "")
                body = rec.get("text", "") or rec.get("general", "")
                recs.append({
                    "id": f"U{i}", "title": title, "grade": grade,
                    "authority": "USPSTF", "license": "public-domain",
                    "url": rec.get("url", "https://www.uspreventiveservicestaskforce.org/"),
                    "text": f"USPSTF Grade {grade}: {title}. {body}".strip(),
                })
        except Exception:
            recs = _USPSTF_OFFLINE

    if keyword:
        kw = keyword.lower()
        recs = [r for r in recs if kw in (r["title"] + " " + r["text"]).lower()]
    return recs


# --- openFDA -----------------------------------------------------------------

_FDA_FIELDS = ("boxed_warning", "contraindications", "drug_interactions",
               "dosage_and_administration", "indications_and_usage")


def openfda_label_lookup(drug: str, offline: Optional[bool] = None) -> list[dict]:
    """Return FDA drug-label sections for `drug` as citation-ready records.

    Falls back to a built-in example set when offline or on any error.
    """
    drug_key = (drug or "").strip().lower()
    if _want_offline(offline):
        return list(_OPENFDA_OFFLINE.get(drug_key, []))

    try:
        import requests
        params: dict = {"search": f'openfda.generic_name:"{drug_key}"', "limit": 1}
        key = os.environ.get("OPENFDA_KEY")
        if key:
            params["api_key"] = key
        resp = requests.get(OPENFDA_BASE, params=params, timeout=30)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if not results:
            return list(_OPENFDA_OFFLINE.get(drug_key, []))
        label = results[0]
        out = []
        for field in _FDA_FIELDS:
            for i, txt in enumerate(label.get(field, [])):
                out.append({
                    "id": f"F-{drug_key}-{field}-{i}",
                    "title": f"{drug_key} — {field}",
                    "authority": "FDA drug label", "section": field,
                    "license": "public-domain", "url": "https://open.fda.gov/",
                    "text": txt,
                })
        return out or list(_OPENFDA_OFFLINE.get(drug_key, []))
    except Exception:
        return list(_OPENFDA_OFFLINE.get(drug_key, []))
