# Wembley Rewind KG — Build Scripts & Process

This README captures the **scripts**, **pipeline**, and **gotchas** we used to build the Live Aid knowledge graph. It’s meant to be both a reproducible recipe and a record of the fixes made along the way.

---

## TL;DR (what you run most)

```bash
# 0) (Optional) create venv
python3.9 -m venv wemb && source wemb/bin/activate
pip install rdflib requests

# 1) Build/augment the core KG (artists, works, performances, setlists, membership)
# (Run from kg/scripts/, adjust paths if different)
python liveaid_sets_and_recordings.py            # release → sets → performances/works/recordings
python mb_band_membership.py                     # MB artist rels → 22_band_members.ttl + 50_instruments.ttl
python fix_labels_to_names.py                    # copy rdfs:label → schema:name where missing
python filter_artist_genres.py                   # keep only mm:hasGenre present in 11_genre.ttl
# (Optional) collapse duplicate works by MB Work ID if needed
python dedupe_works_by_mbid.py

# 2) External enrichment (Wikidata → Songfacts, Genius, YouTube, Images)
python enrich_wikidata_wikipedia.py --out ../25_wikidata_enrich.ttl
python link_songfacts_from_wikidata.py ../20_artists.ttl ../33_recordings_works.ttl ../40_setlists_songs.ttl ../25_wikidata_enrich.ttl --out ../26_songfacts_links.ttl
python link_genius_from_wikidata.py    ../20_artists.ttl ../33_recordings_works.ttl ../40_setlists_songs.ttl ../25_wikidata_enrich.ttl --out ../26_genius_links.ttl
python link_youtube_from_wikidata.py   ../20_artists.ttl ../33_recordings_works.ttl ../40_setlists_songs.ttl ../25_wikidata_enrich.ttl --out ../26_youtube_links.ttl
python link_images_from_wikidata.py    ../20_artists.ttl ../33_recordings_works.ttl ../40_setlists_songs.ttl ../25_wikidata_enrich.ttl --out ../26_wikidata_images.ttl

# 3) Grooming (dedupe + slice “links-only” layer to ship)
python dedupe_enrichment_triples.py   ../24_mb_enrich.ttl ../25_wikidata_enrich.ttl   ../26_wikidata_images.ttl ../26_songfacts_links.ttl ../26_genius_links.ttl ../26_youtube_links.ttl   --out ../27_enrichment_dedup.ttl

python slice_external_links.py ../27_enrichment_dedup.ttl   --keep-descriptions --desc-lang en --desc-max 280   --out ../27_external_links_only.ttl   --removed-out ../27_removed_nonlink_content.ttl   --report-out ../reports/external_slice_removed.csv
```

---

## Data model (what the triples look like)

**Ontologies**  
- `schema:` http://schema.org/  
- `mm:` https://w3id.org/polifonia/ontology/music-meta/  
- Our base: `ex:` http://wembrewind.live/ex#

**Key classes & relations**
- Artists: `schema:Person`, `schema:MusicGroup`, plus `mm:Musician` / `mm:MusicArtist`
- Works (songs): `schema:MusicComposition`, `mm:MusicEntity`
- Recordings: `mm:Recording`
- Performances: `mm:LivePerformance` (also `schema:Event`)
- Setlists: `schema:ItemList` with `schema:ListItem`/`schema:position`
- Links across:
  - Performance → Event: `schema:isPartOf` (and also set `schema:superEvent`)
  - Performance → Artist: `schema:performer`
  - Performance → Works played: `mm:performedWork`
  - Performance → Recordings used: `mm:recordedAs`
  - Work ↔ Recording: `mm:hasRecording`
- Genres: `mm:MusicGenre` via `mm:hasGenre`
- Membership: `mm:MusicEnsembleMembership` + `schema:Role` for roles/dates and `mm:involvesMemberOfMusicEnsemble` / `mm:involvesMusicEnsemble`  
- **Normalisation:** **standardise `schema.org` to `http://`** (not `https://`) everywhere for consistency.

---

## Pipeline at a glance

```
MusicBrainz release/event → (script) liveaid_sets_and_recordings.py
      │
      ├── Performances (mm:LivePerformance) with location + superEvent
      ├── Setlists (schema:ItemList) with ordered ListItems
      ├── Works (schema:MusicComposition) + Recordings (mm:Recording)
      └── Traceability: performance → setlist → work ↔ recording

MusicBrainz artist rels → (script) mb_band_membership.py
      └── mm:MusicEnsembleMembership nodes (+ roles, dates, instruments)

Fixes/Filters
  ├─ fix_labels_to_names.py (copy rdfs:label → schema:name)
  ├─ filter_artist_genres.py (keep only genres in 11_genre.ttl)
  └─ dedupe_works_by_mbid.py (collapse duplicate Works by same MBID)

External enrichment
  ├─ enrich_wikidata_wikipedia.py (QIDs etc.)
  ├─ link_songfacts_from_wikidata.py (P5241/P5287)
  ├─ link_genius_from_wikidata.py   (P6218/P6361/P2373/P6351)
  ├─ link_youtube_from_wikidata.py  (P1651, P2397)
  └─ link_images_from_wikidata.py   (P18, P154)

Grooming
  ├─ dedupe_enrichment_triples.py   (collapse identical IDs/videos)
  └─ slice_external_links.py        (links-only TTL + removed TTL + CSV report)
```

---

## Script catalog (what each one does)

| Script | Purpose | Inputs → Outputs | Notes / Flags |
|---|---|---|---|
| `liveaid_sets_and_recordings.py` | Harvest Live Aid sets from MB release; create performances, setlists, works, recordings | MB Release/Event → `23_liveaid_setlists.ttl`, `24_recordings_works.ttl` | Sets `schema:location`, `schema:isPartOf`, `schema:superEvent`; cleans names; guards duplicates |
| `mb_band_membership.py` | Band membership from MB artist relationships | `20_artists.ttl` → `22_band_members.ttl` (+ may mint missing artists) | Emits `mm:MusicEnsembleMembership` + `schema:Role` (dates & attributes); **lazy-create instruments** in `50_instruments.ttl`; `ex:playsInstrument` |
| `fix_labels_to_names.py` | Copy `rdfs:label` → `schema:name` | any TTL → updated TTL | Idempotent; standardises naming |
| `filter_artist_genres.py` | Keep only `mm:hasGenre` pointing to genres that exist in `11_genre.ttl` | `20_artists.ttl`, `11_genre.ttl` → `20_artists.mainonly.ttl` | `--require-type` optional; trims dead links |
| `dedupe_works_by_mbid.py` | Collapse duplicate `Work_*` nodes sharing the same MB Work ID | `33_recordings_works.ttl` → patched TTL | Keeps canonical; rewires setlists/performances |
| `enrich_wikidata_wikipedia.py` | Add QIDs/links/descriptions/images from Wikidata/Wikipedia | core TTLs → `25_wikidata_enrich.ttl` | Caches; normalises `schema:` to http |
| `link_songfacts_from_wikidata.py` | Add Songfacts IDs/links from WD | core + `25_*` → `26_songfacts_links.ttl` | `--resolve-by-mbid` fallback; adds `schema:identifier` PVs and URLs |
| `link_genius_from_wikidata.py` | Add Genius IDs/links from WD | core + `25_*` → `26_genius_links.ttl` | P6218/P6361 (works) + P2373/P6351 (artists) |
| `link_youtube_from_wikidata.py` | Add YouTube videos (works) + channels (artists) | core + `25_*` → `26_youtube_links.ttl` | Uses `schema:VideoObject`; supports `--manual-csv` for hand-picked clips |
| `link_images_from_wikidata.py` | Add images/logos (Commons) | core + `25_*` → `26_wikidata_images.ttl` | Uses Special:FilePath; P18 and P154 |
| `dedupe_enrichment_triples.py` | Merge + dedupe link layers | `24/25/26_*.ttl` → `27_enrichment_dedup.ttl` | Collapses duplicate `schema:identifier` PVs and duplicate YouTube `VideoObject`s |
| `slice_external_links.py` | Produce **links-only** TTL (+ “removed” TTL + CSV) | `27_enrichment_dedup.ttl` → `27_external_links_only.ttl` + `27_removed_nonlink_content.ttl` + CSV | `--keep-descriptions`, `--desc-lang`, `--desc-max` options |

---

## Key decisions & challenges solved

- **Traceability over simplicity**  
  Modelled both **Works** (*songs*) and **Recordings**, and wire performances to **both**. This gives end-to-end provenance (setlist → work ↔ recording), even though queries are a bit more verbose.

- **Setlist duplication**  
  Early setlists produced duplicate `schema:ListItem`s. Added de-dupers and later deduped by **MB Work ID** to keep one canonical Work and one entry per position.

- **Membership modeling**  
  MB “member of band” relations carry instruments and flags (e.g., `original`). Represented these in `mm:MusicEnsembleMembership` + `schema:Role`, keep roleName (comma-joined instruments/roles), and **lazy-create** `ex:MusicInstrument` nodes (file: `50_instruments.ttl`) and connect via `ex:playsInstrument`.

- **Namespace pains**  
  - Mixed `https://schema.org` vs `http://schema.org` across sources → **normalise to http** in every script.  
  - Missing prefix (`owl:`) in a TTL caused a merge error →  *always bind* `owl`, `rdfs`, `xsd`.

- **Network quirks**  
  - MusicBrainz `NetworkError: EOF occurred in violation of protocol` → wrap calls with retries + backoff.  
  - `urllib3 NotOpenSSLWarning` on macOS LibreSSL is benign; pin `urllib3<2` or use a Python linked with modern OpenSSL.

- **Filtering to “main performers”**  
  Subset Artists/Performances/Works to those connected to the two Live Aid events (Wembley + Philadelphia), then cascade the filter to genres and debut albums to keep the KG manageable.

- **External links without scraping**  
  All external content is **link-only** (Songfacts, Genius, YouTube, Commons images). We store IDs with `schema:identifier` (`schema:PropertyValue`) and add canonical URLs via `schema:sameAs`/`schema:image`/`schema:video`. No copyrighted text is ingested.

- **Delinking option**  
  Created a **slicer** that emits a clean `27_external_links_only.ttl` while preserving all “removed” triples in `27_removed_nonlink_content.ttl`, so you can **toggle** what ships without losing data.

---

## Reproducibility & caches

- Linkers/enrichers cache WD EntityData & WDQS results under `kg/enrichment/cache/...`.  
- For a clean rebuild, delete the cache; otherwise keep it to speed up iterative runs.  
- Recommend **checking in** the final TTLs you plan to ship (e.g., `27_external_links_only.ttl`) and tagging a Git release.

---

## Quick SPARQL smoke tests

**Iconic performance structure (with setlists):**
```sparql
PREFIX ex: <http://wembrewind.live/ex#>
PREFIX schema: <http://schema.org/>
PREFIX mm: <https://w3id.org/polifonia/ontology/music-meta/>

SELECT ?performance ?artist ?workName ?rec
WHERE {
  ?performance a mm:LivePerformance ;
               schema:isPartOf ex:LiveAid1985 ;
               schema:performer ?artist ;
               mm:performedWork ?work .
  OPTIONAL { ?work schema:name ?workName }
  OPTIONAL { ?performance mm:recordedAs ?rec }
}
ORDER BY ?artist ?performance ?workName
```

**YouTube videos on works:**
```sparql
PREFIX schema: <http://schema.org/>
SELECT ?work ?video WHERE {
  ?work a schema:MusicComposition ; schema:video ?v .
  ?v a schema:VideoObject ; schema:url ?video .
  FILTER CONTAINS(STR(?video), "youtube.com/watch?v=")
}
```

**Songfacts links present:**
```sparql
PREFIX schema: <http://schema.org/>
SELECT ?s ?id ?url WHERE {
  ?s schema:identifier [ schema:propertyID "P5241" ; schema:value ?id ] ;
     schema:sameAs ?url .
  FILTER CONTAINS(STR(?url),"songfacts.com/facts")
}
```

---

## Release tips

- **Links-only KG:** merge your base TTLs + `27_external_links_only.ttl`.  
- **Want descriptions in the release?** Re-run slicer with `--keep-descriptions`.  
- **Need to prune a provider (e.g., remove Genius but keep YouTube)?** Extend the slicer with an `--exclude-domains` switch (we can provide one if needed).

---