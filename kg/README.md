# Live Aid KG (Modular Layout)

This folder contains the modularized Turtle files for the Live Aid knowledge graph.

## Files
- 00_prefixes.ttl — Common namespace prefixes.
- 10_core_entities.ttl — Events, venues, organizations (e.g., LiveAid1985, WembleyStadium).
- 20_artists.ttl — Artist and group instances (schema:Person, schema:MusicGroup).
- 30_performances.ttl — mm:LivePerformance nodes and cross-links (event, artist, venue, instruments, conditions).
- 40_setlists_songs.ttl — Setlists as schema:ItemList and song entries via schema:itemListElement.
- 50_instruments.ttl — ex:MusicInstrument instances (ElectricGuitar, Piano, etc.).
- 60_reviews.ttl — schema:Review instances and their links.
- 70_conditions.ttl — schema:MedicalCondition instances and any condition-related facts.
- 80_provenance.ttl — schema:isBasedOn, schema:citation, and similar provenance metadata.
- 90_shacl_shapes.ttl — Validation rules (SHACL).

## Build
`liveaid_instances_master.ttl` is a concatenation of the module files for convenience.

## Notes
- Keep schema/ontology terms (classes/properties) in the `schema/` folder, not here.
- Add new data to the specific module file matching its theme.
- Run SHACL on the master TTL before merging enrichment outputs.
