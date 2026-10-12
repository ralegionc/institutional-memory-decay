# Data

This repository does not carry its raw inputs. The files listed below were present locally when the results in the README were produced, and were deleted on 2026-10-12 to reclaim **55.54 MB** of disk. Every one of them is a download from a public upstream source, so nothing unique was lost.

This file exists so that a re-download can be *checked* rather than assumed to match. It records the exact inventory, not just the source.

## What was removed

55.54 MB in 1 files.

| Path | Size | Files |
|---|---|---|
| `data` | 55.54 MB | 1 |

## Restoring it

An HTTP response cache for the Wikipedia revision and pageview pulls. Every derived parquet in `data/` was kept, so the decay fits stand without it.

## Inventory

`data_manifest.csv` carries every file with its exact byte length and a fingerprint. Files of 32 MiB or less have a full `sha256`. Larger files have `headtail256`: a sha256 over the first mebibyte, the last mebibyte and the exact byte length, which catches a truncated or altered download without rereading gigabytes.

| File | Size | Fingerprint |
|---|---|---|
| `data/http_cache.sqlite` | 55.54 MB | `headtail256:df9a92c2012e84df5caf58c3e4c1a79ecdde76a99350e0931d30b91ef799db5f` |

A source that is live rather than a fixed release will not reproduce these fingerprints. Where that applies it is said above.

