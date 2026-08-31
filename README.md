# Birdy feed

Backend for the Birdy widgets. A daily GitHub Actions workflow picks five
birds, detects and crops the subject with TensorFlow COCO-SSD, and publishes
the current JSON feed and widget-sized JPEGs as a GitHub Pages deployment.
Each bird keeps its English name and includes an optional Lithuanian `nameLt`
looked up from eBird's public Lithuanian taxonomy using its scientific name.

The generated files are uploaded as a Pages artifact rather than committed to
a branch, so daily output does not accumulate in Git history. Image filenames
contain a content hash to prevent clients and CDNs from reusing a stale image.

## Public feed

- Feed: <https://blockabsbebsh.github.io/birdy-feed/latest.json>
- Site: <https://blockabsbebsh.github.io/birdy-feed/>

The generator loads the complete Nuthatch image-bearing pool, deduplicates it
by scientific name, and advances through a stable daily rotation. A species is
not selected again until the rotation has covered the available pool. Each run
logs Nuthatch's reported image-bearing record count and Birdy's usable unique
species count.

Small and large outputs share the same square crop around the detected bird.
Medium gets a separate wide crop. A blurred full-frame fallback is used only
when the detected bird itself cannot fit safely inside the requested aspect
ratio.

## Configuration

1. In repository **Settings → Pages**, set **Source** to **GitHub Actions**.
2. Set the Actions secret `NUTHATCH_API_KEY` to the Nuthatch API key. The
   secret value must never be committed to this repository.
3. Run **Generate daily bird feed** manually for the first deployment.

The eBird taxonomy endpoint is public and does not require an API key. If a Lithuanian
lookup fails or has no match, `nameLt` falls back to the English `name`, so feed
generation and older clients continue to work.

Scheduled generation runs daily at 04:17 UTC. The legacy `feed` branch is no
longer updated and can be deleted after every client has moved to the Pages URL.
