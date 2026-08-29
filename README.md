# Birdy feed

Backend for the Birdy Scriptable widget. A daily GitHub Actions workflow picks
five birds, detects and crops the subject with TensorFlow COCO-SSD, and publishes
the current JSON feed and widget-sized JPEGs to the generated `feed` branch.

The generated branch is force-replaced on every successful run, so previous
photos aren't retained in its Git history.

## Configuration

The repository Actions secret `NUTHATCH_API_KEY` must contain the Nuthatch API
key. The secret value must never be committed to this repository.

Run **Generate daily bird feed** manually after initial setup to create the
first feed. Scheduled generation runs daily at 04:17 UTC.

