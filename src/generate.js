import { createHash } from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import * as tf from "@tensorflow/tfjs";
import cocoSsd from "@tensorflow-models/coco-ssd";
import sharp from "sharp";
import { chooseLithuanianName, safeWikipediaUrl } from "./metadata.js";

const API_KEY = process.env.NUTHATCH_API_KEY;
if (!API_KEY) throw new Error("NUTHATCH_API_KEY is not configured");

const API_URL = "https://nuthatch.lastelm.software/v2/birds";
const BIRDNET_API_URL = "https://birdnet.cornell.edu/taxonomy/api/species";
const OUTPUT_DIR = path.resolve("dist");
const BIRD_COUNT = 5;
const PAGE_SIZE = 100;
const SOURCE_WIDTH = 2600;
const PROBE_WIDTH = 640;
const JPEG_QUALITY = 88;
const ROTATION_MINUTES = Math.floor((24 * 60) / BIRD_COUNT);
const INDEX_HTML = "<!doctype html><html lang=\"en\"><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>Birdy feed</title><body><h1>Birdy feed</h1><p>Current widget data: <a href=\"./latest.json\">latest.json</a></p></body></html>\n";

const SIZES = {
  small: { width: 310, height: 310 },
  medium: { width: 658, height: 310 },
  large: { width: 658, height: 690 },
};

const FALLBACK_CLASSES = new Set([
  "kite", "airplane", "cat", "dog", "sheep", "cow", "teddy bear",
]);

function wsrvUrl(source, parameters = {}) {
  const query = new URLSearchParams({
    url: source,
    output: "jpg",
    q: String(JPEG_QUALITY),
    ...Object.fromEntries(
      Object.entries(parameters).map(([key, value]) => [key, String(value)])
    ),
  });
  return `https://wsrv.nl/?${query}`;
}

async function fetchOk(url, options = {}) {
  const response = await fetch(url, { signal: AbortSignal.timeout(30_000), ...options });
  if (!response.ok) throw new Error(`${response.status} ${response.statusText} for ${url}`);
  return response;
}

async function fetchPage(page) {
  const url = `${API_URL}?hasImg=true&pageSize=${PAGE_SIZE}&page=${page}`;
  const response = await fetchOk(url, { headers: { "api-key": API_KEY } });
  return response.json();
}

function deterministicIndex(value, length) {
  const digest = createHash("sha256").update(value).digest();
  return digest.readUInt32BE(0) % length;
}

async function chooseBirds(now = new Date()) {
  const first = await fetchPage(1);
  const reportedTotal = Number(first.total ?? first.totalResults ?? first.count);
  const pageCount = Number.isFinite(reportedTotal)
    ? Math.max(1, Math.ceil(reportedTotal / PAGE_SIZE))
    : 1;
  const remainingPages = await Promise.all(
    Array.from({ length: pageCount - 1 }, (_, index) => fetchPage(index + 2))
  );
  const records = [first, ...remainingPages].flatMap(page => page.entities ?? []);
  const byScientificName = new Map();

  for (const bird of records) {
    const sciName = String(bird.sciName || "").trim();
    const name = String(bird.name || "").trim();
    const images = [...new Set(
      (bird.images ?? []).filter(url => typeof url === "string" && url.length)
    )];
    if (!sciName || !name || !images.length) continue;

    const key = sciName.toLocaleLowerCase("en-US");
    const existing = byScientificName.get(key);
    if (existing) {
      existing.images = [...new Set([...existing.images, ...images])];
    } else {
      byScientificName.set(key, { name, sciName, images });
    }
  }

  const pool = [...byScientificName.values()].sort((a, b) => {
    const aHash = createHash("sha256").update(a.sciName).digest("hex");
    const bHash = createHash("sha256").update(b.sciName).digest("hex");
    return aHash.localeCompare(bHash);
  });
  console.log(
    `Nuthatch image-bearing records: ${Number.isFinite(reportedTotal) ? reportedTotal : "unknown"}`
  );
  console.log(`Birdy usable unique species with images: ${pool.length}`);

  if (pool.length < BIRD_COUNT) {
    throw new Error(`Only ${pool.length} unique image-bearing species are available`);
  }

  const utcDay = Math.floor(now.getTime() / 86_400_000);
  const offset = (utcDay * BIRD_COUNT) % pool.length;
  return Array.from({ length: BIRD_COUNT }, (_, index) => {
    const bird = pool[(offset + index) % pool.length];
    const imageIndex = deterministicIndex(`${utcDay}|${bird.sciName}`, bird.images.length);
    return {
      name: bird.name,
      sciName: bird.sciName,
      imageUrl: bird.images[imageIndex],
    };
  });
}

async function fetchBirdNetMetadata(scientificName) {
  const fields = "scientific_name,common_names,wikipedia_urls";
  const url = `${BIRDNET_API_URL}/${encodeURIComponent(scientificName)}` +
    `?locale=en,lt,lv&fields=${encodeURIComponent(fields)}`;
  const response = await fetchOk(url, {
    headers: { "User-Agent": "Birdy-Feed/1.0" },
  });
  return response.json();
}

async function workingWikipediaUrl(value) {
  const candidate = safeWikipediaUrl(value);
  if (!candidate) return null;
  try {
    const response = await fetch(candidate, {
      method: "HEAD",
      redirect: "follow",
      signal: AbortSignal.timeout(15_000),
      headers: { "User-Agent": "Birdy-Feed/1.0" },
    });
    if (!response.ok) return null;
    return safeWikipediaUrl(response.url) || candidate;
  } catch (error) {
    console.warn(`Could not verify Wikipedia URL ${candidate}: ${error.message}`);
    return null;
  }
}

async function localizeBird(bird) {
  try {
    const metadata = await fetchBirdNetMetadata(bird.sciName);
    const nameLt = chooseLithuanianName(bird.name, metadata.common_names);
    if (nameLt === bird.name) {
      console.warn(`No trustworthy Lithuanian name for ${bird.sciName}; using ${bird.name}`);
    }

    const entries = await Promise.all(
      ["en", "lt"].map(async language => {
        const url = await workingWikipediaUrl(metadata.wikipedia_urls?.[language]);
        return [language, url];
      }),
    );
    const wikipediaUrls = Object.fromEntries(entries.filter(([, url]) => url));
    return { ...bird, nameLt, wikipediaUrls };
  } catch (error) {
    console.warn(`Could not fetch BirdNET metadata for ${bird.sciName}: ${error.message}`);
    return { ...bird, nameLt: bird.name, wikipediaUrls: {} };
  }
}

async function downloadSource(url) {
  const response = await fetchOk(wsrvUrl(url, { w: SOURCE_WIDTH }));
  return Buffer.from(await response.arrayBuffer());
}

async function detectSubject(model, source) {
  const probe = await sharp(source)
    .resize({ width: PROBE_WIDTH, withoutEnlargement: true })
    .jpeg({ quality: 90 })
    .toBuffer();
  const metadata = await sharp(probe).metadata();
  const decoded = await sharp(probe)
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  const pixels = new Uint8Array(
    decoded.data.buffer,
    decoded.data.byteOffset,
    decoded.data.byteLength
  );
  const tensor = tf.tensor3d(
    pixels,
    [decoded.info.height, decoded.info.width, decoded.info.channels]
  );
  try {
    const predictions = await model.detect(tensor, 40, 0.1);
    const birds = predictions.filter(p => p.class === "bird" && p.score >= 0.15);
    if (birds.length) return { prediction: birds.sort((a, b) => b.score - a.score)[0], metadata };
    const impostors = predictions
      .filter(p => FALLBACK_CLASSES.has(p.class) && p.score >= 0.35)
      .sort((a, b) => b.score - a.score);
    return impostors.length ? { prediction: impostors[0], metadata } : null;
  } finally {
    tensor.dispose();
  }
}

function detectedBox(detection, sourceWidth, sourceHeight) {
  const [x, y, width, height] = detection.prediction.bbox;
  const scaleX = sourceWidth / detection.metadata.width;
  const scaleY = sourceHeight / detection.metadata.height;
  const left = Math.max(0, x * scaleX);
  const top = Math.max(0, y * scaleY);
  const right = Math.min(sourceWidth, (x + width) * scaleX);
  const bottom = Math.min(sourceHeight, (y + height) * scaleY);
  return { left, top, width: right - left, height: bottom - top };
}

function cropDimensions(subject, aspect, padding) {
  const paddedWidth = subject.width * (1 + 2 * padding);
  const paddedHeight = subject.height * (1 + 2 * padding);
  const width = Math.max(paddedWidth, paddedHeight * aspect);
  return { width, height: width / aspect };
}

function placeAxis(desired, subjectStart, subjectEnd, cropSize, frameSize) {
  const minimum = Math.max(0, subjectEnd - cropSize);
  const maximum = Math.min(subjectStart, frameSize - cropSize);
  if (minimum > maximum + 0.01) return null;
  return Math.min(Math.max(desired, minimum), maximum);
}

function cropForSubject(subject, sourceWidth, sourceHeight, aspect) {
  // Prefer comfortable breathing room. If the source is already tightly
  // framed, reduce only the optional padding before conceding to letterboxing.
  let dimensions = cropDimensions(subject, aspect, 0.25);
  if (dimensions.width > sourceWidth || dimensions.height > sourceHeight) {
    const tight = cropDimensions(subject, aspect, 0);
    if (tight.width > sourceWidth + 0.01 || tight.height > sourceHeight + 0.01) {
      return null;
    }
    // Find the most breathing room this particular source can support.
    let low = 0;
    let high = 0.25;
    for (let attempt = 0; attempt < 12; attempt++) {
      const middle = (low + high) / 2;
      const candidate = cropDimensions(subject, aspect, middle);
      if (candidate.width <= sourceWidth && candidate.height <= sourceHeight) low = middle;
      else high = middle;
    }
    dimensions = cropDimensions(subject, aspect, low);
  }

  const cx = subject.left + subject.width / 2;
  const cy = subject.top + subject.height / 2;
  const desiredLeft = cx - dimensions.width / 2;
  // Put the subject slightly above center to leave room for the title.
  const desiredTop = cy + dimensions.height * 0.08 - dimensions.height / 2;
  const left = placeAxis(
    desiredLeft,
    subject.left,
    subject.left + subject.width,
    dimensions.width,
    sourceWidth
  );
  const top = placeAxis(
    desiredTop,
    subject.top,
    subject.top + subject.height,
    dimensions.height,
    sourceHeight
  );
  if (left === null || top === null) return null;
  return { left, top, width: dimensions.width, height: dimensions.height };
}

function integerCrop(box, metadata) {
  const left = Math.max(0, Math.floor(box.left));
  const top = Math.max(0, Math.floor(box.top));
  const right = Math.min(metadata.width, Math.ceil(box.left + box.width));
  const bottom = Math.min(metadata.height, Math.ceil(box.top + box.height));
  return { left, top, width: right - left, height: bottom - top };
}

async function attentionCrop(source, size) {
  return sharp(source)
    .resize(size.width, size.height, { fit: "cover", position: sharp.strategy.attention })
    .jpeg({ quality: JPEG_QUALITY, chromaSubsampling: "4:4:4" })
    .toBuffer();
}

async function wholeFrameLetterbox(source, size) {
  const foreground = await sharp(source)
    .resize(size.width, size.height, { fit: "inside", withoutEnlargement: true })
    .png()
    .toBuffer();
  const foregroundMeta = await sharp(foreground).metadata();
  const backdrop = await sharp(source)
    .resize(size.width, size.height, { fit: "cover", position: "centre" })
    .blur(20)
    .modulate({ brightness: 0.55 })
    .toBuffer();
  return sharp(backdrop)
    .composite([{
      input: foreground,
      left: Math.round((size.width - foregroundMeta.width) / 2),
      top: Math.round((size.height - foregroundMeta.height) / 2),
    }])
    .jpeg({ quality: JPEG_QUALITY, chromaSubsampling: "4:4:4" })
    .toBuffer();
}

async function detectedImage(source, metadata, subject, size, cropAspect) {
  const crop = cropForSubject(subject, metadata.width, metadata.height, cropAspect);
  if (crop) {
    const extraction = integerCrop(crop, metadata);
    return sharp(source)
      .extract(extraction)
      .resize(size.width, size.height, { fit: "fill" })
      .jpeg({ quality: JPEG_QUALITY, chromaSubsampling: "4:4:4" })
      .toBuffer();
  }
  console.log(`  ${size.width}x${size.height}: bird cannot fit safely; preserving the full frame`);
  return wholeFrameLetterbox(source, size);
}

async function renderBird(model, bird, index) {
  console.log(`[${index + 1}/${BIRD_COUNT}] ${bird.name}`);
  console.log(`  Lithuanian: ${bird.nameLt}`);
  const source = await downloadSource(bird.imageUrl);
  const metadata = await sharp(source).metadata();
  const detection = await detectSubject(model, source);
  const subject = detection
    ? detectedBox(detection, metadata.width, metadata.height)
    : null;
  if (detection) {
    console.log(
      `  detected ${detection.prediction.class} at ${(detection.prediction.score * 100).toFixed(1)}%` +
      ` · box ${Math.round(subject.width)}x${Math.round(subject.height)}`
    );
  } else {
    console.log("  no usable detection; using attention crops");
  }
  const images = {};

  for (const [family, size] of Object.entries(SIZES)) {
    // Small and large deliberately use identical square crop coordinates.
    // Medium gets its own wide crop to retain more of the natural photograph.
    const cropAspect = family === "medium" ? size.width / size.height : 1;
    const output = subject
      ? await detectedImage(source, metadata, subject, size, cropAspect)
      : await attentionCrop(source, size);
    const digest = createHash("sha256").update(output).digest("hex").slice(0, 12);
    const filename = `bird-${index + 1}-${family}-${digest}.jpg`;
    await fs.writeFile(path.join(OUTPUT_DIR, filename), output);
    images[family] = filename;
  }
  return {
    name: bird.name,
    nameLt: bird.nameLt,
    sciName: bird.sciName,
    wikipediaUrls: bird.wikipediaUrls,
    images,
  };
}

await fs.rm(OUTPUT_DIR, { recursive: true, force: true });
await fs.mkdir(OUTPUT_DIR, { recursive: true });
await tf.ready();
console.log(`TensorFlow backend: ${tf.getBackend()}`);
const model = await cocoSsd.load({ base: "mobilenet_v2" });
const selected = await chooseBirds();
const localized = await Promise.all(selected.map(localizeBird));
const birds = [];
for (let index = 0; index < localized.length; index++) {
  birds.push(await renderBird(model, localized[index], index));
}

const feed = {
  version: 2,
  generatedAt: new Date().toISOString(),
  rotationMinutes: ROTATION_MINUTES,
  birds,
};
await fs.writeFile(path.join(OUTPUT_DIR, "latest.json"), `${JSON.stringify(feed, null, 2)}\n`);
await fs.writeFile(path.join(OUTPUT_DIR, "index.html"), INDEX_HTML);
console.log(`Generated ${birds.length} birds and ${birds.length * Object.keys(SIZES).length} images`);

