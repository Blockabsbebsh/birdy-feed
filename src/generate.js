import fs from "node:fs/promises";
import path from "node:path";
import * as tf from "@tensorflow/tfjs";
import cocoSsd from "@tensorflow-models/coco-ssd";
import sharp from "sharp";

const API_KEY = process.env.NUTHATCH_API_KEY;
if (!API_KEY) throw new Error("NUTHATCH_API_KEY is not configured");

const API_URL = "https://nuthatch.lastelm.software/v2/birds";
const OUTPUT_DIR = path.resolve("dist");
const BIRD_COUNT = 5;
const PAGE_SIZE = 100;
const SOURCE_WIDTH = 2600;
const PROBE_WIDTH = 640;
const JPEG_QUALITY = 88;
const ROTATION_MINUTES = Math.floor((24 * 60) / BIRD_COUNT);

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

async function chooseBirds() {
  const first = await fetchPage(1);
  const total = first.total ?? first.totalResults ?? first.count ?? PAGE_SIZE;
  const pageCount = Math.max(1, Math.ceil(total / PAGE_SIZE));
  const chosen = [];
  const seen = new Set();

  for (let attempt = 0; chosen.length < BIRD_COUNT && attempt < 30; attempt++) {
    const page = 1 + Math.floor(Math.random() * pageCount);
    const data = page === 1 && attempt === 0 ? first : await fetchPage(page);
    const candidates = (data.entities ?? []).filter(bird => bird.images?.length);
    if (!candidates.length) continue;
    const bird = candidates[Math.floor(Math.random() * candidates.length)];
    const imageUrl = bird.images[Math.floor(Math.random() * bird.images.length)];
    const key = `${bird.name}|${imageUrl}`;
    if (seen.has(key)) continue;
    seen.add(key);
    chosen.push({ name: bird.name, sciName: bird.sciName || "", imageUrl });
  }

  if (chosen.length !== BIRD_COUNT) {
    throw new Error(`Could only choose ${chosen.length} unique birds`);
  }
  return chosen;
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

function subjectBox(detection, sourceWidth, sourceHeight) {
  const [x, y, width, height] = detection.prediction.bbox;
  const scaleX = sourceWidth / detection.metadata.width;
  const scaleY = sourceHeight / detection.metadata.height;
  const padding = 0.4;
  const left = Math.max(0, (x - width * padding) * scaleX);
  const top = Math.max(0, (y - height * padding) * scaleY);
  const right = Math.min(sourceWidth, (x + width * (1 + padding)) * scaleX);
  const bottom = Math.min(sourceHeight, (y + height * (1 + padding)) * scaleY);
  return { left, top, width: right - left, height: bottom - top };
}

function fillBox(subject, sourceWidth, sourceHeight, targetWidth, targetHeight) {
  const aspect = targetWidth / targetHeight;
  let width = subject.width;
  let height = subject.height;
  if (width / height < aspect) width = height * aspect;
  else height = width / aspect;

  if (width > sourceWidth) {
    height *= sourceWidth / width;
    width = sourceWidth;
  }
  if (height > sourceHeight) {
    width *= sourceHeight / height;
    height = sourceHeight;
  }

  let cx = subject.left + subject.width / 2;
  let cy = subject.top + subject.height / 2 + height * 0.09;
  cx = Math.min(Math.max(cx, width / 2), sourceWidth - width / 2);
  cy = Math.min(Math.max(cy, height / 2), sourceHeight - height / 2);

  const left = cx - width / 2;
  const top = cy - height / 2;
  const insideWidth = Math.max(
    0,
    Math.min(subject.left + subject.width, left + width) - Math.max(subject.left, left)
  );
  const insideHeight = Math.max(
    0,
    Math.min(subject.top + subject.height, top + height) - Math.max(subject.top, top)
  );
  const coverage = (insideWidth * insideHeight) / (subject.width * subject.height);
  return { left, top, width, height, coverage };
}

async function attentionCrop(source, size) {
  return sharp(source)
    .resize(size.width, size.height, { fit: "cover", position: sharp.strategy.attention })
    .jpeg({ quality: JPEG_QUALITY, chromaSubsampling: "4:4:4" })
    .toBuffer();
}

function shouldPreserveWholeFrame(metadata, size) {
  const sourceAspect = metadata.width / metadata.height;
  const targetAspect = size.width / size.height;
  const retainedShare = sourceAspect < targetAspect
    ? sourceAspect / targetAspect
    : targetAspect / sourceAspect;
  return targetAspect > 1.8 && retainedShare < 0.72;
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

async function detectedImage(source, metadata, subject, size) {
  const crop = fillBox(subject, metadata.width, metadata.height, size.width, size.height);
  if (crop.coverage >= 0.93) {
    return sharp(source)
      .extract({
        left: Math.max(0, Math.round(crop.left)),
        top: Math.max(0, Math.round(crop.top)),
        width: Math.min(metadata.width, Math.max(1, Math.round(crop.width))),
        height: Math.min(metadata.height, Math.max(1, Math.round(crop.height))),
      })
      .resize(size.width, size.height, { fit: "fill" })
      .jpeg({ quality: JPEG_QUALITY, chromaSubsampling: "4:4:4" })
      .toBuffer();
  }

  const extracted = await sharp(source)
    .extract({
      left: Math.max(0, Math.round(subject.left)),
      top: Math.max(0, Math.round(subject.top)),
      width: Math.min(metadata.width, Math.max(1, Math.round(subject.width))),
      height: Math.min(metadata.height, Math.max(1, Math.round(subject.height))),
    })
    .resize(size.width, size.height, { fit: "inside", withoutEnlargement: true })
    .png()
    .toBuffer();
  const subjectMeta = await sharp(extracted).metadata();
  const backdrop = await sharp(source)
    .resize(size.width, size.height, { fit: "cover", position: sharp.strategy.attention })
    .blur(20)
    .modulate({ brightness: 0.55 })
    .toBuffer();
  return sharp(backdrop)
    .composite([{
      input: extracted,
      left: Math.round((size.width - subjectMeta.width) / 2),
      top: Math.round((size.height - subjectMeta.height) / 2),
    }])
    .jpeg({ quality: JPEG_QUALITY, chromaSubsampling: "4:4:4" })
    .toBuffer();
}

async function renderBird(model, bird, index) {
  console.log(`[${index + 1}/${BIRD_COUNT}] ${bird.name}`);
  const source = await downloadSource(bird.imageUrl);
  const metadata = await sharp(source).metadata();
  const detection = await detectSubject(model, source);
  const subject = detection
    ? subjectBox(detection, metadata.width, metadata.height)
    : null;
  const images = {};

  for (const [family, size] of Object.entries(SIZES)) {
    const filename = `bird-${index + 1}-${family}.jpg`;
    let output;
    if (shouldPreserveWholeFrame(metadata, size)) {
      console.log(`  ${family}: preserving the complete ${metadata.width}x${metadata.height} frame`);
      output = await wholeFrameLetterbox(source, size);
    } else {
      output = subject
        ? await detectedImage(source, metadata, subject, size)
        : await attentionCrop(source, size);
    }
    await fs.writeFile(path.join(OUTPUT_DIR, filename), output);
    images[family] = filename;
  }
  return { name: bird.name, sciName: bird.sciName, images };
}

await fs.rm(OUTPUT_DIR, { recursive: true, force: true });
await fs.mkdir(OUTPUT_DIR, { recursive: true });
await tf.ready();
console.log(`TensorFlow backend: ${tf.getBackend()}`);
const model = await cocoSsd.load({ base: "mobilenet_v2" });
const selected = await chooseBirds();
const birds = [];
for (let index = 0; index < selected.length; index++) {
  birds.push(await renderBird(model, selected[index], index));
}

const feed = {
  version: 1,
  generatedAt: new Date().toISOString(),
  rotationMinutes: ROTATION_MINUTES,
  birds,
};
await fs.writeFile(path.join(OUTPUT_DIR, "latest.json"), `${JSON.stringify(feed, null, 2)}\n`);
console.log(`Generated ${birds.length} birds and ${birds.length * Object.keys(SIZES).length} images`);

