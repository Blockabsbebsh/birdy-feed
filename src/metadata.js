const LATVIAN_ONLY_LETTERS = /[āēīķļņģ]/iu;

function normalizedName(value) {
  return String(value || "")
    .normalize("NFD")
    .replace(/\p{M}/gu, "")
    .toLocaleLowerCase("lt-LT")
    .replace(/[^\p{L}\p{N}]+/gu, " ")
    .trim();
}

function editDistance(left, right) {
  const previous = Array.from({ length: right.length + 1 }, (_, index) => index);
  for (let leftIndex = 1; leftIndex <= left.length; leftIndex++) {
    let diagonal = previous[0];
    previous[0] = leftIndex;
    for (let rightIndex = 1; rightIndex <= right.length; rightIndex++) {
      const above = previous[rightIndex];
      previous[rightIndex] = Math.min(
        previous[rightIndex] + 1,
        previous[rightIndex - 1] + 1,
        diagonal + (left[leftIndex - 1] === right[rightIndex - 1] ? 0 : 1),
      );
      diagonal = above;
    }
  }
  return previous[right.length];
}

export function isSuspiciousLithuanianName(lithuanian, latvian, english) {
  const lt = normalizedName(lithuanian);
  const lv = normalizedName(latvian);
  if (!lt || lt === normalizedName(english)) return true;
  if (LATVIAN_ONLY_LETTERS.test(lithuanian)) return true;
  if (!lv) return false;
  if (lt === lv) return true;

  // BirdNET occasionally puts a slightly misspelled Latvian name in the
  // Lithuanian slot, so equality alone does not catch the bad record.
  const allowedDistance = Math.max(1, Math.floor(Math.max(lt.length, lv.length) * 0.08));
  return editDistance(lt, lv) <= allowedDistance;
}

export function chooseLithuanianName(english, commonNames = {}) {
  const lithuanian = String(commonNames.lt || "").trim();
  const latvian = String(commonNames.lv || "").trim();
  return isSuspiciousLithuanianName(lithuanian, latvian, english)
    ? english
    : lithuanian;
}

export function safeWikipediaUrl(value) {
  try {
    const url = new URL(value);
    if (url.protocol !== "https:") return null;
    if (!/^(?:en|lt)\.wikipedia\.org$/i.test(url.hostname)) return null;
    return url.toString();
  } catch {
    return null;
  }
}
