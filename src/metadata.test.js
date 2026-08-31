import assert from "node:assert/strict";
import test from "node:test";
import {
  chooseLithuanianName,
  isSuspiciousLithuanianName,
  safeWikipediaUrl,
} from "./metadata.js";

test("rejects the Latvian Brown-capped Rosy Finch name in the Lithuanian slot", () => {
  assert.equal(
    chooseLithuanianName("Brown-capped Rosy Finch", {
      lt: "brūnglavas žubīte",
      lv: "Brūngalvas žubīte",
    }),
    "Brown-capped Rosy Finch",
  );
});

test("keeps a distinct Lithuanian name", () => {
  assert.equal(
    chooseLithuanianName("Great Tit", { lt: "Didžioji zylė", lv: "Lielā zīlīte" }),
    "Didžioji zylė",
  );
});

test("rejects exact cross-language duplicates and Latvian-only letters", () => {
  assert.equal(isSuspiciousLithuanianName("Testinis paukštis", "testinis paukštis", "Bird"), true);
  assert.equal(isSuspiciousLithuanianName("Putniņš", "Putniņš", "Bird"), true);
});

test("only accepts English and Lithuanian Wikipedia article URLs", () => {
  assert.match(safeWikipediaUrl("https://lt.wikipedia.org/wiki/Did%C5%BEioji_zyl%C4%97"), /^https:/);
  assert.equal(safeWikipediaUrl("http://lt.wikipedia.org/wiki/Test"), null);
  assert.equal(safeWikipediaUrl("https://example.com/wiki/Test"), null);
});
