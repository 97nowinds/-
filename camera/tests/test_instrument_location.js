const test = require("node:test");
const assert = require("node:assert/strict");
const {apply, evaluate} = require("../static/instrument_location.js");

const now = Date.parse("2026-09-24T13:00:00Z");
const person = {
  track_id: "person_7", global_track_id: "person_7", global_track_ids: ["person_7"],
  x: 4, y: 2, localization_quality: "medium", map_visible: true,
};
const anchor = {instrument_id: "centrifuge_1", name: "离心机", x_m: 9, y_m: 5, confidence: 1};
const event = {
  global_id: "person_7", instrument_id: "centrifuge_1", interaction_label: "正在使用",
  interaction_confidence: 0.98, observed_at: "2026-09-24T12:59:59Z",
};

test("unmatched exact result cannot borrow the raw track id for map fusion", () => {
  const result = evaluate(person, {...event, track_id: person.track_id,
    handoff_match_status: "unmatched"}, anchor, now);
  assert.equal(result, null);
});

test("matched result uses resolved global id rather than camera local track id", () => {
  assert.notEqual(evaluate(person, {...event, track_id: "7",
    handoff_match_status: "matched"}, anchor, now), null);
  assert.equal(evaluate(person, {...event, global_id: "other-global", track_id: person.track_id,
    handoff_match_status: "matched"}, anchor, now), null);
});

test("confirmed instrument use can override an approximate map point", () => {
  const floorMap = {width_m: 12, height_m: 7, people: [person]};
  const result = apply(floorMap, [event], [anchor], now).people[0];
  assert.equal(result.position_source, "instrument_anchor");
  assert.deepEqual([result.x, result.y], [9, 5]);
  assert.deepEqual(result.visual_position, {x: 4, y: 2});
  assert.equal(floorMap.people[0].x, 4);
});

test("unmatched identity, stale event, and weak confidence cannot move a point", () => {
  assert.equal(evaluate(person, {...event, global_id: "person_8"}, anchor, now), null);
  assert.equal(evaluate(person, {...event, observed_at: "2026-09-24T12:59:50Z"}, anchor, now), null);
  assert.equal(evaluate(person, {...event, interaction_confidence: 0.5}, anchor, now), null);
  assert.equal(evaluate({...person, association_conflict: true}, event, anchor, now), null);
});

test("large disagreement with high-quality map needs stronger evidence", () => {
  assert.equal(evaluate({...person, localization_quality: "high"}, {...event, interaction_confidence: 0.9}, anchor, now), null);
  assert.ok(evaluate(
    {...person, person_id: "staff_7", localization_quality: "high"},
    {...event, person_id: "staff_7", interaction_confidence: 1},
    anchor, now,
  ));
});

test("unknown instruments and out-of-map anchors do not move the point", () => {
  const map = {width_m: 12, height_m: 7, people: [person]};
  assert.equal(apply(map, [event], [], now).people[0].position_source, undefined);
  assert.equal(apply(map, [event], [{...anchor, x_m: 15}], now).people[0].position_source, undefined);
});

test("ambiguous identity cannot anchor two map points to one instrument", () => {
  const map = {width_m: 12, height_m: 7, people: [person, {...person, x: 5}]};
  const result = apply(map, [event], [anchor], now);
  assert.equal(result.people[0].position_source, undefined);
  assert.equal(result.people[1].position_source, undefined);
});
