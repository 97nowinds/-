const test = require("node:test");
const assert = require("node:assert/strict");
const {apply} = require("../static/instrument_location.js");

const now = Date.parse("2026-11-11T01:00:00Z");
const person = {
  track_id: "person_7", person_id: "staff_7", x: 6.55, y: 4.32,
  cameras: ["cam_2"], map_visible: true,
};
const anchor = {instrument_id: "instrument_003", name: "超声波清洗机", x_m: 6.82, y_m: 3.96, fixture_id: "bench"};
const event = {
  person_id: "staff_7", global_id: "person_7", camera_id: "cam_2", instrument_id: "instrument_003",
  is_interacting: true, observed_at: "2026-11-11T00:59:59Z", expires_at: "2026-11-11T01:00:04Z",
};
const floor = {
  width_m: 12, height_m: 7, people: [person],
  fixtures: [{id: "bench", type: "bench", x: 1, y: 2.4, width: 8.7, height: 1.8}],
};
const locate = (changes = {}, events = [event], anchors = [anchor]) =>
  apply({...floor, people: [{...person, ...changes}]}, events, anchors, now).people[0];

test("fresh explicit interaction places a nearby person just outside the bench", () => {
  const result = locate();
  assert.equal(result.position_source, "instrument_anchor");
  assert.equal(result.x, 6.82);
  assert.ok(Math.abs(result.y - 4.32) < 1e-9);
  assert.deepEqual(result.visual_position, {x: 6.55, y: 4.32});
  assert.equal(floor.people[0].x, 6.55);
});

test("distance, bench interior, missing anchor and invisible tracking block correction", () => {
  for (const change of [{x: 6.05}, {y: 4.7}, {y: 3.95}, {map_visible: false},
      {localization_state: "held"}, {localization_state: "predicted"}, {association_conflict: true}]) {
    assert.equal(locate(change).position_source, undefined);
  }
  assert.equal(locate({}, [event], [{...anchor, x_m: 20}]).position_source, undefined);
});

test("expired results, negatives and conflicting identity or camera never bind", () => {
  for (const change of [{is_interacting: false}, {model_is_interacting: false},
      {observed_at: "2026-11-11T00:59:50Z"}, {expires_at: "2026-11-11T01:00:00Z"},
      {person_id: "staff_other"}, {camera_id: "cam_1"}, {handoff_match_status: "unmatched"},
      {person_id: null, global_id: null, track_id: "person_7"}]) {
    assert.equal(locate({}, [{...event, ...change}]).position_source, undefined);
  }
});

test("anonymous global identity must resolve to exactly one map point", () => {
  const anonymous = {...event, person_id: null};
  assert.equal(locate({person_id: null}, [anonymous]).instrument_id, anchor.instrument_id);
  const people = [person, {...person, x: 6.6}];
  assert.deepEqual(apply({...floor, people}, [anonymous], [anchor], now).people, people);
});

test("a numbered balance binds only if exactly one instance is in range", () => {
  const balance = {...anchor, instrument_id: "instrument_001"};
  const numbered = {...anchor, instrument_id: "instrument_001_map_2", x_m: 6.65};
  const use = {...event, instrument_id: balance.instrument_id};
  assert.equal(locate({}, [use], [balance, numbered]).position_source, undefined);
  assert.equal(locate({}, [use], [{...balance, x_m: 8}, numbered]).instrument_id, numbered.instrument_id);
});
