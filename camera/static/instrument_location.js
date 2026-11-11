/* Keep the displayed person near the instrument while live use is confirmed. */
const InstrumentLocation = (() => {
  const MAX_EVENT_AGE_MS = 5000;
  const MAX_ANCHOR_DISTANCE_M = 0.75;
  const MAX_MOVE_M = 0.60;
  const AISLE_OFFSET_M = 0.12;

  function identityMatches(person, event) {
    if (event.handoff_match_status === "unmatched") return false;
    if (person.person_id && event.person_id) {
      return String(person.person_id) === String(event.person_id);
    }
    const globalId = event.global_id ?? event.global_track_id;
    return globalId != null && [person.track_id, person.global_track_id, ...(person.global_track_ids || [])]
      .some(id => id != null && String(id) === String(globalId));
  }

  function activeUse(event, now) {
    if (event.is_interacting !== true || event.model_is_interacting === false || event.ended_at) return false;
    const confidence = event.interaction_confidence ?? event.confidence;
    if (confidence != null && (!Number.isFinite(confidence) || confidence < 0.8 || confidence > 1)) return false;
    const observed = Date.parse(event.observed_at || event.captured_at);
    const expires = Date.parse(event.expires_at);
    return now >= observed - 1000 && now - observed <= MAX_EVENT_AGE_MS && now < expires;
  }

  function nearbyPosition(person, anchor, floorMap) {
    const {x, y} = person;
    const {x_m: ax, y_m: ay} = anchor;
    if (![x, y, ax, ay].every(Number.isFinite) || Math.hypot(x - ax, y - ay) > MAX_ANCHOR_DISTANCE_M) return null;
    const bench = (floorMap.fixtures || []).find(item => item.id === anchor.fixture_id && item.type === "bench");
    if (!bench || ax < bench.x || ax > bench.x + bench.width ||
        ay < bench.y || ay > bench.y + bench.height) return null;
    const farEdge = bench.y + bench.height;
    // Use the edge facing the observed person, with a small offset into the aisle.
    const front = y >= farEdge ? farEdge + AISLE_OFFSET_M : y <= bench.y ? bench.y - AISLE_OFFSET_M : null;
    if (front == null || front < 0 || front > floorMap.height_m || ax < 0 || ax > floorMap.width_m ||
        Math.hypot(x - ax, y - front) > MAX_MOVE_M) return null;
    return {x: ax, y: front};
  }

  function apply(floorMap, interactions, equipment, now = Date.now()) {
    if (!floorMap) return floorMap;
    const people = floorMap.people || [];
    const proposals = new Map();
    for (const event of interactions || []) {
      if (!event.instrument_id || !event.camera_id || !activeUse(event, now)) continue;
      const matches = people.filter(person => identityMatches(person, event));
      if (matches.length !== 1) continue;
      const [person] = matches;
      if (person.map_visible !== true || person.association_conflict || person.handoff_waiting ||
          ["held", "predicted"].includes(person.localization_state) ||
          !person.cameras?.includes(event.camera_id)) continue;
      const candidates = [];
      for (const anchor of equipment || []) {
        if (anchor.instrument_id !== event.instrument_id &&
            !anchor.instrument_id?.startsWith(event.instrument_id + "_map_")) continue;
        const position = nearbyPosition(person, anchor, floorMap);
        if (position) candidates.push({anchor, position});
      }
      if (candidates.length !== 1) continue;
      const [candidate] = candidates;
      if (!proposals.has(person)) proposals.set(person, new Map());
      proposals.get(person).set(candidate.anchor.instrument_id, candidate);
    }
    return {
      ...floorMap,
      people: people.map(person => {
        const candidates = proposals.get(person);
        if (candidates?.size !== 1) return person;
        const [{anchor, position}] = candidates.values();
        if ([...proposals.values()].filter(options => options.has(anchor.instrument_id)).length !== 1) return person;
        return {
          ...person, ...position,
          visual_position: {x: person.x, y: person.y},
          position_source: "instrument_anchor",
          instrument_id: anchor.instrument_id,
          instrument_name: anchor.name,
        };
      }),
    };
  }

  return {apply};
})();
if (typeof module !== "undefined") module.exports = InstrumentLocation;
