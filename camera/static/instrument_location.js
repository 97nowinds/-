/* Display-only fusion of confirmed instrument use with the MTMC floor map. */
const InstrumentLocation = (() => {
  const MAX_EVENT_AGE_MS = 5000;
  const MIN_INTERACTION_CONFIDENCE = 0.8;
  const ACCEPT_SCORE = 0.85;

  function number(value) {
    if (value == null || value === "") return null;
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }

  function eventTime(event) {
    const raw = event.observed_at || event.captured_at || event.occurred_at || event.timestamp;
    const parsed = typeof raw === "number" ? raw * 1000 : Date.parse(raw);
    return Number.isFinite(parsed) ? parsed : null;
  }

  function activeUse(event) {
    const label = String(event.interaction_type || event.interaction_label || event.action || event.status || "").toLowerCase();
    return /using|operating|使用|操作/.test(label) || event.using === true;
  }

  function identityScore(person, event) {
    if (event.handoff_match_status === "unmatched") return 0;
    if (event.handoff_match_status === "matched") {
      // Resolved global identity is authoritative; raw track_id stays a frame key.
      if (person.person_id && event.person_id && String(person.person_id) === String(event.person_id)) return 1;
      const aliases = new Set([person.track_id, person.global_track_id, ...(person.global_track_ids || [])].filter(Boolean).map(String));
      return event.global_id != null && aliases.has(String(event.global_id)) ? 0.9 : 0;
    }
    if (person.person_id && event.person_id && String(person.person_id) === String(event.person_id)) return 1;
    const aliases = new Set([person.track_id, person.global_track_id, ...(person.global_track_ids || [])].filter(Boolean).map(String));
    return [event.global_id, event.global_track_id, event.track_id].some((id) => id != null && aliases.has(String(id))) ? 0.9 : 0;
  }

  function evaluate(person, event, anchor, nowMs = Date.now()) {
    if (!person || !event || !anchor || !activeUse(event) || person.association_conflict) return null;
    const identity = identityScore(person, event);
    const confidence = number(event.interaction_confidence ?? event.confidence);
    const observedAt = eventTime(event);
    const x = number(anchor.x_m);
    const y = number(anchor.y_m);
    if (!identity || confidence == null || confidence < MIN_INTERACTION_CONFIDENCE ||
        confidence > 1 || observedAt == null || x == null || y == null || x < 0 || y < 0) return null;
    const ageMs = nowMs - observedAt;
    if (ageMs < -1000 || ageMs > MAX_EVENT_AGE_MS) return null;
    const anchorConfidence = number(anchor.confidence) ?? 1;
    if (anchorConfidence < 0 || anchorConfidence > 1) return null;
    const freshness = 1 - Math.max(0, ageMs) / MAX_EVENT_AGE_MS;
    const score = 0.5 * confidence + 0.2 * identity + 0.2 * freshness + 0.1 * anchorConfidence;
    const distance = Math.hypot(x - Number(person.x), y - Number(person.y));
    const required = person.localization_quality === "high" && distance > 2 ? 0.95 : ACCEPT_SCORE;
    if (score < required) return null;
    return {x, y, score: Math.round(score * 1000) / 1000, instrument_id: String(event.instrument_id), observed_at: observedAt};
  }

  function apply(floorMap, interactions, instruments, nowMs = Date.now()) {
    if (!floorMap) return floorMap;
    const people = floorMap.people || [];
    const anchors = new Map((instruments || []).filter((item) =>
      item?.instrument_id && number(item.x_m) != null && number(item.y_m) != null &&
      number(item.x_m) <= floorMap.width_m && number(item.y_m) <= floorMap.height_m
    ).map((item) => [String(item.instrument_id), item]));
    return {
      ...floorMap,
      people: people.map((person) => {
        let winner = null;
        for (const event of interactions || []) {
          if (people.filter((candidate) => identityScore(candidate, event) > 0).length !== 1) continue;
          const anchor = anchors.get(String(event?.instrument_id || ""));
          const decision = evaluate(person, event, anchor, nowMs);
          if (decision && (!winner || decision.score > winner.score)) winner = decision;
        }
        return winner ? {
          ...person,
          x: winner.x,
          y: winner.y,
          visual_position: {x: person.x, y: person.y},
          position_source: "instrument_anchor",
          position_score: winner.score,
          instrument_id: winner.instrument_id,
          instrument_name: anchors.get(winner.instrument_id)?.name || winner.instrument_id,
          zone: anchors.get(winner.instrument_id)?.zone || person.zone,
          map_visible: true,
        } : person;
      }),
    };
  }

  return {evaluate, apply};
})();
if (typeof module !== "undefined") module.exports = InstrumentLocation;
