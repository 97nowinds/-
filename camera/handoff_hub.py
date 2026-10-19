"""Optional single-hub topology, separate from published map calibration."""
from copy import deepcopy
import json
import math
from pathlib import Path


def load_handoff_transitions(floor_config, policy_path=None):
    transitions = deepcopy(floor_config.get("camera_transitions", []))
    if policy_path is None or not Path(policy_path).exists():
        return transitions
    policy = json.loads(Path(policy_path).read_text(encoding="utf-8"))
    if not policy.get("enabled", False):
        return transitions
    cameras = set(policy["cameras"])
    minimum = float(policy.get("minimum_body_overlap", .35))
    if len(cameras) < 2 or not 0 < minimum <= 1 or not policy.get("hub_id"):
        raise ValueError("Invalid handoff hub policy")
    regions = {camera: [] for camera in cameras}
    for transition in transitions:
        for route in [transition] + transition.get("image_handoff_routes", []):
            for camera, gates in route.get("image_handoff_gates", {}).items():
                if camera not in regions:
                    continue
                for gate in gates.values():
                    for polygon in ([gate["points"]] if "points" in gate else []) + gate.get("additional_polygons", []):
                        if polygon not in regions[camera]:
                            regions[camera].append(polygon)
    if any(not regions[camera] for camera in cameras):
        raise ValueError("Every hub camera needs existing annotated image regions")
    for transition in transitions:
        pair = (transition.get("from"), transition.get("to"))
        if not all(camera in cameras for camera in pair):
            continue
        previous_routes = [transition] + transition.get("image_handoff_routes", [])
        overlap_minimum = min([float(route["temporal_overlap_min_similarity"])
            for route in previous_routes if "temporal_overlap_min_similarity" in route]
            + [float(transition.get("simultaneous_reid_threshold", .75))])
        transition.pop("image_handoff_routes", None)
        transition["image_handoff_gates"] = {
            camera: {direction: {"points": regions[camera][0],
                                  "additional_polygons": regions[camera][1:],
                                  "minimum_body_overlap": minimum}
                     for direction in ("entry", "exit")}
            for camera in pair}
        transition.update(hub_id=policy["hub_id"], hub_requires_confirmation=True,
                          simultaneous_overlap_validated=True,
                          temporal_overlap_min_similarity=overlap_minimum)
    return transitions


def body_overlap_ratio(box, polygon):
    """Area of an observed normalized body rectangle inside one polygon."""
    if box is None or len(box) != 4 or not all(math.isfinite(float(v)) for v in box):
        return 0.0
    x, y, width, height = map(float, box)
    if width <= 0 or height <= 0:
        return 0.0
    vertices = [tuple(map(float, point)) for point in polygon]
    for axis, boundary, keep_greater in ((0, x, True), (0, x+width, False),
                                         (1, y, True), (1, y+height, False)):
        if not vertices:
            return 0.0
        clipped = []
        previous = vertices[-1]
        inside_previous = previous[axis] >= boundary if keep_greater else previous[axis] <= boundary
        for current in vertices:
            inside = current[axis] >= boundary if keep_greater else current[axis] <= boundary
            if inside != inside_previous:
                delta = current[axis]-previous[axis]
                ratio = (boundary-previous[axis])/delta
                clipped.append(tuple(previous[k]+ratio*(current[k]-previous[k]) for k in (0, 1)))
            if inside:
                clipped.append(current)
            previous, inside_previous = current, inside
        vertices = clipped
    area = abs(sum(a[0]*b[1]-b[0]*a[1] for a, b in zip(vertices, vertices[1:]+vertices[:1])))/2
    return min(1.0, area/(width*height))
