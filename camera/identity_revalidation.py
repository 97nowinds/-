"""Fresh local face evidence can correct names without merging body tracks."""
import math

def can_replace_local_identity(target, identity, camera_id, now):
    from face_recovery import fresh_body
    confirmed = identity.get('_face_confirmed_monotonic')
    if (identity.get('identity_source') != 'face' or not identity.get('_face_revalidation')
            or identity.get('_face_camera_id') != camera_id or identity.get('_face_votes', 0) < 3
            or type(confirmed) not in (int, float) or not math.isfinite(confirmed)
            or not 0 <= now-confirmed <= 2 or not fresh_body(target, now, camera_id)):
        return False
    observation = target.get('camera_observations', {}).get(camera_id, target)
    local = observation.get('source_local_id', observation.get('local_id'))
    return local is not None and local == identity.get('_face_local_id')

def face_owned_by_body(face, body, peers=()):
    x, y, w, h = face
    cx, cy = x+w/2, y+h/2
    def inside_head(box):
        bx, by, bw, bh = box
        return bx <= cx <= bx+bw and by-.04*bh <= cy <= by+.64*bh
    return inside_head(body) and not any(inside_head(peer) for peer in peers)
