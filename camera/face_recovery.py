"""Registered identity recovery; never merges trajectories or uses map distance."""
import math


FRESH_SECONDS = 2.0


def fresh_body(target, now, camera_id=None):
    if not target:
        return False
    observations = target.get('camera_observations', {})
    if camera_id is not None:
        observation = observations.get(camera_id)
        if observation is None:
            observation = target if target.get('camera_id') == camera_id else {}
        times = [observation.get('last_seen')]
    else:
        times = [target.get('last_seen'), *(v.get('last_seen') for v in observations.values())]
    return any(isinstance(t, (int, float)) and math.isfinite(t)
               and 0 <= now-t <= FRESH_SECONDS for t in times)


def can_reclaim_identity(owner, target, identity, camera_id, now):
    """Require fresh multi-vote face evidence on this exact camera/local body."""
    if (identity.get('identity_source') != 'face'
            or identity.get('_face_camera_id') != camera_id
            or identity.get('_face_votes', 0) < 3
            or not fresh_body(target, now, camera_id)
            or fresh_body(owner, now)):
        return False
    confirmed = identity.get('_face_confirmed_monotonic')
    if not isinstance(confirmed, (int, float)) or not math.isfinite(confirmed):
        return False
    if not 0 <= now-confirmed <= FRESH_SECONDS:
        return False
    observed = target.get('camera_observations', {}).get(camera_id, target)
    local = observed.get('source_local_id', observed.get('local_id'))
    return local is not None and local == identity.get('_face_local_id')


def cached_identity_is_current(coordinator, identity, global_id):
    """A released worker cache must not publish or silently reclaim an old name."""
    cached = identity.get('global_track_id')
    seen = set()
    while cached in coordinator.redirects and cached not in seen:
        seen.add(cached)
        cached = coordinator.redirects[cached]
    return bool(cached == global_id and identity.get('person_id')
                and coordinator.person_locks.get(identity['person_id']) == global_id)
