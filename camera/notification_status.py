"""Read-only readiness of the group notification chain; never expose secrets."""
from feishu_app import EVIDENCE_CREDENTIALS


def notification_status(runtime):
    service = runtime.service
    with service.lock:
        groups = [{
            'id': route['id'],
            'audience': route.get('audience', route['id']),
            'event_types': list(route['event_types']),
            'webhook_configured': bool(service.env.get(route.get('webhook_env', ''))),
            'signature_configured': bool(service.env.get(route.get('secret_env', ''))),
        } for route in service.config['routes']]
        outbox = {row[0]: row[1] for row in service.db.execute(
            'SELECT state,count(*) FROM outbox GROUP BY state')}
        producer = any(p['role'] == 'producer' and bool(service.env.get(p['token_env']))
                       for p in service.config.get('principals', []))
        silent = list(service.config.get('silent_events', []))
        rules = dict(service.config.get('ppe_rules', {}))
    with runtime.registry.connect() as db:
        snapshots = db.execute('SELECT count(*) FROM snapshots WHERE expires>?',
                               (runtime.registry.wall(),)).fetchone()[0]
    evidence = service.evidence is not None and EVIDENCE_CREDENTIALS.exists()
    blockers = []
    if not producer:
        blockers.append('producer_token_missing')
    if not snapshots:
        blockers.append('no_fresh_snapshot_index')
    if not evidence:
        blockers.append('evidence_upload_not_configured')
    if not any(g['webhook_configured'] and 'safety.no_mask' in g['event_types']
               and g['audience'] == 'experimenters' for g in groups):
        blockers.append('experimenters_ppe_route_missing')
    return {
        'groups': groups, 'silent_events': silent, 'ppe_rules': rules,
        'outbox_counts': outbox,
        'evidence_upload_configured': evidence,
        'producer_auth_configured': producer,
        'fresh_snapshot_count': snapshots,
        'require_authority_snapshot': bool(service.config.get('require_authority_snapshot')),
        'member_polling_configured': runtime.server.member_client is not None,
        'snapshot_publisher_enabled': runtime.server.live_publisher is not None,
        'poll_status': runtime.last_poll_status,
        'readiness_blockers': blockers,
        'meaning': 'Configuration and local readiness only; not proof of delivery or detection accuracy.',
    }
