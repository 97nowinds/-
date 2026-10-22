"""Validated PPE policy edits; rule changes are scoped, persisted and audited."""
import json
import math


def validate_policy(policy):
    if not isinstance(policy,dict) or set(policy)-{'ppe_rules','ppe_confirmation','ppe_preparation'}:
        raise ValueError('unknown policy fields')
    rules=policy.get('ppe_rules',{})
    if not isinstance(rules,dict) or set(rules)-{'mask','lab_coat','gloves'}:
        raise ValueError('unsupported PPE rule')
    for rule in rules.values():
        if (not isinstance(rule,dict) or set(rule)!={'enabled','minimum_confidence'}
                or type(rule['enabled']) is not bool or type(rule['minimum_confidence']) not in (int,float)
                or not math.isfinite(rule['minimum_confidence']) or not .5<=rule['minimum_confidence']<=1):
            raise ValueError('invalid PPE threshold')
    if 'ppe_preparation' in policy:
        preparation=policy['ppe_preparation']
        if (not isinstance(preparation,dict) or set(preparation)!=
                {'enabled','camera_ids','grace_seconds','end_on_confirmed_interaction'}
                or type(preparation['enabled']) is not bool
                or type(preparation['end_on_confirmed_interaction']) is not bool
                or not isinstance(preparation['camera_ids'],list) or not preparation['camera_ids']
                or any(not isinstance(c,str) or not c for c in preparation['camera_ids'])
                or type(preparation['grace_seconds']) is not int
                or not 10<=preparation['grace_seconds']<=600):
            raise ValueError('invalid PPE preparation policy')
    if 'ppe_confirmation' not in policy:return
    settings=policy['ppe_confirmation']
    fields={'minimum_frames','minimum_seconds','maximum_gap_seconds','escalation_seconds'}
    if not isinstance(settings,dict) or set(settings)!=fields:raise ValueError('invalid confirmation settings')
    if type(settings['minimum_frames']) is not int or not 2<=settings['minimum_frames']<=100:
        raise ValueError('require distinct frames')
    for key in fields-{'minimum_frames'}:
        value=settings[key]
        if type(value) not in (int,float) or not math.isfinite(value) or not 1<=value<=3600:
            raise ValueError('invalid confirmation time')
    if settings['escalation_seconds']<=settings['minimum_seconds']:raise ValueError('escalation must follow confirmation')


def authorize(service,actor):
    if actor['role']!='admin':raise PermissionError('maintenance administrator required')


def read_policy(service,actor):
    authorize(service,actor)
    with service.lock:
        return {key:service.config.get(key,{}) for key in ('ppe_rules','ppe_confirmation','ppe_preparation') if key in service.config}


def set_policy(service,actor,policy):
    authorize(service,actor)
    if not {'ppe_rules','ppe_confirmation'}<=set(policy):raise ValueError('complete policy required')
    validate_policy(policy)
    with service.lock,service.db:
        # Reset accumulated, unconfirmed observations on a rule change. Existing
        # incidents remain in the audit trail and require fresh recovery evidence.
        for row in service.db.execute('SELECT key,payload FROM ppe_states').fetchall():
            state=json.loads(row['payload'])
            if not state['emitted']:service.db.execute('DELETE FROM ppe_states WHERE key=?',(row['key'],))
            else:
                state.update(frames=[],continuous_since=None,escalation_since=None)
                service.db.execute('UPDATE ppe_states SET payload=? WHERE key=?',(json.dumps(state),row['key']))
        service.config={**service.config,**json.loads(json.dumps(policy))}
        persisted={k:service.config[k] for k in ('ppe_rules','ppe_confirmation','ppe_preparation') if k in service.config}
        service.db.execute('INSERT OR REPLACE INTO settings VALUES(?,?)',('ppe_policy',json.dumps(persisted)))
        service._audit(actor['id'],'policy.update',json.dumps(policy,sort_keys=True))
