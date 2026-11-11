"""Claim durable delivery jobs, then perform network I/O outside the DB lock."""
import json
import uuid


def deliver(service):
    from management_notifications import route_accepts,stamp,feishu_payload
    from feishu_app import EVIDENCE_CREDENTIALS,FeishuAppClient,load_credentials
    now=service.wall()
    with service.lock,service.db:
        service.db.execute("UPDATE outbox SET state='pending' WHERE state='sending' AND lease_until<=?",(now,))
        row=service.db.execute('SELECT o.*,e.payload FROM outbox o JOIN events e ON e.id=o.event_id '
            "WHERE o.state='pending' AND o.due<=? ORDER BY o.priority DESC,o.due LIMIT 1",(now,)).fetchone()
        if not row or now-service.last_send<1:return False
        event=json.loads(row['payload'])
        route=next((r for r in service.config['routes'] if r['id']==row['route_id']),None)
        if event.get('severity')!='L3' and now-stamp(event['observed_at'])>service.config.get('delivery_max_age_seconds',300):
            service.db.execute("UPDATE outbox SET state='expired' WHERE event_id=? AND route_id=?",(row['event_id'],row['route_id']))
            return True
        allowed = (route and event['event_type'] not in service.config.get('silent_events', [])
                   and route_accepts(route, event))
        if event.get('warning_number'):
            allowed = allowed and service.incidents.ppe_open(
                event['incident_id'], before_red=event['severity'] != 'L3')
        if not allowed:
            service.db.execute("UPDATE outbox SET state='cancelled' WHERE event_id=? AND route_id=?",(row['event_id'],row['route_id']))
            return True
        url=service.env.get(route.get('webhook_env',''),'')
        if not url:
            service.db.execute('UPDATE outbox SET due=? WHERE event_id=? AND route_id=?',(now+30,row['event_id'],row['route_id']))
            return False
        claim=uuid.uuid4().hex
        changed=service.db.execute("UPDATE outbox SET state='sending',lease_until=?,claim=? WHERE event_id=? AND route_id=? AND state='pending'",
                        (now+60,claim,row['event_id'],row['route_id'])).rowcount
        if changed!=1:return False
        service.last_send=now
    try:
        if event['event_type'] == 'environment.flame':
            # Emergency text never waits for capture/upload. A durable worker
            # supplements every destination with the same three-camera album.
            payload=feishu_payload(service.message(event)+'\n三路现场快照随后补发；不代表三路均已检测到火焰。',
                service.env.get(route.get('secret_env',''),''),service.wall())
        elif event.get('warning_number') and event['severity'] == 'L3':
            # Escalation text must reach administrators even if image upload is unavailable.
            payload = feishu_payload(service.message(event)+'\n证据与处置记录：实验室通知管理 → 告警处置',
                                     service.env.get(route.get('secret_env', ''), ''), service.wall())
        elif event.get('evidence_id'):
            from alert_evidence import group_evidence_payload
            with service.image_lock:
                if service.evidence is None or not EVIDENCE_CREDENTIALS.exists():raise ValueError('evidence transport unavailable')
                revision=EVIDENCE_CREDENTIALS.stat().st_mtime_ns
                if service.image_client is None or service.image_credentials_revision!=revision:
                    service.image_client=FeishuAppClient(load_credentials(EVIDENCE_CREDENTIALS),wall=service.wall)
                    service.image_credentials_revision=revision
                client=service.image_client;app_id=client.credentials['app_id']
                with service.lock:
                    old=service.db.execute('SELECT image_key FROM evidence_uploads WHERE evidence_id=? AND app_id=?',
                        (event['evidence_id'],app_id)).fetchone()
                key=old['image_key'] if old else client.upload_image(service.evidence.read(event['evidence_id']))
                if not old:
                    with service.lock,service.db:
                        service.db.execute('INSERT OR REPLACE INTO evidence_uploads VALUES(?,?,?)',(event['evidence_id'],app_id,key))
            payload=group_evidence_payload(service.message(event),key,service.env.get(route.get('secret_env',''),''),service.wall(),
                title='实验室火焰报警｜证据快照' if event['event_type'].startswith('environment.') else '🟡 实验室防护警告｜证据快照')
        else:payload=feishu_payload(service.message(event),service.env.get(route.get('secret_env',''),''),service.wall())
        service.sender(url,payload)
        state,error='sent',None
    except Exception as exc:
        state='failed' if row['attempts']>=4 and event.get('severity')!='L3' else 'pending'
        error=type(exc).__name__
    with service.lock,service.db:
        service.db.execute('UPDATE outbox SET state=?,attempts=attempts+1,due=?,error=?,lease_until=0 '
            'WHERE event_id=? AND route_id=? AND claim=?',
            (state,service.wall()+min(300,2**(row['attempts']+1)),error,row['event_id'],row['route_id'],claim))
    return True
