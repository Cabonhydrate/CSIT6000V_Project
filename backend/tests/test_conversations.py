"""CRUD and cascade behavior on existing SQLite conversations."""
from uuid import uuid4
from fastapi.testclient import TestClient
from app import db
from app.main import app


def test_full_history_rename_delete_and_cascade():
    with TestClient(app) as client:
        first=client.post('/api/conversations',json={'title':'Old title'}).json()['id']
        second=client.post('/api/conversations',json={'title':'Another discussion'}).json()['id']
        assert first != second
        run_id=uuid4().hex
        with db.connect() as con:
            con.execute('INSERT INTO messages(id,conversation_id,role,content,citations,created_at) VALUES(?,?,?,?,?,?)',
                        (uuid4().hex,first,'user','What is CC?','[]',db.now()))
            con.execute('INSERT INTO runs(id,conversation_id,status,mode,question,paper_ids,created_at) VALUES(?,?,?,?,?,?,?)',
                        (run_id,first,'succeeded','basic','What is CC?','[]',db.now()))
        db.event(run_id,'run_completed',{'status':'succeeded'})

        listed=client.get('/api/conversations')
        assert listed.status_code==200
        conversations=listed.json()
        assert {c['id'] for c in conversations} == {first,second}
        assert next(c for c in conversations if c['id']==first)['message_count']==1
        assert all(c['updated_at'] for c in conversations)
        assert next(c for c in conversations if c['id']==first)['message_count']==1

        changed=client.patch(f'/api/conversations/{first}',json={'title':'  Paper terminology  '})
        assert changed.status_code==200
        assert changed.json()['title']=='Paper terminology'
        assert client.get(f'/api/conversations/{first}').json()['title']=='Paper terminology'
        assert client.patch(f'/api/conversations/{first}',json={'title':'  '}).status_code==400

        delete=client.delete(f'/api/conversations/{first}')
        assert delete.status_code==200
        assert client.get(f'/api/conversations/{first}').status_code==404
        assert db.one('SELECT id FROM messages WHERE conversation_id=?',(first,)) is None
        assert db.one('SELECT id FROM runs WHERE conversation_id=?',(first,)) is None
        assert db.one('SELECT id FROM run_events WHERE run_id=?',(run_id,)) is None
        assert client.get(f'/api/conversations/{second}').status_code==200


def test_active_run_prevents_delete():
    with TestClient(app) as client:
        conv=client.post('/api/conversations',json={'title':'Busy'}).json()['id']
        rid=uuid4().hex
        db.execute('INSERT INTO runs(id,conversation_id,status,mode,question,paper_ids,created_at) VALUES(?,?,?,?,?,?,?)',
                   (rid,conv,'running','adaptive','hello','[]',db.now()))
        res=client.delete(f'/api/conversations/{conv}')
        assert res.status_code==409
        assert client.get(f'/api/conversations/{conv}').status_code==200
        db.execute('UPDATE runs SET status=? WHERE id=?',('cancelled',rid))
        assert client.delete(f'/api/conversations/{conv}').status_code==200


def test_missing_conversation_management_404():
    with TestClient(app) as client:
        assert client.patch('/api/conversations/unknown',json={'title':'Updated'}).status_code==404
        assert client.delete('/api/conversations/unknown').status_code==404
