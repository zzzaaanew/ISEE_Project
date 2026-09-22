"""Local dashboard: subprocess workers, bounded event cursors, actual inputs only."""
from __future__ import annotations
import argparse
import json
import re
import subprocess
import sys
import threading
import uuid
from datetime import datetime,timezone
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse,parse_qs
from .placement_policy import POLICIES
from .replay_runner import atomic_json

ROOT=Path(__file__).resolve().parents[1]


class ReplayController:
    def __init__(self,manifest,runs):
        self.manifest=Path(manifest).resolve();self.runs=Path(runs).resolve();self.runs.mkdir(parents=True,exist_ok=True)
        self.lock=threading.Lock();self.processes={}

    def path(self,run_id):
        if not re.fullmatch(r'run_[A-Za-z0-9_]+',run_id):raise ValueError('Invalid run ID')
        p=self.runs/run_id
        if not p.is_dir():raise FileNotFoundError(run_id)
        return p

    def status(self,run_id):
        p=self.path(run_id);file=p/'status.json'
        state=json.loads(file.read_text(encoding='utf-8')) if file.exists() else {'status':'starting','progress':0}
        proc=self.processes.get(run_id)
        if proc and proc.poll() is not None and state['status'] in ('starting','validating','running','stopping'):
            state={'status':'failed','error':f'Worker exited unexpectedly ({proc.returncode}); see worker.err.log','progress':state.get('progress',0)};atomic_json(file,state)
        if 'timestamp' not in state and 'time' in state and (p/'manifest.json').exists():
            m=json.loads((p/'manifest.json').read_text(encoding='utf-8'))
            from datetime import timedelta
            state['timestamp']=(datetime.fromisoformat(m['start_time'])+timedelta(seconds=state['time'])).isoformat()
            state['placement_policy']=m['placement_policy']
        return {'run_id':run_id,**state}

    def start(self,payload):
        policy=payload.get('policy','risk_prefer_packed');scheduler=payload.get('scheduler','Las')
        if policy not in POLICIES or scheduler not in ('Fifo','Las','Tiresias2Q'):raise ValueError('Unsupported policy or scheduler')
        if not self.manifest.exists():raise ValueError('Actual-data manifest is not ready. Complete input preparation first.')
        with self.lock:
            if any(p.poll() is None for p in self.processes.values()):raise ValueError('A replay is already running')
            ident='run_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]
            out=self.runs/ident;out.mkdir();atomic_json(out/'status.json',{'status':'starting','progress':0})
            with (out/'worker.log').open('wb') as stdout,(out/'worker.err.log').open('wb') as stderr:
                proc=subprocess.Popen([sys.executable,'-B','-u','-m','risklab.replay_runner','--manifest',str(self.manifest),'--output',str(out),'--policy',policy,'--scheduler',scheduler],cwd=ROOT,stdout=stdout,stderr=stderr,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            self.processes[ident]=proc
            return {'run_id':ident,'status':'starting','pid':proc.pid}

    def stop(self,ident):
        path=self.path(ident);(path/'stop.request').touch();return {'run_id':ident,'status':'stopping'}

    def events(self,ident,cursor,tail=False):
        if cursor<0:raise ValueError('Invalid cursor')
        path=self.path(ident)/'events.jsonl'
        if not path.exists():return {'events':[],'cursor':0}
        rows=[]
        with path.open('rb') as f:
            if cursor>path.stat().st_size:raise ValueError('Cursor exceeds log size')
            if tail:
                end=path.stat().st_size;begin=max(0,end-131072);f.seek(begin);data=f.read(end-begin)
                if begin:data=data.partition(b'\n')[2]
                lines=data.split(b'\n')[:-1]
                return {'events':[json.loads(line) for line in lines[-100:] if line],'cursor':end,'tail':True}
            f.seek(cursor)
            for _ in range(100):
                start=f.tell();line=f.readline(1024*1024)
                if not line or not line.endswith(b'\n'):f.seek(start);break
                rows.append(json.loads(line))
            return {'events':rows,'cursor':f.tell()}


def make_server(manifest,runs,port=8115):
    controller=ReplayController(manifest,runs)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def reply(self,code,value,kind='application/json; charset=utf-8'):
            data=json.dumps(value,ensure_ascii=False,allow_nan=False).encode() if kind.startswith('application/json') else value.encode()
            self.send_response(code);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)
        def do_GET(self):
            parsed=urlparse(self.path);parts=parsed.path.strip('/').split('/')
            try:
                if parsed.path=='/':return self.reply(200,(ROOT/'risklab/web/replay.html').read_text(encoding='utf-8'),'text/html; charset=utf-8')
                if parsed.path=='/config':return self.reply(200,{'ready':controller.manifest.exists(),'manifest':str(controller.manifest),'policies':list(POLICIES),'schedulers':['Fifo','Las','Tiresias2Q'],'runs':[p.name for p in sorted(controller.runs.glob('run_*'),reverse=True) if p.is_dir()][:50]})
                if parts[0]=='runs' and len(parts)>=2:
                    ident=parts[1]
                    if len(parts)==2:return self.reply(200,controller.status(ident))
                    if parts[2]=='events':return self.reply(200,controller.events(ident,int(parse_qs(parsed.query).get('cursor',['0'])[0]),parse_qs(parsed.query).get('tail',['0'])[0]=='1'))
                    if parts[2]=='summary':return self.reply(200,json.loads((controller.path(ident)/'summary.json').read_text(encoding='utf-8')))
                    if parts[2]=='report':return self.reply(200,(controller.path(ident)/'report.html').read_text(encoding='utf-8'),'text/html; charset=utf-8')
                self.reply(404,{'error':'Not found'})
            except FileNotFoundError:self.reply(404,{'error':'Artifact not available yet'})
            except (ValueError,KeyError) as exc:self.reply(400,{'error':str(exc)})
        def do_POST(self):
            try:
                # Browser requests must originate on this exact local origin.
                origin=self.headers.get('Origin')
                if origin and origin not in (f'http://127.0.0.1:{self.server.server_port}',f'http://localhost:{self.server.server_port}'):return self.reply(403,{'error':'Origin rejected'})
                if not self.headers.get('Content-Type','').startswith('application/json'):return self.reply(415,{'error':'JSON required'})
                length=int(self.headers.get('Content-Length','0'))
                if length<0 or length>4096:raise ValueError('Request too large')
                payload=json.loads(self.rfile.read(length) or b'{}');parts=urlparse(self.path).path.strip('/').split('/')
                if parts==['runs']:return self.reply(202,controller.start(payload))
                if len(parts)==3 and parts[0]=='runs' and parts[2]=='stop':return self.reply(202,controller.stop(parts[1]))
                self.reply(404,{'error':'Not found'})
            except FileNotFoundError:self.reply(404,{'error':'Run not found'})
            except (ValueError,KeyError,TypeError) as exc:self.reply(400,{'error':str(exc)})
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler);server.controller=controller
    return server


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,default=ROOT/'inputs/reportfaithful_30m_20260922/run_manifest.json');p.add_argument('--runs',type=Path,default=ROOT/'runs_actual');p.add_argument('--port',type=int,default=8115);p.add_argument('--no-browser',action='store_true');a=p.parse_args()
    server=make_server(a.manifest,a.runs,a.port);print(f'RiskLab: http://127.0.0.1:{server.server_port}',flush=True)
    if not a.no_browser:__import__('webbrowser').open(f'http://127.0.0.1:{server.server_port}')
    try:server.serve_forever()
    finally:server.server_close()


if __name__=='__main__':main()
