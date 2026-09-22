import json
from pathlib import Path
import pandas as pd
from risklab.replay_runner import execute,atomic_json
from risklab.feeds import sha256
root=Path('runs_actual/policy_revision_20260922');root.mkdir(exist_ok=True)
manifest=Path('inputs/reportfaithful_30m_20260922/run_manifest.json');results=[]
for policy in ['risk_blind_packed','risk_mask_packed','risk_prefer_packed']:
 print('Running '+policy,flush=True);results.append(execute(manifest,root/('run_'+policy),policy=policy,scheduler_name='Las'));print(json.dumps(results[-1]),flush=True)
execute(manifest,root/'run_risk_prefer_packed_repeat',policy='risk_prefer_packed',scheduler_name='Las')
checks={f:sha256(root/'run_risk_prefer_packed'/f)==sha256(root/'run_risk_prefer_packed_repeat'/f) for f in ['events.jsonl','decisions.jsonl','summary.json']};atomic_json(root/'reproducibility.json',checks);assert all(checks.values())
pd.DataFrame(results).to_csv(root/'comparison.csv',index=False,encoding='utf-8-sig')
print('REPRODUCIBLE '+json.dumps(checks),flush=True)
