"""
Build a 100% standalone interactive 100-GPU Simulation Player HTML (zero server required, works anywhere)
using the actual 100-GPU snapshots and events from runs_actual/benchmark_20261017.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path


def main():
    blox_dir = Path(__file__).resolve().parent
    bench_dir = blox_dir / "runs_actual" / "benchmark_20261017"
    report_dir = blox_dir.parent.parent.parent.parent / "보고서" / "Blox연동"
    artifact_dir = Path(r"C:\Users\jehan\.gemini\antigravity\brain\492e8e47-19a0-4e49-8667-1677687f0d46")

    target_runs = [
        "01_baseline_blind",
        "03_junho_prefer_e19",
        "04_solA_falcon_prefer",
        "05_solA_vuln_weighted_dp",
        "06_solA_metaheuristic",
        "08_solA_full_co_opt",
        "11_solAB_metaheuristic_diff10",
        "12_solAB_full_co_opt_diff10",
        "16_mixed31_43_metaheuristic",
        "17_mixed31_43_full_co_opt",
    ]

    # Select dense steps around all 4 crash waves + regular intervals across all 611 steps
    key_steps = sorted(set(
        list(range(0, 611, 10))
        + list(range(140, 147))   # Wave 1 (Server 137)
        + list(range(433, 448))   # Wave 2 (Server 134) & Wave 3 (Server 131)
        + list(range(538, 551))   # Wave 4 (Servers 131 & 140)
        + [610]
    ))

    bundle = {"steps": key_steps, "runs": {}}

    for rid in target_runs:
        rpath = bench_dir / rid
        snaps = [json.loads(line) for line in (rpath / "snapshots.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        evs = [json.loads(line) for line in (rpath / "events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        summary = json.loads((rpath / "summary.json").read_text(encoding="utf-8"))

        run_frames = []
        for pos, s_idx in enumerate(key_steps):
            snap = snaps[s_idx]
            t_curr = float(snap["time"])
            t_prev = float(snaps[s_idx - 1]["time"]) if s_idx > 0 else -1.0

            # Compact GPU array: [server_short, local_id, state_code, job_id_str, pf, b1, b2, eligible, alert]
            # state_code: 0=available, 1=running, 2=maintenance/failed, 3=cooldown
            compact_gpus = []
            for g in snap["gpus"]:
                st_map = {"available": 0, "running": 1, "maintenance": 2, "failed": 2, "drained": 2, "cooldown": 3}
                sc = st_map.get(g["state"], 0)
                srv_short = g["server_id"].replace("10.140.0.", "")
                jid = str(g["job_id"]) if g.get("job_id") is not None else ""
                pf = round(float(g.get("probability", 0.0)), 4)
                b1 = round(float(g.get("b1_probability") or 0.0), 4)
                b2 = round(float(g.get("b2_probability") or 0.0), 4)
                el = 1 if g.get("eligible", True) else 0
                al = 1 if g.get("risk_alert", False) else 0
                compact_gpus.append([srv_short, int(g["local_id"]), sc, jid, pf, b1, b2, el, al])

            # Events in this step window
            step_evs = []
            for e in evs:
                et = float(e.get("time", 0.0))
                if t_prev < et <= t_curr and e.get("type") in ("gang_failure", "job_launch", "job_relaunch", "scheduler_preemption", "job_complete"):
                    glist = [u.replace("10.140.0.", "") for u in (e.get("gpu_ids") or ([e["gpu_id"]] if e.get("gpu_id") else []))]
                    step_evs.append([e.get("timestamp", "")[11:19], e["type"], str(e.get("job_id") or ""), ",".join(glist[:8])])
            # Always keep gang_failure events first if truncated
            fail_evs = [x for x in step_evs if x[1] == "gang_failure"]
            other_evs = [x for x in step_evs if x[1] != "gang_failure"]
            step_evs = (fail_evs + other_evs[-18:])[:22]

            cum_hits = sum(1 for e in evs if float(e.get("time", 0.0)) <= t_curr and e.get("type") == "gang_failure")
            cum_prm = sum(1 for e in evs if float(e.get("time", 0.0)) <= t_curr and e.get("type") == "scheduler_preemption")

            run_frames.append({
                "step": s_idx,
                "ts": snap["timestamp"],
                "comp": snap["jobs"]["completed"],
                "run": snap["jobs"]["running"],
                "q": snap["jobs"]["queued"],
                "hits": cum_hits,
                "prm": cum_prm,
                "gpus": compact_gpus,
                "evs": step_evs,
            })

        bundle["runs"][rid] = {
            "summary": {
                "completed_jobs": summary["completed_jobs"],
                "job_failure_lost_s": summary["job_failure_lost_seconds"],
                "preempt_s": summary["scheduling_preemption_overhead_seconds"],
                "vuln_exp": round(summary["vuln_weighted_risk_exposure"] / 1e9, 2),
                "avg_jct_s": round(summary["avg_jct_seconds"], 1) if summary["avg_jct_seconds"] else None,
                "p95_wait_h": round(summary["p95_queue_wait_seconds"] / 3600.0, 2),
                "avoided": summary["avoided_xid_episodes_on_idle"],
                "active_hits": summary["active_xid_job_interruptions"],
            },
            "frames": run_frames,
        }

    json_str = json.dumps(bundle, ensure_ascii=False, separators=(",", ":"))

    html_template = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Blox + RiskLab · 실시간 100-GPU 시뮬레이션 리플레이 플레이어</title>
<script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
<style>
  * {{ box-sizing: border-box; }}
  body {{
    font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, 'Malgun Gothic', sans-serif;
    margin: 0;
    padding: 18px;
    background: var(--background, #0f172a);
    color: var(--foreground, #f8fafc);
  }}
  .player-card {{
    background: var(--card, #1e293b);
    border: 1px solid var(--border, #334155);
    border-radius: 14px;
    padding: 16px 20px;
    margin-bottom: 14px;
  }}
  .ctrl-btn {{
    border: 1px solid var(--border, #475569);
    background: var(--secondary, #0f172a);
    color: var(--foreground, #f8fafc);
    padding: 7px 13px;
    border-radius: 8px;
    font-size: 12.5px;
    font-weight: 700;
    cursor: pointer;
  }}
  .ctrl-btn.play-active {{
    background: #059669;
    color: #ffffff;
    border-color: #10b981;
  }}
  .ctrl-btn.wave-btn {{
    background: rgba(245, 158, 11, 0.15);
    border-color: rgba(245, 158, 11, 0.5);
    color: #fbbf24;
  }}
  .ctrl-btn.wave-btn.active-wave {{
    background: #d97706;
    color: #ffffff;
  }}
  select.pol-select {{
    background: var(--secondary, #0f172a);
    color: var(--foreground, #f8fafc);
    border: 1px solid var(--border, #475569);
    padding: 7px 10px;
    border-radius: 8px;
    font-size: 12.5px;
    font-weight: 700;
  }}
  .compare-wrap {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
  }}
  @media (max-width: 1100px) {{
    .compare-wrap {{ grid-template-columns: 1fr; }}
  }}
  .srv-grid-wrap {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(225px, 1fr));
    gap: 8px;
  }}
  .srv-box {{
    border: 1px solid var(--border, #334155);
    background: var(--secondary, #0f172a);
    border-radius: 9px;
    padding: 7px 8px;
  }}
  .gpu-row {{
    display: grid;
    grid-template-columns: repeat(8, minmax(0, 1fr));
    gap: 3px;
    margin-top: 5px;
  }}
  .gpu-cell {{
    height: 38px;
    border-radius: 4px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    font-size: 10px;
    font-weight: 700;
    cursor: pointer;
    border: 1.5px solid transparent;
    overflow: hidden;
    transition: transform 0.08s ease;
  }}
  .gpu-cell:hover {{ transform: scale(1.1); z-index: 5; }}
  .gpu-st-0 {{ background: rgba(148, 163, 184, 0.22); color: var(--muted-foreground, #94a3b8); }}
  .gpu-st-1 {{ background: #0d9488; color: #ffffff; }}
  .gpu-st-2 {{ background: #dc2626; color: #ffffff; animation: pulseCrash 0.8s infinite alternate; }}
  .gpu-st-3 {{ background: #7c3aed; color: #ffffff; }}
  .gpu-alert {{ border: 1.5px dashed #f59e0b; }}
  .gpu-excl {{ border: 2px solid #ef4444; }}
  @keyframes pulseCrash {{
    0% {{ transform: scale(1.0); box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }}
    100% {{ transform: scale(1.04); box-shadow: 0 0 0 4px rgba(239, 68, 68, 0); }}
  }}
  .ev-log {{
    height: 135px;
    overflow-y: auto;
    font-family: ui-monospace, Consolas, monospace;
    font-size: 12px;
    background: #090d16;
    color: #cbd5e1;
    border-radius: 8px;
    padding: 8px 10px;
    margin-top: 10px;
    border: 1px solid #334155;
  }}
  .ev-fail {{ color: #f87171; font-weight: 800; background: rgba(220, 38, 38, 0.2); padding: 2px 5px; border-radius: 4px; }}
  .ev-launch {{ color: #34d399; }}
  .ev-prm {{ color: #fbbf24; }}
</style>
</head>
<body>
  <div style="max-width:1540px;margin:0 auto;">
    <!-- Top Player Header & Controls -->
    <div class="player-card">
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;">
        <div>
          <h1 style="font-size:20px;font-weight:800;margin:0;">🎬 Blox + RiskLab 100-GPU 시뮬레이션 실시간 재생 &amp; 1:1 비교 플레이어</h1>
          <p style="font-size:12.5px;color:var(--muted-foreground,#94a3b8);margin:3px 0 0 0;">
            직접 <strong>▶ 재생(Play)</strong> 또는 <strong>⏭ +1 스텝</strong>을 눌러 15개 서버(100 GPU)에서 작업들이 배치되고, XID 폭발 시 왼쪽(기존)은 피격되지만 오른쪽(제안 메타휴리스틱)은 유휴 상태로 회피하는 모습을 확인하세요!
          </p>
        </div>
        <div style="display:flex;gap:6px;flex-wrap:wrap;align-items:center;">
          <button id="btn-play" class="ctrl-btn play-active">▶ 자동 재생 (Play)</button>
          <button id="btn-prev" class="ctrl-btn">⏮ -1 스텝</button>
          <button id="btn-next" class="ctrl-btn">⏭ +1 스텝</button>
          <select id="sel-speed" class="pol-select">
            <option value="1100">배속: 1x (천천히 관찰)</option>
            <option value="550" selected>배속: 2x (권장)</option>
            <option value="200">배속: 5x (빠르게)</option>
          </select>
        </div>
      </div>

      <!-- Crash Wave Instant Jump Bar -->
      <div style="display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-top:12px;padding-top:12px;border-top:1px solid var(--border,#334155);">
        <span style="font-size:12px;font-weight:800;color:#fbbf24;">⚡ 주요 XID 장애 폭발 시점으로 즉시 이동 (클릭 후 ▶ 재생 또는 ⏭ +1 스텝을 눌러보세요):</span>
        <button class="ctrl-btn wave-btn" data-step="142">Wave 1: Step 142→143 (서버 137 4-GPU 폭발)</button>
        <button class="ctrl-btn wave-btn" data-step="435">Wave 2: Step 435→436 (서버 134 8-GPU 통째 폭발!)</button>
        <button class="ctrl-btn wave-btn" data-step="444">Wave 3: Step 444→445 (서버 131 3-GPU 폭발)</button>
        <button class="ctrl-btn wave-btn active-wave" data-step="540">Wave 4: Step 540→541 (서버 131·140 16-GPU 동시 대폭발!)</button>
      </div>

      <!-- Scrubber Slider -->
      <div style="display:flex;align-items:center;gap:12px;margin-top:12px;">
        <div id="step-badge" style="font-family:monospace;font-size:13px;font-weight:800;padding:5px 12px;border-radius:6px;background:var(--secondary,#0f172a);border:1px solid var(--border,#475569);min-width:270px;text-align:center;">
          Step 540 / 610
        </div>
        <input type="range" id="scrubber" min="0" max="{len(key_steps)-1}" value="0" style="flex:1;cursor:pointer;height:8px;accent-color:#2563eb;">
        <span id="narrative-hint" style="font-size:12.5px;font-weight:700;color:#38bdf8;"></span>
      </div>

      <!-- GPU Click Inspector -->
      <div id="gpu-inspector" style="margin-top:10px;padding:8px 12px;border-radius:8px;background:rgba(37,99,235,0.12);border:1px solid rgba(59,130,246,0.35);font-size:12.5px;display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px;">
        <span>👆 <strong>100개 GPU 중 원하는 칸을 클릭해 보세요:</strong> 해당 GPU의 실시간 <strong>Branch 1(단기 텔레메트리 위험도)</strong>, <strong>Branch 2(만성 이력 위험도)</strong>, <strong>결합 위험도</strong> 및 실행 중인 <strong>Job ID</strong>가 즉시 표시됩니다.</span>
      </div>
    </div>

    <!-- Side-by-Side Cluster Simulation View -->
    <div class="compare-wrap">
      <!-- Left Side -->
      <div class="player-card">
        <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:10px;">
          <span style="font-size:15px;font-weight:800;">🅰️ 좌측 시뮬레이터 (기존 방식 선택)</span>
          <select id="sel-left" class="pol-select"></select>
        </div>
        <div id="kpi-left" style="display:grid;grid-template-columns:repeat(5,1fr);gap:6px;margin-bottom:10px;"></div>
        <div id="grid-left" class="srv-grid-wrap"></div>
        <div id="log-left" class="ev-log"></div>
      </div>

      <!-- Right Side -->
      <div class="player-card">
        <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:10px;">
          <span style="font-size:15px;font-weight:800;">🅱️ 우측 시뮬레이터 (제안 최적화 모델 선택)</span>
          <select id="sel-right" class="pol-select"></select>
        </div>
        <div id="kpi-right" style="display:grid;grid-template-columns:repeat(5,1fr);gap:6px;margin-bottom:10px;"></div>
        <div id="grid-right" class="srv-grid-wrap"></div>
        <div id="log-right" class="ev-log"></div>
      </div>
    </div>

    <!-- Color Legend -->
    <div class="player-card" style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;font-size:12px;">
      <div style="display:flex;gap:14px;flex-wrap:wrap;align-items:center;">
        <span><span style="display:inline-block;width:11px;height:11px;border-radius:3px;background:rgba(148,163,184,0.35);vertical-align:middle;margin-right:4px;"></span>유휴/사용가능 (Idle)</span>
        <span><span style="display:inline-block;width:11px;height:11px;border-radius:3px;background:#0d9488;vertical-align:middle;margin-right:4px;"></span>작업 실행 중 (#JobID 뒷 4자리)</span>
        <span><span style="display:inline-block;width:11px;height:11px;border-radius:3px;background:#dc2626;vertical-align:middle;margin-right:4px;"></span>💥 XID 장애 폭발 / 하드웨어 격리</span>
        <span><span style="display:inline-block;width:11px;height:11px;border-radius:3px;background:#7c3aed;vertical-align:middle;margin-right:4px;"></span>🔄 XID 43 소프트 리셋 (60s)</span>
        <span><strong>주황 점선 테두리:</strong> ML 고위험 경고 GPU &nbsp;|&nbsp; <strong>붉은 실선 테두리:</strong> Top-K 격리 제외 GPU</span>
      </div>
    </div>
  </div>

<script>
const BUNDLE = {json_str};

const POLICY_NAMES = {{
  "01_baseline_blind": "01. Blind (위험도 미사용 · 실패손실 1,305s)",
  "03_junho_prefer_e19": "03. 준호 E19 + 단일작업 Greedy DP (실패손실 1,020s)",
  "04_solA_falcon_prefer": "04. 해법A(Falcon+Pareto) + 단일작업 DP (실패손실 720s)",
  "05_solA_vuln_weighted_dp": "05. 해법A + 취약도 다중작업 Exact DP (실패손실 0s!)",
  "06_solA_metaheuristic": "06. 해법A + 메타휴리스틱 2-Opt+SA (실패손실 0s · 위험노출 최저!)",
  "08_solA_full_co_opt": "08. 최종 통합: 메타휴리스틱 + 선점가드 (선점 -25.6% · JCT -930s!)",
  "11_solAB_metaheuristic_diff10": "11. [해법A+B 소프트리셋] 메타휴리스틱 (실패손실 0s!)",
  "12_solAB_full_co_opt_diff10": "12. [해법A+B 소프트리셋] 최종 통합 (JCT -1,009s!)",
  "16_mixed31_43_metaheuristic": "16. [혼합 XID 31+43] 메타휴리스틱 (실패손실 0s!)",
  "17_mixed31_43_full_co_opt": "17. [혼합 XID 31+43] 최종 통합 (실패손실 -78.7% · JCT -1,033s!)"
}};

const STEP_NOTES = {{
  142: "⚡ [Wave 1 폭발 20분 전] 서버 137의 위험도가 상승합니다. 다음 스텝(+1)을 눌러 137번 서버 폭발 결과를 확인하세요!",
  143: "💥 [Wave 1 폭발 발생!] 서버 137(GPU 0~3)에 XID 장애 발생! 좌측(03번)은 Job 7747102가 피격되지만 우측(06번)은 비어 있어 손실 0초!",
  435: "⚡ [Wave 2 폭발 20분 전] 8-GPU 대형 작업(#3586)과 4-GPU 작업(#3707)이 도착했습니다! 좌측과 우측이 어느 서버에 배치했는지 비교해 보세요.",
  436: "💥 [Wave 2 대규모 폭발!] 서버 134의 8개 GPU 전체가 55분간 다운됩니다! 좌측은 작업이 폭발했지만 우측(메타휴리스틱)은 서버 139·132로 대피시켜 손실 0초!",
  444: "⚡ [Wave 3 폭발 직전] 만성 위험 서버 131에서 곧 3-GPU 장애가 발생합니다.",
  445: "💥 [Wave 3 폭발 발생!] 서버 131(GPU 0~2) 장애 발생! Blind(01번)는 2개 작업이 동시 피격됩니다.",
  540: "⚡ [Wave 4 연쇄폭발 21분 전] 8-GPU 대형 작업(#6400) 도착! 좌측(단일작업 DP)은 서버 141을 1-GPU(#6468)에게 뺏겨 #6400을 위험 서버(131/140)에 넣지만, 우측(메타휴리스틱)은 #6400을 안전 통서버 141에 배정합니다!",
  541: "💥 [Wave 4 16-GPU 동시 대폭발!] 서버 131(8개)과 서버 140(8개) 총 16개 GPU가 동시에 폭발합니다! 좌측은 #6400이 직격탄을 맞지만, 우측(메타휴리스틱)은 두 서버 모두 비워두어 피격 0건!"
}};

const $ = id => document.getElementById(id);
let posIdx = BUNDLE.steps.indexOf(540);
if (posIdx < 0) posIdx = 0;
let isPlaying = false;
let timer = null;

function renderSide(side, runId, frameIdx) {{
  const runObj = BUNDLE.runs[runId];
  const frame = runObj.frames[frameIdx];
  const sum = runObj.summary;

  $(`kpi-${{side}}`).innerHTML = `
    <div style="padding:6px;border-radius:6px;border:1px solid var(--border,#334155);text-align:center;">
      <div style="font-size:11px;color:var(--muted-foreground,#94a3b8);">현재 완료 작업</div>
      <div style="font-size:14px;font-weight:800;">${{frame.comp.toLocaleString()}}</div>
    </div>
    <div style="padding:6px;border-radius:6px;border:1px solid var(--border,#334155);text-align:center;">
      <div style="font-size:11px;color:var(--muted-foreground,#94a3b8);">실행 / 대기</div>
      <div style="font-size:14px;font-weight:800;">${{frame.run}} / ${{frame.q}}</div>
    </div>
    <div style="padding:6px;border-radius:6px;border:1px solid var(--border,#334155);text-align:center;">
      <div style="font-size:11px;color:var(--muted-foreground,#94a3b8);">누적 작업 피격</div>
      <div style="font-size:14px;font-weight:800;color:${{frame.hits > 0 ? '#f87171' : '#34d399'}};">${{frame.hits}}회 피격</div>
    </div>
    <div style="padding:6px;border-radius:6px;border:1px solid var(--border,#334155);text-align:center;">
      <div style="font-size:11px;color:var(--muted-foreground,#94a3b8);">총 실패손실</div>
      <div style="font-size:14px;font-weight:800;color:${{sum.job_failure_lost_s === 0 ? '#34d399' : '#fbbf24'}};">${{sum.job_failure_lost_s}}s</div>
    </div>
    <div style="padding:6px;border-radius:6px;border:1px solid var(--border,#334155);text-align:center;">
      <div style="font-size:11px;color:var(--muted-foreground,#94a3b8);">평균 JCT</div>
      <div style="font-size:14px;font-weight:800;">${{sum.avg_jct_s ? sum.avg_jct_s.toLocaleString() + 's' : 'N/A'}}</div>
    </div>
  `;

  // Group GPUs by server
  const srvMap = new Map();
  for (const g of frame.gpus) {{
    if (!srvMap.has(g[0])) srvMap.set(g[0], []);
    srvMap.get(g[0]).push(g);
  }}

  const gridEl = $(`grid-${{side}}`);
  gridEl.innerHTML = "";
  for (const [srv, gpus] of srvMap) {{
    const box = document.createElement("div");
    box.className = "srv-box";
    const runCnt = gpus.filter(x => x[2] === 1).length;
    const downCnt = gpus.filter(x => x[2] >= 2).length;
    if (downCnt > 0) box.style.borderColor = "#ef4444";

    const head = document.createElement("div");
    head.style.cssText = "display:flex;justify-content:space-between;font-size:11.5px;font-weight:700;";
    head.innerHTML = `<span>서버 ${{srv}}</span><span style="color:${{downCnt > 0 ? '#f87171' : '#94a3b8'}}">${{downCnt > 0 ? '💥 장애/리셋 ' + downCnt + 'GPU' : runCnt + '/' + gpus.length + ' 가동'}}</span>`;

    const row = document.createElement("div");
    row.className = "gpu-row";
    for (const g of gpus.sort((a, b) => a[1] - b[1])) {{
      const [sName, lid, stCode, jid, pf, b1, b2, elig, alert] = g;
      const cell = document.createElement("div");
      cell.className = `gpu-cell gpu-st-${{stCode}}${{alert ? ' gpu-alert' : ''}}${{!elig ? ' gpu-excl' : ''}}`;
      const stLabel = jid ? "#" + jid.slice(-4) : (stCode === 2 ? "DOWN" : (stCode === 3 ? "RST" : "idle"));
      cell.innerHTML = `<span style="font-size:9px;opacity:0.85;">G${{lid}}</span><span style="font-size:9.5px;">${{stLabel}}</span>`;
      cell.onclick = () => {{
        const stNames = ["AVAILABLE (유휴)", "RUNNING (실행 중)", "MAINTENANCE / FAILED (XID 장애 격리)", "COOLDOWN (XID 43 소프트 리셋)"];
        $("gpu-inspector").innerHTML = `
          <span>🔍 <strong>[${{runId}}] 서버 10.140.0.${{sName}} · GPU ${{lid}}</strong> &nbsp;|&nbsp; 상태: <strong>${{stNames[stCode]}}</strong> &nbsp;|&nbsp; 실행 작업: <strong>${{jid ? 'Job ' + jid : '없음(유휴)'}}</strong></span>
          <span>결합 위험도(p<sup>fuse</sup>): <strong>${{pf}}</strong> &nbsp;|&nbsp; Branch 1(단기 텔레메트리): <strong>${{b1}}</strong> &nbsp;|&nbsp; Branch 2(만성 이력): <strong>${{b2}}</strong></span>
        `;
      }};
      row.appendChild(cell);
    }}
    box.append(head, row);
    gridEl.appendChild(box);
  }}

  const logEl = $(`log-${{side}}`);
  if (!frame.evs.length) {{
    logEl.innerHTML = `<div style="color:#64748b;">[Step ${{frame.step}}] 이 구간에는 신규 시작/선점/장애 이벤트가 없습니다.</div>`;
  }} else {{
    logEl.innerHTML = frame.evs.map(e => {{
      const [tm, tp, jid, gstr] = e;
      let cls = "";
      let icon = "•";
      if (tp === "gang_failure") {{ cls = "ev-fail"; icon = "💥 [XID 작업 피격!]"; }}
      else if (tp === "job_launch" || tp === "job_relaunch") {{ cls = "ev-launch"; icon = "🚀 [작업 배치]"; }}
      else if (tp === "scheduler_preemption") {{ cls = "ev-prm"; icon = "⏸️ [스케줄러 선점]"; }}
      return `<div class="${{cls}}" style="margin-bottom:3px;">[${{tm}}] ${{icon}} ${{tp}} · Job ${{jid || '—'}} · GPU [${{gstr}}]</div>`;
    }}).join("");
  }}
}}

function renderCurrent() {{
  $("scrubber").value = posIdx;
  const stepNum = BUNDLE.steps[posIdx];
  const leftId = $("sel-left").value;
  const rightId = $("sel-right").value;
  const ts = BUNDLE.runs[leftId].frames[posIdx].ts.replace("T", " ").slice(0, 16) + " UTC";
  $("step-badge").textContent = `Step ${{stepNum}} / 610 · ${{ts}}`;
  $("narrative-hint").textContent = STEP_NOTES[stepNum] || "";

  document.querySelectorAll(".wave-btn").forEach(b => {{
    b.classList.toggle("active-wave", Number(b.dataset.step) === stepNum || Number(b.dataset.step) + 1 === stepNum);
  }});

  renderSide("left", leftId, posIdx);
  renderSide("right", rightId, posIdx);
}}

function togglePlay() {{
  isPlaying = !isPlaying;
  $("btn-play").textContent = isPlaying ? "⏸ 일시정지 (Pause)" : "▶ 자동 재생 (Play)";
  if (isPlaying) tick();
  else if (timer) {{ clearTimeout(timer); timer = null; }}
}}

function tick() {{
  if (!isPlaying) return;
  const speed = Number($("sel-speed").value);
  timer = setTimeout(() => {{
    if (!isPlaying) return;
    posIdx = (posIdx + 1) % BUNDLE.steps.length;
    renderCurrent();
    tick();
  }}, speed);
}}

// Populate dropdowns
for (const [k, label] of Object.entries(POLICY_NAMES)) {{
  const o1 = document.createElement("option");
  o1.value = k; o1.textContent = label;
  $("sel-left").appendChild(o1);

  const o2 = document.createElement("option");
  o2.value = k; o2.textContent = label;
  $("sel-right").appendChild(o2);
}}
$("sel-left").value = "04_solA_falcon_prefer";
$("sel-right").value = "06_solA_metaheuristic";

$("btn-play").onclick = togglePlay;
$("btn-prev").onclick = () => {{ if (isPlaying) togglePlay(); posIdx = Math.max(0, posIdx - 1); renderCurrent(); }};
$("btn-next").onclick = () => {{ if (isPlaying) togglePlay(); posIdx = Math.min(BUNDLE.steps.length - 1, posIdx + 1); renderCurrent(); }};
$("scrubber").oninput = e => {{ if (isPlaying) togglePlay(); posIdx = Number(e.target.value); renderCurrent(); }};
$("sel-left").onchange = renderCurrent;
$("sel-right").onchange = renderCurrent;

document.querySelectorAll(".wave-btn").forEach(btn => {{
  btn.onclick = () => {{
    if (isPlaying) togglePlay();
    const targetStep = Number(btn.dataset.step);
    const found = BUNDLE.steps.indexOf(targetStep);
    if (found >= 0) {{ posIdx = found; renderCurrent(); }}
  }};
}});

renderCurrent();
</script>
</body>
</html>
"""

    out_report_path = report_dir / "BLOX_실시간_시뮬레이션_플레이어_2026-10-17.html"
    out_artifact_path = artifact_dir / "blox_interactive_simulation_player.html"
    out_report_path.write_text(html_template, encoding="utf-8")
    out_artifact_path.write_text(html_template, encoding="utf-8")
    print(f"[Standalone Player Built] Size={len(html_template)/1024:.1f} KB -> {out_report_path} & {out_artifact_path}")


if __name__ == "__main__":
    main()
