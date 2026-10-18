"""Inspect actual run evidence and append the cumulative research note."""
from pathlib import Path
import argparse
import hashlib
import html
import json
import re
import sys

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT/'ML'))
from run_falcon_acmetrace_st_top100 import score_metrics, digest, write_json


def mean_zero_filled(values):
    """Match original E19's zero-positive-origin AP/recall convention."""
    return float(np.mean([0.0 if value is None else value for value in values]))


def inspect(output):
    summary = json.loads((output/'summary.json').read_text(encoding='utf-8'))
    contract = json.loads((output/'execution_contract.json').read_text(encoding='utf-8'))
    metrics = json.loads((output/'metrics.json').read_text(encoding='utf-8'))
    trace = [json.loads(line) for line in (output/'trace.jsonl').read_text(encoding='utf-8').splitlines()]
    feature = json.loads((output/'feature_manifest.json').read_text(encoding='utf-8'))
    count, decisions, positives = 0, 0, 0
    for row in metrics:
        part = output/'full_population'/f'score_part_{row["origin_index"]:04d}.parquet'
        if digest(part) != row['score_sha256']:
            raise ValueError('Score artifact modified')
        frame = pq.read_table(part).to_pandas()
        if frame.duplicated(['timestamp','gpu_uid']).any():
            raise ValueError('Duplicate prediction keys')
        if not (frame.groupby('timestamp').size() == 1992).all():
            raise ValueError('Incomplete scoring population')
        y = frame.label.to_numpy().astype('uint8')
        if hashlib.sha256(y.tobytes()).hexdigest() != row['label_sha256']:
            raise ValueError('Labels changed')
        fresh = score_metrics(frame.falcon_risk_score.to_numpy().reshape(-1,1992), y.reshape(-1,1992))
        for key in ['pr_auc','hits_at_100','positives','recall_at_100','decision_times']:
            if fresh[key] != row[key]:
                raise ValueError(f'Metric mismatch: {key}')
        order = np.argsort(-frame.falcon_risk_score.to_numpy().reshape(-1,1992),axis=1,kind='stable')[:,:100]
        positions = (np.arange(len(order))[:,None]*1992+order).ravel()
        top = frame.iloc[positions].copy()
        top['rank'] = np.tile(np.arange(1,101),len(order))
        topdir = output/'top100'; topdir.mkdir(exist_ok=True)
        top.to_parquet(topdir/part.name,index=False,compression='zstd')
        count += len(frame); decisions += len(order); positives += int(y.sum())
    selected = [r for r in trace if r['stage']=='fit_done' and r.get('tag')=='selection']
    final = [r for r in trace if r['stage']=='fit_done' and str(r.get('tag','')).startswith('origin_')]
    if len(selected) != 5 or len(final) != len(metrics):
        raise ValueError('Missing actual model fits')
    if count != summary['score_rows'] or decisions != summary['decision_times']:
        raise ValueError('Coverage mismatch')
    if len(metrics)==64 and (count!=6081576 or decisions!=3053 or positives!=52945):
        raise ValueError('Full evaluation population/labels differ from E19')
    result = {'status': 'FULL_PIPELINE_VERIFIED' if len(metrics)==64 else 'STANDALONE_PILOT_VERIFIED',
        'origins':len(metrics),'rows':count,'decision_times':decisions,'positive_rows':positives,
        'feature_count':len(feature['features']),'selection_fits':len(selected),'final_fits':len(final),
        'origin_mean_pr_auc':mean_zero_filled([r['pr_auc'] for r in metrics]),
        'origin_mean_recall_at_100':mean_zero_filled([r['recall_at_100'] for r in metrics]),
        'aggregation_policy':'64-origin macro; undefined zero-positive AP/recall contributes zero, matching original E19 summary',
        'undefined_origin_indices':[r['origin_index'] for r in metrics if r['pr_auc'] is None],
        'positive_origin_mean_pr_auc':float(np.mean([r['pr_auc'] for r in metrics if r['pr_auc'] is not None])),
        'hits_at_100':sum(r['hits_at_100'] for r in metrics),
        'pooled_recall_at_100':sum(r['hits_at_100'] for r in metrics)/positives,
        'paired_reference':{name:{'origin_mean_pr_auc':mean_zero_filled([r['reference'][name]['pr_auc'] for r in metrics]),
              'hits_at_100':sum(r['reference'][name]['hits_at_100'] for r in metrics),
              'pooled_recall_at_100':sum(r['reference'][name]['hits_at_100'] for r in metrics)/positives}
              for name in metrics[0]['reference']},
        'selected_learner':summary['selected_learner'],'backend':summary['backend'],
        'scope':'Fresh Falcon adaptation ML; EventPolicy tuning/inference not executed',
        'source_sha256':contract['source_sha256'],'score_parts_verified':len(metrics),
        'top100_rows':decisions*100,'elapsed_seconds':summary['elapsed_seconds'],
        'origin_seconds':[r['seconds'] for r in metrics]}
    if summary.get('report_repaired'):
        result['report_repaired'] = True
        result['training_rerun'] = False
        result['repair_evidence'] = str(output/'independent_verification.json')
    write_json(output/'audit.json',result)
    matrix_path = ROOT/'requirements_fidelity.json'
    if len(metrics)==64 and matrix_path.exists():
        matrix = json.loads(matrix_path.read_text(encoding='utf-8'))
        matrix['scope'] = 'full64-origin adaptation verified; original Falcon fidelity remains partial'
        for requirement in matrix['requirements']:
            requirement['runtime_evidence'] = str(output/'audit.json')+'; '+str(output/'trace.jsonl')
            requirement['output_artifact'] = str(output/'audit.json')
            if requirement['requirement_id']=='R9':
                requirement['status']='confirmed'
                requirement['deviation_and_consequence']='All64 origins fresh ST fits verified; original offline split differs'
        write_json(matrix_path,matrix)
    return result, trace


def record(output, audit, trace):
    md = WORKSPACE/'실험 기록/실험_연대기.md'
    marker = f'실행 증거: {output.relative_to(WORKSPACE).as_posix()}/audit.json'
    content = md.read_text(encoding='utf-8')
    if marker in content:
        return
    number = max(map(int,re.findall(r'^## Experiment (\d+)',content,re.M)))+1
    completed = audit['origins']==64
    title = 'Falcon 특징 결합 ST·Top100 전체 실행' if completed else 'Falcon 특징 결합 ST·Top100 첫 origin 실행'
    fits = [r for r in trace if r['stage']=='fit_done']
    candidates = json.loads((output/'selected_learner.json').read_text(encoding='utf-8'))['candidates']
    entry = f'\n\n## Experiment {number} = {title} = {"실행 검증" if completed else "부분 실행"}\n\n[요약]\n\n'
    entry += '- E19가 주력이고 Falcon은 비교군이라는 사용자 결정을 유지했다. 승인한 설계를 새 격리 폴더에 구현·실행했다. 원본 Falcon ticket 실험의 전체 재현은 아니다.\n'
    entry += f'- 실제 판정: {audit["status"]}. {audit["origins"]}/64 origins, {audit["rows"]:,} GPU-time rows, {audit["decision_times"]:,}시점.\n\n[실행]\n\n'
    entry += '- G-004·G-007·G-008. SHA256 일치한 frozen Parquet 5개에서 새 캐시/특징을 생성했다. Dense132+XID11=143 특징을 결합하고 하나의 선정된 learner에 입력한다. E19 점수는 학습·특징·선택에 들어가지 않는다.\n'
    entry += '- All-XID24h inclusive-bin GT, 직전10분 제외,36h purge,21일 ST,24시간 재학습,30분 전체1,992GPU scoring,seed20260905를 적용했다. 최초 평가 이전 성숙한3일 validation에서 AP로 선택하고 family/preset을 고정했다.\n'
    entry += '- XGBoost/CatBoost/LightGBM은 CUDA/CUDA/OpenCL 학습,RF/IF는 CPU2threads다. 기존 E14/E19 재학습은 재개하지 않았다. GPU 약2.7GB/3GB·99–100%의 GUI 경쟁이 있어 시간은 공유 GPU 환경 실측이다.\n'
    entry += '- 실행기 Falcon_AcmeTrace_ST_Top100/ML/run_falcon_acmetrace_st_top100.py; 감사·Top100 export tools/record_run.py. 실제 model fit 수와 score artifact hash/중복/전수 coverage/label hash/지표 재계산을 확인했다.\n\n[결과]\n\n'
    entry += '| 과거 validation 후보 | AP | Recall@100 |\n| --- | ---: | ---: |\n'
    for r in candidates:
        entry += f'| {r["family"]} ({r["backend"]}) | {r["pr_auc"]:.6f} | {r["recall_at_100"]:.6f} |\n'
    entry += f'\n- 선정 {audit["selected_learner"]} ({audit["backend"]}), origin 평균AP {audit["origin_mean_pr_auc"]:.6f}, origin 평균Recall@100 {audit["origin_mean_recall_at_100"]:.6f}, hits {audit["hits_at_100"]:,}.\n'
    entry += f'- 같은 구간 원본E19 dual: AP {audit["paired_reference"]["fusion_dual_inertia"]["origin_mean_pr_auc"]:.6f}, hits {audit["paired_reference"]["fusion_dual_inertia"]["hits_at_100"]:,}. 전체64 평균과 첫 origin만의 성능을 혼용하지 않는다.\n'
    entry += f'- 새 실행 elapsed {audit["elapsed_seconds"]:.1f}초; origin별시간은audit.json. 후보 선택{audit["selection_fits"]}fits + 최종{audit["final_fits"]}fits. Top100 export {audit["top100_rows"]:,}행.\n\n[분석]\n\n'
    analysis_path = output/'result_analysis.json'
    if analysis_path.exists():
        analysis = json.loads(analysis_path.read_text(encoding='utf-8'))
        table = '\n| 동일64-origin 방법 | origin 평균 AP | pooled Recall@100 | Top100 GPU-time 적중 |\n| --- | ---: | ---: | ---: |\n'
        labels = {'falcon':'Falcon 이식 XGB/ST21', 'b1_probability':'원본B1 ADST/Cascade',
                  'b2_probability':'원본B2 ST21', 'experiment14_fusion_probability':'E14 Fusion',
                  'fusion_no_inertia':'E19 관성 없음', 'fusion_b1_inertia':'E19 B1 관성',
                  'fusion_b2_inertia':'E19 B2 관성', 'fusion_dual_inertia':'E19 양쪽 관성'}
        for name, result in analysis['models'].items():
            table += f'| {labels[name]} | {result["origin_mean_ap"]:.6f} | {result["pooled_recall_at_100"]:.2%} | {result["hits"]:,} |\n'
        entry = entry.replace('\n\n[분석]\n\n', '\n'+table+'\n[분석]\n\n')
        pair = analysis['paired_comparison']
        timing = analysis['timing']
        entry += f'- 동일개발구간 관찰: Falcon AP 상대차이 {pair["ap_relative_change"]:.2%}, pooledRecall차이 {pair["recall_delta"]*100:.2f}%p. AP 일별승/패/동률 {pair["ap_origin_wins"]}/{pair["ap_origin_losses"]}/{pair["ap_origin_ties"]}.\n'
        for block, result in pair['circular_block_bootstrap'].items():
            lower, upper = result['ap_delta_percentile95']
            rlower, rupper = result['pooled_recall_delta_percentile95']
            entry += f'- {block}일 circular block bootstrap 10,000회: Falcon−E19 AP차이95%구간 [{lower:.6f}, {upper:.6f}], pooledRecall차이 [{rlower*100:.2f}, {rupper*100:.2f}]%p. 개발구간의 기술적 불확실성으로 새기간 일반화 검정이 아니다.\n'
        for name, subset in analysis['subsets'].items():
            f = subset['models']['falcon']; e = subset['models']['fusion_dual_inertia']
            entry += f'- {name} ({subset["origins"]}origins): Falcon/E19 AP {f["origin_mean_ap"]:.6f}/{e["origin_mean_ap"]:.6f}, pooledRecall {f["pooled_recall_at_100"]:.2%}/{e["pooled_recall_at_100"]:.2%}.\n'
        entry += f'- 시간: pilot+continuation {timing["end_to_end_compute_seconds"]:.1f}초,64-origin합 {timing["sum64_origin_seconds"]:.1f}초,최종fit(결측치처리포함) {timing["final64_fit_including_imputation_seconds"]:.1f}초,최종특징 {timing["final64_feature_seconds"]:.1f}초. Origin평균 {timing["origin_mean_seconds"]:.1f}초. 환경설치/대기/보고복구시간 제외,동일CPU대조 미실행이므로 GPU가속배수 미기록.\n'
        entry += '- 후보선택 관찰: validation AP는XGB가최고지만 Recall@100은RF가최고였다. AP로동결한선택을평가구간결과로번복하지않는다. 향후Top100목표와선택지표를새validation에서정렬할필요가있다.\n'
        entry += '- 원인 가설: 고정21일창·143열단일XGB와 E19의 ADST/Cascade·B2분리·calibration/residual/inertia 차이가 성능격차에 관여할수있다. 이결과는특정모듈의인과효과를분리하지못한다. 현재E19주력결정을유지하고Falcon이식은비교군으로보존한다.\n'
        event_cluster = [r for r in analysis['per_origin'] if r['origin_index'] in [52,53,54]]
        cluster_gap = sum(r['e19_hits']-r['falcon_hits'] for r in event_cluster)
        overall_gap = analysis['models']['fusion_dual_inertia']['hits']-analysis['models']['falcon']['hits']
        entry += f'- 차이는모든날에균일하지않다. origin52–54의적중차이 {cluster_gap:,}행은전체순적중차이 {overall_gap:,}행의 {cluster_gap/overall_gap:.2%}다. 사후실패구간탐색이며 이구간을제외하여주지표를변경하지않는다.\n'
    entry += '- AP는64-origin macro이며 양성이 없는origin34는0으로 집계하여 원본E19 summary와 일치시켰다. 원시 checkpoint의undefined(None)는 보존한다. 양성origin63개만의 평균과 혼용하지 않는다.\n'
    if audit.get('report_repaired'):
        entry += '- 모든64학습·점수 저장 후 최종집계의None 처리 오류가 발생했다. 실패상태·stderr·학습소스·점수는 보존하고 후처리만 복구했다. 원본XID에서 GT·reference점수·key를 독립 재검증했으며 재학습하지 않았다.\n'
    entry += '- peer host/task·ECC/remap·고장유형별 ticket learner·Multi-Horizon·지속성 특징은 미구현/데이터부족이다. EventPolicy는 kernel만 있고 튜닝·경고평가는 미실행이다. AP 후보선택·고정preset·ST는 원문과 다른 이식 조건이다.\n'
    entry += '- 성능은 이식된 전체방법의 비교다. 모델/특징도 달라 E19 Fusion만의 인과적 효과나 원본 Falcon 전체 대비 우월성으로 해석하지 않는다. 현재64-origin은 개발자료이며 새 미관측 test가 아니다.\n\n[실행할 것]\n\n'
    entry += ('1. 새기간의 validation에서 AP선택 대 Recall@100선택을 사전고정하여 비교하고 별도미관측test를 확보한다.\n' if completed else '1. 첫 origin의 실행 시간·자원·비교 조건 확인 후 전체64 실행 범위를 판단한다.\n')
    entry += '2. 원본Falcon 근접 재현에는host/task·추가hardware채널·typed ticket·정확설정이 필요하다.\n'
    if completed:
        entry += '3. 시간단축 후보는중복되는21일특징의GPU-time별재사용캐시와창통계계산최적화다. raw hash/과거cutoff를키로고정하고 median imputer는각학습구간에서만fit한다. 실제가속비는같은origin·설정의CPU/GPU대조로측정한다. 아직추가실행하지않았다.\n'
    entry += '\n'+marker+'\n'
    content += entry
    renderer = (WORKSPACE/'E14_E19_plan_20261017/record_stopped_falcon_analysis.py').read_text(encoding='utf-8')
    start = renderer.index('def inline(text):')
    end = renderer.index("assert page.count('<h2>Experiment ')",start)
    namespace = {'html':html,'re':re,'content':content}
    exec(renderer[start:end],namespace)
    page = namespace['page']
    if page.count('<h2>Experiment ') != len(re.findall(r'^## Experiment ',content,re.M)):
        raise ValueError('Logbook rendering differs')
    md.write_text(content,encoding='utf-8')
    (md.parent/'실험_연대기.html').write_text(page,encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    audit,trace=inspect(args.output.resolve())
    record(args.output.resolve(),audit,trace)
    print(json.dumps(audit,ensure_ascii=False,indent=2))
