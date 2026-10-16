from pathlib import Path
import os,json,csv,math,hashlib
os.environ.setdefault('MPLCONFIGDIR','/tmp/cmgan-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
ROOT=Path(__file__).resolve().parents[3]; OUT=Path(__file__).resolve().parent
protected=[ROOT/'docs/main_zh.md',OUT/'snr_rmse_curves_v1.png',OUT/'t60_rmse_curves_v1.png']
hashes={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
roots={'BERP':'papercompare/berp_t60_v7_fixed/test_results','DAMTL':'papercompare/damtl/test','Single-task KAN':'runs/single_t60_kan_v2_mse_seed42/test','Proposed':'runs/kan_multitask_norm_w_denoise_b005_seed42/test'}
rows={}; metrics={}
for name,root in roots.items():
 ds=[json.loads((ROOT/root/f'evaluation_results_test{i}.json').read_text()) for i in range(1,5)]
 rows[name]=[d['per_sample_results'] for d in ds];metrics[name]=[d['standard_metrics'] for d in ds]
rows['noiseaware']=[];metrics['noiseaware']=[]
for i in range(1,5):
 root=ROOT/f'papercompare/noiseaware/fix-v7dataset-test/fix_normalized_run1_test_test{i}'
 metrics['noiseaware'].append(json.loads((root/'summary.json').read_text())['t60_metrics'])
 with (root/'detailed_results.csv').open() as f:
  rows['noiseaware'].append([dict(sample_name=r['Sample_Name'],snr=float(r['SNR_dB']),true_t60=float(r['T60_True']),error_ms=float(r['Error_ms'])) for r in csv.DictReader(f)])
for method,sets in rows.items():
 for i,rs in enumerate(sets):
  assert len(rs)==1080
  assert {(r['sample_name'],float(r['snr']),r['true_t60']) for r in rs}=={(r['sample_name'],float(r['snr']),r['true_t60']) for r in rows['Proposed'][i]}
styles={'BERP':('#557DA5','s','--'),'DAMTL':('#8E74A6','^','-.'),'noiseaware':('#38958C','D',':'),'Proposed':('#D56838','o','-')}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.labelsize':12,'axes.edgecolor':'#525960','axes.linewidth':.8})
fig,axes=plt.subplots(2,2,figsize=(10.5,8),sharey=True)
fig.subplots_adjust(left=.085,right=.98,bottom=.09,top=.87,wspace=.17,hspace=.43)
audit={}
for i,ax in enumerate(axes.flat):
 audit[f'test{i+1}']={}
 for method,(color,marker,ls) in styles.items():
  rs=rows[method][i];groups=[[r for r in rs if (float(r['snr'])==x if x<15 else float(r['snr'])>=15)] for x in [-5,0,5,10,15]]
  assert sum(map(len,groups))==1080
  vals=[math.sqrt(sum(r['error_ms']**2 for r in g)/len(g)) for g in groups]
  audit[f'test{i+1}'][method]={'rmse':vals,'counts':list(map(len,groups))}
  ax.plot(range(5),vals,label=method,color=color,marker=marker,linestyle=ls,linewidth=2.5 if method=='Proposed' else 1.7,markersize=6.5,markerfacecolor=color if method=='Proposed' else 'white',markeredgewidth=1.3)
 ax.set_title(f"({chr(97+i)}) Test {i+1}: {'Simulated' if i%2==0 else 'Real'} RIRs\n{'Seen' if i<2 else 'Unseen'} noise",fontsize=12.5,pad=12,linespacing=1.5)
 ax.set_xticks(range(5),['−5','0','5','10','≥15']);ax.set_xlim(-.15,4.15);ax.set_ylim(0,280)
 ax.yaxis.set_major_locator(MultipleLocator(50));ax.grid(axis='y',color='#E0E4E8',linewidth=.75);ax.set_axisbelow(True)
 ax.spines[['top','right']].set_visible(False);ax.set_xlabel('SNR group (dB)',labelpad=7)
 if i%2==0:ax.set_ylabel('RMSE (ms)')
fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.53,.985),ncol=4,frameon=False,handlelength=2.6,columnspacing=2.1)
fig.savefig(OUT/'snr_test1_4_preview.png',dpi=300,facecolor='white');plt.close(fig)
order=['BERP','DAMTL','noiseaware','Single-task KAN','Proposed']
lines=['| 方法 | Test 1 | Test 2 | Test 3 | Test 4 | 四集均值 |','|---|---:|---:|---:|---:|---:|']
for m in order:
 vals=[d['rmse_ms'] for d in metrics[m]];vals.append(sum(vals)/4)
 lines.append('| '+m+' | '+' | '.join(f'{v:.2f}' for v in vals)+' |')
print('\n'.join(lines))
full=['# Test 1–4 主结果表预览','', 'RMSE/MAE 单位 ms；Avg. 为四个测试集指标的算术平均。','', '| 测试集 | 方法 | RMSE ↓ | MAE ↓ | PCC ↑ | R² ↑ |','|---|---|---:|---:|---:|---:|']
keys=['rmse_ms','mae_ms','pearson_r','r2_score']
for i in range(5):
 for m in order:
  vals=[metrics[m][i][k] if i<4 else sum(d[k] for d in metrics[m])/4 for k in keys]
  full.append('| '+(f'Test {i+1}' if i<4 else 'Avg.')+' | '+m+' | '+' | '.join(f'{v:.{2 if j<2 else 4}f}' for j,v in enumerate(vals))+' |')
(OUT/'testwise_results_preview.md').write_text('\n'.join(full)+'\n')
(OUT/'testwise_preview_data.json').write_text(json.dumps({'snr':audit,'metrics':metrics},ensure_ascii=False,indent=2))
assert all(hashlib.sha256(p.read_bytes()).hexdigest()==h for p,h in hashes.items())
print('Verified: 1080 matched samples per test; manuscript and prior figures unchanged.')
