// No dependencies. Run from any directory with node display_analysis.mjs.
import fs from 'node:fs'
import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { distanceToRetina } from '../../../../services/viz/src/geometry.js'
import { fileURLToPath } from 'node:url'
import { createTrackingPipeline } from '../../../../services/viz/src/feed/pipeline.js'
const out = fileURLToPath(new URL('.', import.meta.url))
const [header, ...lines] = fs.readFileSync(out + 'ui_replay.csv', 'utf8').trim().split(/\r?\n/)
const keys = header.split(',')
const rows = lines.map(line => Object.fromEntries(line.split(',').map((v,i)=>[keys[i], keys[i]==='run'?v:Number(v)])))
const quantile = (a,p) => { const s=[...a].sort((a,b)=>a-b), i=(s.length-1)*p, j=Math.floor(i); return s[j]+(s[Math.ceil(i)]-s[j])*(i-j) }
const csv = (name, data) => fs.writeFileSync(out+name, Object.keys(data[0]).join(',')+'\n'+data.map(r=>Object.values(r).join(',')).join('\n')+'\n')
// Display-conditioned table uses finite measurements only (not held NaN rows),
// exactly the condition of the reviewer's numbers. Grace is included separately below.
const accuracy = ['clear','near','critical'].map((level,i)=>{
 const s=rows.filter(r=>r.level===i && Number.isFinite(r.min_mm)), gt=s.map(r=>r.gt_midpoint_mm)
 return {level,n:s.length,truth_median_mm:quantile(gt,.5),truth_p10_mm:quantile(gt,.1),truth_p90_mm:quantile(gt,.9),true_below_05_fraction:gt.filter(x=>x<.5).length/s.length,true_below_2_fraction:gt.filter(x=>x<2).length/s.length,median_signed_error_mm:quantile(s.map(r=>r.min_mm-r.gt_midpoint_mm),.5)}
})
csv('display_accuracy.csv',accuracy)
function replay(dwell, actual=false) {
 let run=null,p,prev=null,changes=0,flips=0,nearChanges=0,available=0,nearFrames=0
 const labels=[], displayedRows=[]
 for(const r of rows){
  if(r.run!==run){run=r.run;p=createTrackingPipeline(actual ? {} : {nearestDwellMs:dwell});p.reset('replay');prev=null}
  const valid=Number.isFinite(r.min_mm), t=r.time_ms
  const f=valid?{tip_left:[0,r.left_mm-12,0],tip_right:[0,r.right_mm-12,0],t}:null
  p.publish(f,f?'replay':'degraded',t);p.tick(t)
  const displayed=p.readout.get()
  const minimum=displayed.frame ? Math.min(distanceToRetina(displayed.frame.tip_left),distanceToRetina(displayed.frame.tip_right)) : Infinity
  assert.equal(displayed.levels ? {safe:0,warn:1,danger:2}[displayed.levels.min] : -1,r.level)
  if(displayed.frame) displayedRows.push({...r,min_mm:minimum})
  const label=displayed.nearestTip
  if(label!=null){available++;if(minimum<2)nearFrames++}
  // Published label changes include readout throttling and grace, exclude availability edges; no transitions across runs.
  if(prev!=null&&label!=null&&prev!==label){changes++;if(prev!=='tie'&&label!=='tie')flips++;if(minimum<2)nearChanges++}
  prev=label;labels.push(label)
 }
 return {variant:actual?'implemented':`dwell_${dwell}`,changes,changes_per_min:changes*1800/rows.length,direct_flips:flips,direct_flips_per_min:flips*1800/rows.length,changes_below_2_fraction:nearChanges/changes,display_below_2_fraction:nearFrames/available,labels,displayedRows}
}
const variants=[0,300,400,500].map(d=>replay(d))
const final=replay(500,true)
const noise=rows.filter(r=>Number.isFinite(r.min_mm)).map(r=>Math.abs(r.left_mm-r.right_mm))
const delta=[]
for(let i=1;i<rows.length;i++){const a=rows[i-1],b=rows[i];if(a.run===b.run&&Number.isFinite(a.min_mm)&&Number.isFinite(b.min_mm))delta.push((b.left_mm-b.right_mm)-(a.left_mm-a.right_mm))}
const median=quantile(delta,.5), sigma=1.4826*quantile(delta.map(x=>Math.abs(x-median)),.5)/Math.sqrt(2)
const summary={accuracy,variants:variants.map(({labels,displayedRows,...r})=>r),implemented:(({labels,displayedRows,...r})=>r)(final),gap_mm:{p10:quantile(noise,.1),median:quantile(noise,.5),p90:quantile(noise,.9),p95:quantile(noise,.95),robust_noise_sigma:sigma}}
// Sustained asymmetry check: at least 1 s with |L−R| > 0.15 mm and fixed nearer side;
// report acquisition within each episode (excludes missing/grace frames).
summary.sustained=variants.map(v=>{let episodes=0,followed=0,maxLatency=0
 for(let i=0;i<rows.length;){const start=i,target=rows[i].left_mm < rows[i].right_mm ? 'left' : 'right',run=rows[i].run
  if(!Number.isFinite(rows[i].min_mm)||Math.abs(rows[i].left_mm-rows[i].right_mm)<=.15){i++;continue}
  while(i<rows.length&&rows[i].run===run&&Number.isFinite(rows[i].min_mm)&&Math.abs(rows[i].left_mm-rows[i].right_mm)>.15&&(rows[i].left_mm<rows[i].right_mm?'left':'right')===target)i++
  if((i-start)/30>=1){episodes++;const j=v.labels.slice(start,i).indexOf(target);if(j>=0){followed++;maxLatency=Math.max(maxLatency,j*1000/30)}}
 }
 return {variant:v.variant,episodes,followed,max_latency_ms:maxLatency}
})
// Supplemental table includes grace and the last published numeric frame.
const withGrace = ['clear','near','critical'].map((level,i)=>{
 const sample=final.displayedRows.filter(r=>r.level===i), gt=sample.map(r=>r.gt_midpoint_mm)
 return {level,n:sample.length,truth_median_mm:quantile(gt,.5),truth_p10_mm:quantile(gt,.1),truth_p90_mm:quantile(gt,.9),true_below_05_fraction:gt.filter(x=>x<.5).length/sample.length,true_below_2_fraction:gt.filter(x=>x<2).length/sample.length,median_signed_error_mm:quantile(sample.map(r=>r.min_mm-r.gt_midpoint_mm),.5)}
})
csv('display_accuracy_with_grace.csv',withGrace)
summary.accuracy_with_grace=withGrace
csv('nearest_stability.csv',[...summary.variants,summary.implemented])
summary.source_sha256 = Object.fromEntries(['src/feed/pipeline.js','src/config.js','src/geometry.js','src/ui.js'].map(path=>[path,createHash('sha256').update(fs.readFileSync(new URL('../../../../services/viz/'+path,import.meta.url))).digest('hex')]))
fs.writeFileSync(out+'display_analysis.json',JSON.stringify(summary,null,2)+'\n')
console.log(JSON.stringify(summary,null,2))
