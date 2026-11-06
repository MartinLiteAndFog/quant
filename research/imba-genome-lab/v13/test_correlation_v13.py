from pathlib import Path
from playwright.sync_api import sync_playwright
import json
p=Path('/mnt/data/imba_genome_parameter_lab_v13_presets_charts.html')
with sync_playwright() as pw:
 b=pw.chromium.launch(headless=True,executable_path='/usr/bin/chromium',args=['--no-sandbox','--disable-gpu','--disable-webgl'])
 page=b.new_page();errs=[];page.on('pageerror',lambda e:errs.append(str(e)))
 page.set_content(p.read_text(encoding='utf-8'),wait_until='domcontentloaded',timeout=120000)
 page.wait_for_function('window.__QUANT_V13_READY === true',timeout=120000)
 # Test the 2022 Raw Top100 with 30d ranking, 30% guard, strict threshold
 for cap in ['off','0.85','0.9']:
  data=page.evaluate('''(cap) => {
    state.mode='top100';state.rank=30;state.guard=30;
    state.priceMode='renko_usd';state.box='all';
    state.familyEnabled={none:true,tp:true,ttp:true,both:true};
    state.corrLimit=cap==='off'?null:Number(cap);state.corrFallback=false;
    state.filtered=eligible();state.portKey='';state.port=null;
    const p=choosePortfolio();
    let violations=0, pairs=0;
    if(state.corrLimit!==null){for(let d=0;d<p.length;d+=14){
      const ids=p[d].ids.filter(i=>i>=0);
      for(let i=0;i<ids.length;i++)for(let j=i+1;j<ids.length;j++){
        pairs++;
        if(pearsonPrevious28(ids[i],ids[j],d)>state.corrLimit+1e-9)violations++;
      }
    }}
    return {wealth:p.at(-1).wealth,corr:state.corrLimit,violations,pairs,
      minSelected:Math.min(...p.filter((x,i)=>i%14===0).map(x=>x.ids.filter(v=>v>=0).length)),
      maxSelected:Math.max(...p.filter((x,i)=>i%14===0).map(x=>x.ids.filter(v=>v>=0).length))};
  }''',cap)
  print(cap,data,flush=True)
  assert not data['violations'],data
  if cap=='off':assert abs(data['wealth']-3493.61356857)<0.01
 assert not errs,errs
 b.close()