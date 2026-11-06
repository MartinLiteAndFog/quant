from pathlib import Path
from playwright.sync_api import sync_playwright
import json, time, math
ROOT=Path('/mnt/data');PAGE=ROOT/'imba_genome_parameter_lab_v13_presets_charts.html'
errs=[];checks={}
with sync_playwright() as p:
 browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox','--disable-gpu','--disable-webgl'])
 page=browser.new_page(viewport={'width':1480,'height':1000}, device_scale_factor=1)
 page.on('pageerror',lambda e:errs.append(str(e)))
 page.set_content(PAGE.read_text(encoding='utf-8'),wait_until='domcontentloaded',timeout=120000)
 page.wait_for_function('window.__QUANT_V13_READY === true',timeout=120000)
 # screenshot presets after click and wait for HUD
 for k,val in [('original_2026_14d_20',9367.049251),('balanced_2026_30d_30',7242.873165),('balanced_2026_30d_25',10654.97391),('raw_2026_30d_30_comparison',3493.613568)]:
  page.select_option('#snapshotPreset',k)
  page.click('#applySnapshot')
  page.wait_for_function('!state.rendering && state.port !== null',timeout=120000)
  obj=page.evaluate('''() => ({year:state.dataset,mode:state.mode,guard:state.guard,rank:state.rank,wealth:choosePortfolio().at(-1).wealth,active:$('activeSnapshot').textContent})''')
  assert abs(obj['wealth']-val)<.04,(k,obj,val)
  checks[k]=obj
  print('PRESET',k,obj,flush=True)
 # independent annual pool - may switch if training in current year
 page.evaluate("() => {let e=$('poolMode');e.value='rawYear2025';e.dispatchEvent(new Event('change',{bubbles:true}));}")
 page.wait_for_timeout(300)
 state=page.evaluate('''() => ({year:state.dataset,mode:state.mode,ids:eligible().length,trainingSize:YEAR_RAW_POOLS['2025'].length})''')
 assert state['year']=='2026' and state['ids']==100 and state['trainingSize']==100,state
 print('ANNUAL 2025->2026',state,flush=True)
 page.evaluate("() => {let e=$('dataset');e.value='2025';e.dispatchEvent(new Event('change',{bubbles:true}));}")
 page.wait_for_function('state.dataset===\'2025\' && !state.rendering',timeout=120000)
 state=page.evaluate('''() => ({year:state.dataset,mode:state.mode,ids:eligible().length,warning:$('dataWarning').textContent})''')
 assert state['ids']==0 and 'kein Lookahead' in state['warning'],state
 print('LOOKAHEAD BLOCK',state,flush=True)
 page.evaluate("() => {let e=$('dataset');e.value='2026';e.dispatchEvent(new Event('change',{bubbles:true}));}");page.wait_for_function('state.dataset===\'2026\' && !state.rendering',timeout=120000)
 page.click('#chartsJump')
 page.wait_for_function('document.getElementById(\'researchChart\').data?.length > 0',timeout=120000)
 for name in ['mix','sharpe','annual']:
  page.evaluate("(name) => document.querySelector('[data-research-mode=\"'+name+'\"]').click()",name)
  page.wait_for_timeout(550)
  x=page.evaluate('''() => ({mode:researchMode, traces: $('researchChart').data?.length??0, first: $('researchChart').data?.[0]?.name??null, note:$('researchChartNote').textContent.slice(0,120)})''')
  assert x['mode']==name and x['traces']>=1,(name,x)
  print('CHART',x,flush=True)
 page.click('[data-research-mode="mix"]')
 page.wait_for_timeout(800)
 page.screenshot(path=str(ROOT/'imba_genome_v13_work/v13_browser_preview.png'),full_page=True,timeout=120000)
 assert not errs,errs
 print('TESTS PASSED, JS ERRORS',errs)
 browser.close()
(ROOT/'imba_genome_v13_work/browser_checks.json').write_text(json.dumps(checks,indent=2,ensure_ascii=False))