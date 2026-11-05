from playwright.sync_api import sync_playwright
from pathlib import Path
p=Path('/mnt/data/imba_genome_parameter_lab_v10_ui.html')
with sync_playwright() as pw:
  browser=pw.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox','--disable-dev-shm-usage','--enable-unsafe-swiftshader'])
  page=browser.new_page(viewport={'width':1440,'height':1050})
  errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
  page.set_content(p.read_text(),wait_until='domcontentloaded',timeout=120000)
  page.wait_for_function('window.__QUANT_V9_READY===true',timeout=120000)
  page.wait_for_timeout(600)
  print('initial:',page.evaluate("()=>[state.dataset,state.day,state.smooth,state.sidebarOpen,document.querySelectorAll('aside.right .panel.folded').length]"),flush=True)
  page.locator('#sidebarToggle').click();page.wait_for_timeout(300)
  assert page.locator('.cols.sidebarHidden').count()==1
  page.locator('#sidebarToggle').click();page.wait_for_timeout(300)
  assert page.locator('.cols.sidebarHidden').count()==0
  page.locator('aside.right section.panel h2').first.click()
  page.locator('#smoothLevel').select_option('2')
  assert page.evaluate('state.smooth')==2
  page.locator('#slider').fill('15')
  page.wait_for_timeout(700)
  print('day15:',page.evaluate("()=>[state.day,document.getElementById('dayLabel').textContent]"),flush=True)
  page.select_option('#dataset','2023')
  page.wait_for_function('state.dataset==="2023"',timeout=120000)
  page.wait_for_timeout(750)
  print('year2023:',page.evaluate("()=>[state.day, DATES.length,choosePortfolio()[state.day].wealth]"),flush=True)
  page.locator('#playRev').click();page.wait_for_timeout(950)
  page.locator('#pause').click();page.wait_for_timeout(200)
  print('played back day:',page.evaluate('state.day'),flush=True)
  assert page.evaluate('state.day') < 364
  page.screenshot(path='/mnt/data/imba_genome_v9/v10_preview.png',full_page=True)
  print('PAGE ERRORS',errors[:10],flush=True)
  assert not errors
  browser.close()