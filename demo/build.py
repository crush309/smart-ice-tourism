#!/usr/bin/env python3
"""
build.py — 从基础版 index.html + insert文件 组装完整版
策略：把文件切成7段，逐段拼接，避免级联替换
"""
import re

BASE = r'C:\Users\crush\Desktop\比赛\智慧城市技术与创意设计大赛\demo'

def read(fname):
    with open(f'{BASE}/{fname}', 'r', encoding='utf-8') as f:
        return f.read()

html = read('index.html')
css = read('css_insert.txt')
hml = read('html_insert.txt')
j1  = read('js_part1.txt')
j2  = read('js_part2.txt')
j3  = read('js_part3.txt')

# ═══════════════════════════════════════════
# 1. 在 </style> 之前插入新 CSS
# ═══════════════════════════════════════════
html = html.replace('</style>', css + '\n</style>', 1)

# ═══════════════════════════════════════════
# 2. 在 </head> 之前插入 ECharts CDN
# ═══════════════════════════════════════════
html = html.replace('</head>',
    '<script src="https://cdn.bootcdn.net/ajax/libs/echarts/5.5.0/echarts.min.js"></script>\n</head>', 1)

# ═══════════════════════════════════════════
# 3. 在 <canvas id="three-canvas"> 之前插入新 HTML 面板
# ═══════════════════════════════════════════
html = html.replace('<canvas id="three-canvas">', hml + '\n\n<canvas id="three-canvas">', 1)

# ═══════════════════════════════════════════
# 4. 在巡检面板之前插入 AI 预测面板
# ═══════════════════════════════════════════
html = html.replace('<!-- 巡检工作面板 -->',
'''<!-- AI客流预测面板 -->
<div id="predict-panel" style="margin-top:10px;">
  <div class="panel-title">🤖 AI客流预测 (置信度92%)</div>
  <div class="predict-chart" id="predict-chart"></div>
  <div class="predict-list" id="predict-list"></div>
</div>

<!-- 巡检工作面板 -->''', 1)

# ═══════════════════════════════════════════
# 5. 在各景点客流标题后添加列表容器
# ═══════════════════════════════════════════
html = html.replace(
    '<div class="panel-title">📋 各景点客流</div>',
    '<div class="panel-title">📋 各景点客流</div>\n  <div id="visitor-list"></div>', 1)

# ═══════════════════════════════════════════
# 6. 替换右侧网评舆情区块（完整替换）
# ═══════════════════════════════════════════
# 旧版 review section (HTML part)
old_review_start = '<div class="panel-title">💬 网评舆情</div>'
old_review_end = '</div>\n  </div>\n\n<!-- 巡检工作面板 -->'

# 找到网评舆情到巡检面板之间的内容并替换
idx_start = html.find(old_review_start)
idx_end = html.find('<!-- 巡检工作面板 -->')
if idx_start >= 0 and idx_end > idx_start:
    before = html[:idx_start]
    after  = html[idx_end:]
    new_review_html = '''<div class="panel-title">💬 网评舆情 <span style="font-size:9px;color:var(--text-muted);">实时更新</span></div>
  <div class="review-feed" id="review-feed"></div>
  <div class="sentiment-bar-wrap">
    <span class="sentiment-bar-label">😊正面</span>
    <div class="sentiment-bar-track"><div class="sentiment-bar-pos" id="sentiment-pos-bar" style="width:65%;">65%</div></div>
  </div>
  <div class="sentiment-bar-wrap">
    <span class="sentiment-bar-label">😞负面</span>
    <div class="sentiment-bar-track"><div class="sentiment-bar-neg" id="sentiment-neg-bar" style="width:35%;">35%</div></div>
  </div>
  <div style="font-size:10px;font-weight:600;color:var(--accent-ice);margin-top:8px;margin-bottom:4px;">🏷️ 关键词云</div>
  <div class="keyword-cloud" id="keyword-cloud"></div>
  <div style="font-size:10px;font-weight:600;color:var(--accent-ice);margin-top:8px;margin-bottom:4px;">📈 情感趋势</div>
  <div class="sentiment-chart" id="sentiment-chart"></div>
  <div style="font-size:10px;font-weight:600;color:var(--accent-ice);margin-top:8px;margin-bottom:4px;">🔍 服务短板识别</div>
  <div class="service-chart" id="service-chart"></div>
  <div class="gap-list" id="gap-list"></div>
</div>
  </div>

<!-- 巡检工作面板 -->'''
    html = before + new_review_html + after

# ═══════════════════════════════════════════
# 7. 删除旧的 Review Data JS 代码块
# ═══════════════════════════════════════════
# 删除从 "// ── Review Data ──" 到 "// ── Inspection Data ──" 之间的所有内容
html = re.sub(
    r'\n// ── Review Data ──.*?(?=\n// ── Inspection Data ──)',
    '\n// ── Review Data (moved to new system below) ──',
    html, flags=re.DOTALL
)

# ═══════════════════════════════════════════
# 8. 在 "// ── Start ──" 之前插入所有新 JS
# ═══════════════════════════════════════════
all_js = '\n' + j1 + '\n' + j2 + '\n' + j3 + '\n'
html = html.replace('\n// ── Start ──', all_js + '\n// ── Start ──', 1)

# ═══════════════════════════════════════════
# 9. 修改 Start 区域初始化调用
# ═══════════════════════════════════════════
html = html.replace(
    'initCameras();\nrenderInspection();\nupdateReviews();\nrenderUI();\ndrawSparkline();\nanimate();',
    'initCameras();\ninitNewPanels();\nrenderInspectionEnhanced();\nupdateReviews();\nupdateSemanticAnalysis();\nupdateVisitorList();\nrenderUI();\ndrawSparkline();\nanimate();',
    1)

# ═══════════════════════════════════════════
# 10. 修改 animate：在 requestAnimationFrame 之前插入帧逻辑
# ═══════════════════════════════════════════
animate_insert = '''
  // ── 每帧平滑更新 ──
  const frameNow = performance.now();
  if (!window._lf) window._lf = frameNow;
  const fd = (frameNow - window._lf) / 1000;
  window._lf = frameNow;
  lerpData(Math.min(0.1, fd));

  // 每秒定期更新
  if (!window._ls) window._ls = 0;
  if (frameNow - window._ls > 1000) { window._ls = frameNow; periodicUpdate(); }

  // 每5秒刷新数据目标
  if (!window._ld) window._ld = 0;
  if (frameNow - window._ld > 5000) { window._ld = frameNow; tickDataTargets(); }

  // 电子围栏颜色更新
  buildingDefs.forEach(b => {
    const cap = b.width * b.depth * 12;
    updateFenceHeatmap(b.label, b.density / Math.max(1, cap));
  });

  // 围栏脉冲动画
  Object.values(fences).forEach(f => {
    f.lines.children.forEach(c => {
      if (c.userData && c.userData.baseY !== undefined) {
        c.position.y = c.userData.baseY + Math.sin(frameNow * 0.004 + c.userData.phase) * 0.3;
        c.material.opacity = 0.3 + Math.abs(Math.sin(frameNow * 0.004 + c.userData.phase)) * 0.5;
      }
    });
  });

  requestAnimationFrame(animate);'''

html = html.replace('\n  requestAnimationFrame(animate);', animate_insert, 1)

# ═══════════════════════════════════════════
# 11. 修改 bubbles 更新代码（带排队信息）
# ═══════════════════════════════════════════
old_bubble_line = 'bub.div.innerHTML = `${bub.building.label}<br><span class="val">${Math.round(bub.building.density)}</span><span class="unit">人</span>`;'
new_bubble_line = '''const ccap = bub.building.width * bub.building.depth * 12;
      const cr = bub.building.density / Math.max(1, ccap);
      const cqm = Math.round(cr * 45);
      let cqc = '#22c55e';
      if (cr > 0.85) cqc = '#ef4444';
      else if (cr > 0.6) cqc = '#f59e0b';
      bub.div.innerHTML = `${bub.building.label}<br><span class="val">${Math.round(bub.building.density)}</span><span class="unit">人</span><br><span style="font-size:9px;color:${cqc}">🕐~${cqm}分钟</span>`;'''
html = html.replace(old_bubble_line, new_bubble_line, 1)

# ═══════════════════════════════════════════
# 12. 添加 buildingGroups 数组 + 电子围栏创建
# ═══════════════════════════════════════════
html = html.replace(
    'buildingDefs.forEach(def => {',
    'const buildingGroups = [];\n\n  buildingDefs.forEach(def => {', 1)

html = html.replace(
    'const bld = createBuilding(def.x, def.z, def);\n  scene.add(bld);\n  buildings.push({ group: bld, ...def });',
    'const bld = createBuilding(def.x, def.z, def);\n  scene.add(bld);\n  def._group = bld;\n  buildingGroups.push(bld);\n  buildings.push({ group: bld, ...def });', 1)

# ═══════════════════════════════════════════
# 13. 添加建筑点击事件（在 animate 函数定义之后）
# ═══════════════════════════════════════════
click_code = '''
// ── 建筑点击 → 钻取详情 ──
window.addEventListener('click', (e) => {
  if (e.target.closest('#detail-modal') || e.target.closest('#solution-modal')) return;
  const mx = (e.clientX / window.innerWidth) * 2 - 1;
  const my = -(e.clientY / window.innerHeight) * 2 + 1;
  const rc = new THREE.Raycaster();
  rc.setFromCamera(new THREE.Vector2(mx, my), camera);
  const hits = rc.intersectObjects(buildingGroups, true);
  if (hits.length > 0) {
    let o = hits[0].object;
    while (o && !o.userData.building) o = o.parent;
    if (o && o.userData.building) openDetail(o.userData.building, o);
  }
});
'''
html = html.replace('\n// ── Start ──', click_code + '\n// ── Start ──', 1)

# ═══════════════════════════════════════════
# 写入
# ═══════════════════════════════════════════
with open(f'{BASE}/index.html', 'w', encoding='utf-8') as f:
    f.write(html)

import sys
sys.stdout.reconfigure(encoding='utf-8')
print(f"BUILD OK: {len(html)} chars, ~{html.count(chr(10))} lines")