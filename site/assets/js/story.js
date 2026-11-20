// /story/: the hand fills as you read. The state is a pure function of scroll position, so scrolling
// back up un-draws tiles and a link to #draw-9 opens with nine tiles. Loaded after site.js.
(function () {
  var reduce = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;
  function $(s, r) { return (r || document).querySelector(s); }
  function $$(s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); }

  var draws = $$('.draw'), rack = $('[data-rack]');
  if (!draws.length || !rack) return;
  var slots = $$('.slot', rack), sets = $$('.set', rack), wall = $$('.stage-wall .mt');
  var countEl = $('[data-stage-count]'), actEl = $('[data-stage-act]'), winEl = $('[data-stage-win]'), live = $('[data-stage-live]');
  var meta = draws.map(function (d) {
    var head = $('.draw-head', d);
    return { act: ($('.mj-eyebrow', head).textContent.split('·')[1] || '').trim(), title: $('h2', head).textContent };
  });
  var shown = -1;

  function current() {
    var vh = window.innerHeight, n = 0;
    draws.forEach(function (d, i) { if (d.getBoundingClientRect().top <= vh * 0.5) n = i + 1; });
    if (window.scrollY + vh >= document.documentElement.scrollHeight - 4) n = draws.length;  // the last, short draw
    return n;
  }

  function render(n) {
    if (n === shown) return;
    shown = n;
    slots.forEach(function (s) {
      var k = +s.getAttribute('data-n');
      s.classList.toggle('drawn', k <= n);
      s.classList.toggle('current', k === n);
      if (k === n) s.setAttribute('aria-current', 'step'); else s.removeAttribute('aria-current');
    });
    sets.forEach(function (st) { st.classList.toggle('done', $$('.slot', st).every(function (s) { return s.classList.contains('drawn'); })); });
    wall.forEach(function (t, i) { t.classList.toggle('gone', i >= wall.length - n); });
    draws.forEach(function (d, i) { d.classList.toggle('is-current', i + 1 === n); });
    var won = n === draws.length;
    rack.classList.toggle('won', won);
    if (winEl) winEl.hidden = !won;
    if (n === 0) { countEl.textContent = 'The deal'; actEl.textContent = 'Scroll to draw your first tile'; }
    else { countEl.textContent = 'Draw ' + n + ' of ' + draws.length + ' · ' + meta[n - 1].act; actEl.textContent = meta[n - 1].title; }
    if (live && n > 0) live.textContent = won ? 'Draw 14 of 14. Four sets and a pair: a winning hand.' : 'Draw ' + n + ' of ' + draws.length + ': ' + meta[n - 1].title;
  }

  // ---- Set-pieces that step forward as you read through them ----
  var scrubs = [];
  function scrub(hostSel, itemSel, onStep) {
    var host = $(hostSel); if (!host) return;
    var items = $$(itemSel, host); if (!items.length) return;
    host.classList.add('scrub');
    items.forEach(function (it) { it.setAttribute('data-scrub-item', ''); });
    scrubs.push({ host: host, items: items, k: -1, onStep: onStep });
  }
  function stepScrubs() {
    var vh = window.innerHeight;
    scrubs.forEach(function (s) {
      var r = s.host.getBoundingClientRect();
      // An item lights as it passes 65% of the way down the screen, so the sequence plays as you read it.
      var p = (vh * 0.65 - r.top) / Math.max(r.height, 1);
      var k = Math.max(0, Math.min(s.items.length, Math.ceil(p * s.items.length)));
      if (k === s.k) return;
      s.k = k;
      s.items.forEach(function (it, i) { it.classList.toggle('lit', i < k); });
      if (s.onStep) s.onStep(k, s.items);
    });
  }
  if (!reduce) {
    scrub('#turn-steps', '.steps > li');
    scrub('#timeline', '.wall .tile-btn', function (k, btns) {
      var b = btns[k - 1];
      if (b && b.getAttribute('aria-pressed') !== 'true') b.click();
    });
    scrub('.bars', '.bar');
    scrub('#merge-flow', '.mq-card');
  }

  // ---- Scroll loop ----
  var ticking = false;
  function update() { ticking = false; render(current()); stepScrubs(); }
  window.addEventListener('scroll', function () { if (!ticking) { ticking = true; requestAnimationFrame(update); } }, { passive: true });
  window.addEventListener('resize', function () { shown = -1; update(); });
  window.addEventListener('hashchange', function () { shown = -1; update(); });
  update();

  // ---- The rack as navigation: one tab stop, arrows move between tiles ----
  slots.forEach(function (s, i) { s.tabIndex = i === 0 ? 0 : -1; });
  rack.addEventListener('keydown', function (e) {
    var k = slots.indexOf(document.activeElement); if (k < 0) return;
    var to = e.key === 'ArrowRight' || e.key === 'ArrowDown' ? k + 1 : e.key === 'ArrowLeft' || e.key === 'ArrowUp' ? k - 1 :
      e.key === 'Home' ? 0 : e.key === 'End' ? slots.length - 1 : null;
    if (to === null) return;
    e.preventDefault();
    var t = slots[Math.max(0, Math.min(slots.length - 1, to))];
    slots.forEach(function (x) { x.tabIndex = x === t ? 0 : -1; });
    t.focus();
  });
  rack.addEventListener('click', function (e) {
    var s = e.target.closest('.slot'); if (!s) return;
    var target = document.getElementById('draw-' + s.getAttribute('data-n'));
    if (target) { setTimeout(function () { var h = $('h2', target); if (h) { h.tabIndex = -1; h.focus({ preventScroll: true }); } }, reduce ? 0 : 450); }
  });
})();
