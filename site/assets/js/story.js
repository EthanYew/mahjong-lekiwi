// /story/: set-pieces that step forward as you read through them. The hand itself is hand.js.
// Each graphic starts as its top passes 85% of the way down the screen and has fully played by the time
// it sits centred on screen (for a short graphic, when all of it is in view). Once lit, items stay lit:
// scrolling back up doesn't replay anything. Skipped with reduced motion, so everything shows at once.
(function () {
  if (window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  function $(s, r) { return (r || document).querySelector(s); }
  function $$(s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); }

  var scrubs = [];
  function scrub(hostSel, itemSel, onStep) {
    var host = $(hostSel); if (!host) return;
    var items = $$(itemSel, host); if (!items.length) return;
    host.classList.add('scrub');
    items.forEach(function (it) { it.setAttribute('data-scrub-item', ''); });
    scrubs.push({ host: host, items: items, k: 0, onStep: onStep });
  }
  scrub('#turn-steps', '.steps > li');
  scrub('#timeline', '.wall .tile-btn', function (k, btns) {
    var b = btns[k - 1];
    if (b && b.getAttribute('aria-pressed') !== 'true') b.click();
  });
  scrub('.bars', '.bar');
  scrub('#merge-flow', '.mq-card');

  function step() {
    var vh = window.innerHeight;
    scrubs.forEach(function (s) {
      if (s.k >= s.items.length) return;
      var r = s.host.getBoundingClientRect();
      var start = vh * 0.85;
      var end = Math.min(vh * 0.5 - r.height / 2, start - vh * 0.2);  // centred on screen, or at least a fifth of a screen of scrolling
      var p = (start - r.top) / (start - end);
      var k = Math.max(0, Math.min(s.items.length, Math.ceil(p * s.items.length)));
      if (k <= s.k) return;  // never undo
      s.k = k;
      s.items.forEach(function (it, i) { if (i < k) it.classList.add('lit'); });
      if (s.onStep) s.onStep(k, s.items);
    });
  }
  var ticking = false;
  window.addEventListener('scroll', function () { if (!ticking) { ticking = true; requestAnimationFrame(function () { ticking = false; step(); }); } }, { passive: true });
  window.addEventListener('resize', step);
  step();
})();
