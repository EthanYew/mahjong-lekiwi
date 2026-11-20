// The reading hand (_includes/hand-rack.html): 14 tiles, one per section, that fill as you read.
// The state is a pure function of scroll position, so scrolling back up un-draws tiles and a #link opens
// with the right count. Each tile links to its section; that link is also what the tile tracks.
//   rail  (/story/): always open beside the story; Hide leaves a "Show hand" button. Remembered per device.
//   strip (/):       a column of small tiles that opens into the rail on hover or focus; "Keep open" pins it.
(function () {
  var reduce = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;
  function $(s, r) { return (r || document).querySelector(s); }
  function $$(s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); }
  function store(k, v) { try { localStorage.setItem(k, v); } catch (e) {} }

  var stage = $('.stage'), rack = $('[data-rack]');
  if (!stage || !rack) return;
  var root = document.documentElement, strip = stage.classList.contains('stage--strip');
  var unit = stage.getAttribute('data-unit') || 'Draw';
  var slots = $$('.slot', rack), sets = $$('.set', rack), wall = $$('.stage-wall .mt');
  var targets = slots.map(function (s) { return document.getElementById(s.getAttribute('href').slice(1)); });
  var meta = slots.map(function (s) { return { act: s.getAttribute('data-act'), title: s.getAttribute('data-title') }; });
  var N = slots.length;
  var countEl = $('[data-stage-count]'), actEl = $('[data-stage-act]'), winEl = $('[data-stage-win]'), live = $('[data-stage-live]');
  var head = $('[data-stage-head]'), mini = $('[data-stage-mini]');
  var shown = -1, peeking = 0;

  function current() {
    var vh = window.innerHeight, n = 0;
    targets.forEach(function (t, i) { if (t && t.getBoundingClientRect().top <= vh * 0.5) n = i + 1; });
    if (window.scrollY + vh >= document.documentElement.scrollHeight - 4) n = N;  // the last, short section
    return n;
  }

  function label(n) {
    if (n === 0) {
      countEl.textContent = strip ? 'Contents' : 'The deal';
      actEl.textContent = strip ? 'Scroll to fill the hand' : 'Scroll to draw your first tile';
    } else { countEl.textContent = unit + ' ' + n + ' of ' + N + ' · ' + meta[n - 1].act; actEl.textContent = meta[n - 1].title; }
  }

  function render(n) {
    if (n === shown) return;
    shown = n;
    slots.forEach(function (s, i) {
      var k = i + 1;
      s.classList.toggle('drawn', k <= n);
      s.classList.toggle('current', k === n);
      if (k === n) s.setAttribute('aria-current', 'step'); else s.removeAttribute('aria-current');
    });
    sets.forEach(function (st) { st.classList.toggle('done', $$('.slot', st).every(function (s) { return s.classList.contains('drawn'); })); });
    wall.forEach(function (t, i) { t.classList.toggle('gone', i >= wall.length - n); });
    targets.forEach(function (t, i) { if (t) t.classList.toggle('is-current', i + 1 === n); });
    var won = n === N;
    rack.classList.toggle('won', won);
    if (winEl) winEl.hidden = !won;
    if (mini) mini.textContent = n + '/' + N;
    var p = peeking && $('.slot.peek', rack);
    if (p) peek(p); else label(n);  // a hovered tile's "Go to" or "You are here" follows the scroll too
    if (live && n > 0) live.textContent = won ? unit + ' 14 of 14. Four sets and a pair: a winning hand.' : unit + ' ' + n + ' of ' + N + ': ' + meta[n - 1].title;
  }

  var ticking = false;
  function update() { ticking = false; render(current()); }
  window.addEventListener('scroll', function () { if (!ticking) { ticking = true; requestAnimationFrame(update); } }, { passive: true });
  window.addEventListener('resize', function () { shown = -1; update(); });
  window.addEventListener('hashchange', function () { shown = -1; update(); });
  window.addEventListener('load', function () { shown = -1; update(); });
  update();

  // ---- Jumping: focus lands on the section's heading once the scroll has arrived ----
  function goTo(n) {
    var t = targets[n - 1], h = t && (/^H[1-6]$/.test(t.tagName) ? t : $('h2', t));
    if (h) setTimeout(function () { if (!h.hasAttribute('tabindex')) h.tabIndex = -1; h.focus({ preventScroll: true }); }, reduce ? 0 : 450);
  }

  // ---- The rack as navigation: one tab stop, arrows move between tiles ----
  slots.forEach(function (s, i) { s.tabIndex = i === 0 ? 0 : -1; });
  rack.addEventListener('keydown', function (e) {
    var k = slots.indexOf(document.activeElement); if (k < 0) return;
    var to = e.key === 'ArrowRight' || e.key === 'ArrowDown' ? k + 1 : e.key === 'ArrowLeft' || e.key === 'ArrowUp' ? k - 1 :
      e.key === 'Home' ? 0 : e.key === 'End' ? N - 1 : null;
    if (to === null) return;
    e.preventDefault();
    var t = slots[Math.max(0, Math.min(N - 1, to))];
    slots.forEach(function (x) { x.tabIndex = x === t ? 0 : -1; });
    t.focus();
  });
  rack.addEventListener('click', function (e) {
    var s = e.target.closest('.slot'); if (s) { unpeek(); goTo(+s.getAttribute('data-n')); }
  });

  // ---- Peek: hovering or tabbing to a tile plays its draw and shows its title, before you jump ----
  function peek(s) {
    var n = +s.getAttribute('data-n');
    peeking = n;
    slots.forEach(function (x) { x.classList.toggle('peek', x === s); });
    sets.forEach(function (st) { st.classList.toggle('peek', st.contains(s)); });
    head.classList.add('peeking');
    countEl.textContent = (n === shown ? 'You are here · ' : 'Go to ' + unit.toLowerCase() + ' ' + n + ' · ') + meta[n - 1].act;
    actEl.textContent = meta[n - 1].title;
  }
  function unpeek() {
    if (!peeking) return;
    peeking = 0;
    slots.forEach(function (x) { x.classList.remove('peek'); });
    sets.forEach(function (st) { st.classList.remove('peek'); });
    head.classList.remove('peeking');
    label(shown);
  }
  rack.addEventListener('pointerover', function (e) {
    var s = e.target.closest('.slot'); if (s && e.pointerType !== 'touch') peek(s);
  });
  rack.addEventListener('pointerleave', function () {
    var f = document.activeElement;
    if (rack.contains(f) && f.matches(':focus-visible')) peek(f); else unpeek();
  });
  rack.addEventListener('focusin', function (e) {
    var s = e.target.closest('.slot'); if (s && s.matches(':focus-visible')) peek(s);
  });
  rack.addEventListener('focusout', function (e) { if (!rack.contains(e.relatedTarget)) unpeek(); });

  // ---- Contents: all 14 sections at a glance ----
  var toc = $('[data-toc]'), tocLinks = toc ? $$('a[data-n]', toc) : [], tocRead = toc && $('[data-toc-read]', toc);
  function openToc() {
    if (!toc || !toc.showModal) return;
    var cur = null;
    tocLinks.forEach(function (a) {
      var k = +a.getAttribute('data-n');
      a.classList.toggle('read', k <= shown);
      if (k === shown) { a.setAttribute('aria-current', 'location'); cur = a; } else a.removeAttribute('aria-current');
    });
    var noun = unit.toLowerCase() === 'draw' ? 'drawn' : 'read';
    if (tocRead) tocRead.textContent = shown === 0 ? 'Nothing ' + noun + ' yet. Pick any tile to start there.' :
      shown === N ? 'All 14 ' + noun + ': a winning hand.' : 'You have ' + noun + ' ' + shown + ' of ' + N + '.';
    toc.showModal();
    (cur || tocLinks[0]).focus();
  }
  $$('[data-toc-open]').forEach(function (b) { b.addEventListener('click', openToc); });
  if (toc) {
    toc.addEventListener('click', function (e) {
      if (e.target === toc || e.target.closest('[data-toc-close]')) { toc.close(); return; }
      var a = e.target.closest('a[data-n]'); if (!a) return;
      e.preventDefault();
      toc.close();
      location.hash = a.getAttribute('href').slice(1);
      goTo(+a.getAttribute('data-n'));
    });
  }

  if (strip) {
    // ---- Strip: opens on hover or keyboard focus, closes when you leave; "Keep open" pins it ----
    var pinBtn = $('[data-stage-pin]'), keepBtn = $('[data-stage-keep]'), timer = null;
    var pinned = root.classList.contains('hand-pinned');
    function setOpen(open) {
      stage.classList.toggle('is-open', open);
      pinBtn.setAttribute('aria-expanded', String(open));
      if (!open) unpeek();
    }
    function setPinned(p, save) {
      pinned = p;
      root.classList.toggle('hand-pinned', p);
      keepBtn.textContent = p ? 'Hide' : 'Keep open';
      keepBtn.setAttribute('aria-pressed', String(p));
      if (save) store('mj-hand-open', p ? 'pinned' : 'closed');
      shown = -1; requestAnimationFrame(update);  // pinning moves the article, so recount
    }
    setPinned(pinned, false);
    setOpen(pinned);
    function maybeClose() {
      var f = document.activeElement;
      if (pinned || (stage.contains(f) && f.matches(':focus-visible')) || (toc && toc.open)) return;
      setOpen(false);
    }
    stage.addEventListener('pointerenter', function (e) {
      if (e.pointerType === 'touch') return;
      clearTimeout(timer); timer = setTimeout(function () { setOpen(true); }, 90);
    });
    stage.addEventListener('pointerleave', function (e) {
      if (e.pointerType === 'touch') return;
      clearTimeout(timer); timer = setTimeout(maybeClose, 280);
    });
    stage.addEventListener('focusin', function (e) { if (e.target.closest('.slot') && e.target.matches(':focus-visible')) setOpen(true); });
    stage.addEventListener('focusout', function (e) { if (!stage.contains(e.relatedTarget)) setTimeout(maybeClose, 0); });
    stage.addEventListener('keydown', function (e) { if (e.key === 'Escape' && !pinned) { setOpen(false); pinBtn.focus(); } });
    pinBtn.addEventListener('click', function () { setPinned(true, true); setOpen(true); keepBtn.focus(); });
    keepBtn.addEventListener('click', function () {
      if (pinned) { setPinned(false, true); setOpen(false); pinBtn.focus(); }
      else { setPinned(true, true); }
    });
    if (toc) toc.addEventListener('close', function () { setTimeout(maybeClose, 0); });
  } else {
    // ---- Rail: Hide leaves a "Show hand" button ----
    var hideBtn = $('[data-stage-hide]'), showBtn = $('[data-stage-show]');
    function setOff(off, save) {
      root.classList.toggle('stage-off', off);
      if (showBtn) showBtn.hidden = !off;
      if (hideBtn) hideBtn.setAttribute('aria-expanded', String(!off));
      if (save) store('mj-story-hand', off ? 'off' : 'on');
    }
    setOff(root.classList.contains('stage-off'), false);
    if (hideBtn) hideBtn.addEventListener('click', function () { setOff(true, true); showBtn.focus(); });
    if (showBtn) showBtn.addEventListener('click', function () { setOff(false, true); hideBtn.focus(); });
  }
})();
