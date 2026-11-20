// Optional enhancements. The page is fully readable without this file: every interactive
// piece starts from static HTML and only gains controls here.
(function () {
  var reduce = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;
  function $(s, r) { return (r || document).querySelector(s); }
  function $$(s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); }
  function press(group, btn) { $$('button', group).forEach(function (b) { b.setAttribute('aria-pressed', b === btn ? 'true' : 'false'); }); }

  // Looping clips stand in for GIFs; with reduced motion, pause them and show controls.
  if (reduce) $$('video[loop]').forEach(function (v) { v.removeAttribute('autoplay'); v.pause(); v.controls = true; });

  // ---- Reveal on scroll; start bar fills and count-ups when they arrive ----
  function arrive(el) { el.classList.add('is-in'); $$('[data-count]', el).forEach(countUp); }
  var targets = $$('.reveal');
  if (!('IntersectionObserver' in window) || reduce) {
    targets.forEach(function (el) { el.classList.add('is-in'); });
  } else {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) { if (e.isIntersecting) { arrive(e.target); io.unobserve(e.target); } });
    }, { threshold: 0.12 });
    targets.forEach(function (el) { io.observe(el); });
  }
  function countUp(el) {
    if (reduce) return;
    var n = parseInt(el.getAttribute('data-count'), 10), of = el.getAttribute('data-of');
    if (!n) return;
    var t0 = null;
    function step(t) {
      if (t0 === null) t0 = t;
      var p = Math.min((t - t0) / 900, 1), v = Math.round(n * (1 - Math.pow(1 - p, 3)));
      el.textContent = of ? v + '/' + of : String(v);
      if (p < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }

  // ---- Work-kind filters (turn steps, architecture) ----
  function filter(chips, items) {
    chips.hidden = false;
    chips.addEventListener('click', function (e) {
      var b = e.target.closest('.chip');
      if (!b) return;
      var kind = b.getAttribute('data-kind');
      press(chips, b);
      items.forEach(function (it) {
        var ks = (it.getAttribute('data-kinds') || '').split(' ');
        it.classList.toggle('is-dim', kind !== 'all' && ks.indexOf(kind) < 0);
      });
    });
  }
  var steps = document.getElementById('turn-steps');
  if (steps) filter($('.chips', steps), $$('.steps > li', steps));
  $$('[data-filter-chips]').forEach(function (chips) {
    var host = chips.closest('section');
    filter(chips, $$('[data-kinds]', host));
  });

  // ---- Game loop: step through a short game ----
  var flow = $('[data-turnflow]');
  if (flow) {
    var play = flow.parentNode.querySelector('.play');
    play.hidden = false;
    var btn = $('[data-turn-next]', play), status = $('.play-status', play);
    var script = [
      ['draw', 'Turn 1: draw a tile from the wall. The hand now has 14.'],
      ['check', 'Turn 1: is it a winning hand? No.'],
      ['discard', 'Turn 1: discard one tile, back to 13. Next turn.'],
      ['draw', 'Turn 2: draw a tile. 14 again.'],
      ['check', 'Turn 2: four sets and a pair? Not yet.'],
      ['discard', 'Turn 2: discard one tile, back to 13.'],
      ['draw', 'Turn 3: draw a tile. 14.'],
      ['check', 'Turn 3: four sets and a pair? Yes!'],
      ['win', 'Win! Real hands take many more turns: 43 on 17 November, 15 on 19 November.']
    ];
    var i = -1;
    btn.addEventListener('click', function () {
      i = (i + 1) % (script.length + 1);
      $$('[data-step]', flow).forEach(function (n) { n.classList.remove('is-on'); });
      if (i === script.length) { i = -1; status.textContent = 'Step through a short game, one move at a time.'; btn.textContent = 'Play a turn'; return; }
      var s = script[i];
      $('[data-step="' + s[0] + '"]', flow).classList.add('is-on');
      status.textContent = s[1];
      btn.textContent = i === script.length - 1 ? 'Play again' : 'Next step';
    });
  }

  // ---- The empty seat ----
  var seat = $('#empty-seat');
  if (seat) {
    var seg = $('.seg', seat); seg.hidden = false;
    var states = {
      visit: { label: 'Once a year', verdict: 'Four players. Game on.', note: 'My grandparents are in town from Malaysia, so we have enough kaki to play.', fill: {} },
      year: { label: 'The rest of the year', verdict: 'One seat short.', note: 'My grandparents are home in Malaysia. Dad, my sister and I make three.', fill: { gm: 'me', gp: 'empty' } },
      remote: { label: 'The goal', verdict: 'Robotto takes the empty seat.', note: 'My grandparents play through it, remote controlling the moves from Malaysia while the arm plays here. Not yet built.', fill: { gm: 'me', gp: 'remote' } },
      auto: { label: 'The goal', verdict: 'Robotto takes the empty seat.', note: 'Or it plays its own hand as an autonomous extra player, so there are always enough kaki. Not yet: today it plays mahjong solitaire.', fill: { gm: 'me', gp: 'auto' } }
    };
    var base = { gm: ['G', 'Grandma'], gp: ['G', 'Grandpa'] };
    var look = { me: ['E', 'Me'], empty: ['?', 'Empty'], remote: ['R', 'Robotto, steered from Malaysia'], auto: ['R', 'Robotto, on its own'] };
    seg.addEventListener('click', function (e) {
      var b = e.target.closest('button'); if (!b) return;
      press(seg, b);
      var st = states[b.getAttribute('data-seat')];
      $$('.seat', seat).forEach(function (s) {
        var who = s.getAttribute('data-who'); if (!base[who]) return;
        var f = st.fill[who], l = f ? look[f] : base[who];
        s.classList.toggle('is-empty', f === 'empty'); s.classList.toggle('is-robot', f === 'remote' || f === 'auto');
        s.classList.toggle('is-me', f === 'me');
        $('.dot', s).textContent = l[0]; $('.nm', s).textContent = l[1];
      });
      $('[data-seat-label]', seat).textContent = st.label;
      $('[data-seat-verdict]', seat).textContent = st.verdict;
      $('[data-seat-note]', seat).textContent = st.note;
    });
  }

  // ---- Interventions: log a hand ----
  var sim = $('[data-sim]');
  if (sim) {
    var budget = $('[data-budget]', sim), asksEl = $('[data-asks]', sim), verdict = $('[data-verdict]', sim);
    var n = 0, asks = 0, audit = false, recov = 0;
    function render(last) {
      var tiles = $$('.mt', budget);
      while (tiles.length < Math.max(3, n)) { var t = document.createElement('span'); t.className = 'mt sm'; budget.appendChild(t); tiles.push(t); }
      while (tiles.length > Math.max(3, n)) { budget.removeChild(tiles.pop()); }
      tiles.forEach(function (t, k) { t.className = 'mt sm' + (k < n ? (k < 3 ? ' used' : ' over') : ''); });
      asksEl.textContent = asks;
      var bad = n > 3 || audit;
      verdict.classList.toggle('bad', bad);
      var why = audit ? 'Not clean: the audit found a mismatch. That’s a silent error.' :
        n > 3 ? 'Not clean: ' + n + ' interventions, the limit is 3.' :
        'Clean so far: ' + n + ' of 3 interventions used' + (asks ? ', ' + asks + (asks === 1 ? ' ask' : ' asks') + ' counted but not charged' : '') + (recov ? ', ' + recov + (recov === 1 ? ' recovery' : ' recoveries') + ' by the robot' : '') + '.';
      verdict.textContent = (last ? last + ' logged. ' : '') + why;
    }
    $('.sim-btns', sim).addEventListener('click', function (e) {
      var b = e.target.closest('button'); if (!b) return;
      var c = b.getAttribute('data-counts');
      if (c === 'reset') { n = 0; asks = 0; audit = false; recov = 0; render(); return; }
      if (c === 'yes') n++; else if (c === 'ask') asks++; else if (c === 'no') recov++; else if (c === 'audit') audit = true;
      render(b.getAttribute('data-name'));
    });
  }

  // ---- Timeline: a wall of face-down tiles. Turn one over to read it ----
  var list = $('[data-wall]');
  if (list) {
    var wall = $('.wall'), detail = $('.wall-detail'), btns = [];
    $$('li', list).forEach(function (li) {
      var b = document.createElement('button');
      b.type = 'button'; b.className = 'tile-btn'; b.setAttribute('aria-pressed', 'false');
      b.setAttribute('aria-label', li.getAttribute('data-when') + ': ' + li.getAttribute('data-short'));
      b.innerHTML = '<span class="flip" aria-hidden="true"><span class="back">' + li.getAttribute('data-when') + '</span>' +
        '<span class="face"><span>' + li.getAttribute('data-short') + '</span><i class="k-' + li.getAttribute('data-kind') + '"></i></span></span>';
      b.addEventListener('click', function () {
        var on = b.getAttribute('aria-pressed') === 'true';
        btns.forEach(function (x) { x.setAttribute('aria-pressed', 'false'); });
        if (on) { detail.hidden = true; return; }
        b.setAttribute('aria-pressed', 'true');
        detail.innerHTML = li.innerHTML.replace('class="mj-card"', '');
        detail.hidden = false;
      });
      btns.push(b); wall.appendChild(b);
    });
    // Arrow keys move along the wall.
    wall.addEventListener('keydown', function (e) {
      var k = btns.indexOf(document.activeElement); if (k < 0) return;
      var to = e.key === 'ArrowRight' ? k + 1 : e.key === 'ArrowLeft' ? k - 1 : e.key === 'Home' ? 0 : e.key === 'End' ? btns.length - 1 : null;
      if (to === null) return;
      e.preventDefault(); var nb = btns[Math.max(0, Math.min(btns.length - 1, to))];
      btns.forEach(function (x) { x.tabIndex = x === nb ? 0 : -1; }); nb.focus();
    });
    btns[btns.length - 3].click();
    btns.forEach(function (x, k) { x.tabIndex = k === btns.length - 3 ? 0 : -1; });
    wall.setAttribute('role', 'group'); wall.setAttribute('aria-label', 'Milestones, June to November. Use the arrow keys to move between tiles.');
  }

  // ---- Glossary: the first use of each term gets a definition on click ----
  var gdata = $('#glossary-data'), pop = $('#gloss-pop'), article = $('article');
  if (gdata && pop && article) {
    var terms = [];
    try { terms = JSON.parse(gdata.textContent) || []; } catch (e) { terms = []; }
    var skip = 'a, button, code, h1, h2, h3, h4, figure, figcaption, .ig, .felt-panel, .bars, .read-next, .page-title';
    var blocks = $$('p, li, blockquote', article).filter(function (el) { return !el.closest(skip); });
    var pairs = [];
    terms.forEach(function (t) { t.match.forEach(function (m) { pairs.push({ t: t, m: m }); }); });
    pairs.sort(function (a, b) { return b.m.length - a.m.length; });
    var done = {}, gid = 0;
    function esc(s) { return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); }
    pairs.forEach(function (p) {
      if (done[p.t.term]) return;
      var re = new RegExp('(^|[^A-Za-z0-9-])(' + esc(p.m) + ')(?![A-Za-z0-9])');
      for (var bi = 0; bi < blocks.length && !done[p.t.term]; bi++) {
        var walker = document.createTreeWalker(blocks[bi], NodeFilter.SHOW_TEXT, null);
        var node;
        while ((node = walker.nextNode())) {
          if (node.parentNode.closest('.gloss, a, code, button')) continue;
          var mm = re.exec(node.nodeValue);
          if (!mm) continue;
          var start = mm.index + mm[1].length;
          var rest = node.splitText(start); rest.splitText(mm[2].length);
          var g = document.createElement('button');
          g.type = 'button'; g.className = 'gloss'; g.setAttribute('aria-expanded', 'false');
          g.id = 'gloss-' + (++gid); g.textContent = rest.nodeValue;
          g.setAttribute('data-term', p.t.term); g.setAttribute('data-def', p.t.def);
          rest.parentNode.replaceChild(g, rest);
          done[p.t.term] = true;
          break;
        }
      }
    });
    var open = null;
    function close(refocus) {
      if (!open) return;
      open.setAttribute('aria-expanded', 'false'); open.removeAttribute('aria-describedby');
      pop.hidden = true;
      if (refocus) open.focus();
      open = null;
    }
    function show(g) {
      close();
      pop.innerHTML = '';
      var b = document.createElement('b'); b.textContent = g.getAttribute('data-term');
      pop.appendChild(b); pop.appendChild(document.createTextNode(g.getAttribute('data-def')));
      pop.hidden = false;
      var r = g.getBoundingClientRect(), w = pop.offsetWidth;
      var left = Math.max(12, Math.min(r.left + window.scrollX, window.scrollX + document.documentElement.clientWidth - w - 12));
      var top = r.bottom + window.scrollY + 8;
      if (r.bottom + pop.offsetHeight + 16 > window.innerHeight && r.top > pop.offsetHeight + 80) top = r.top + window.scrollY - pop.offsetHeight - 8;
      pop.style.left = left + 'px'; pop.style.top = top + 'px';
      g.setAttribute('aria-expanded', 'true'); g.setAttribute('aria-describedby', 'gloss-pop');
      open = g;
    }
    document.addEventListener('click', function (e) {
      var g = e.target.closest('.gloss');
      if (g) { if (open === g) close(); else show(g); return; }
      if (!e.target.closest('#gloss-pop')) close();
    });
    document.addEventListener('keydown', function (e) { if (e.key === 'Escape' && open) close(true); });
    window.addEventListener('resize', function () { close(); });
  }

  // ---- Reading progress: a hand of 14 tiles turns face up as you read ----
  var hand = $('[data-progress]');
  if (hand && article) {
    var tiles = $$('i', hand), ticking = false;
    function update() {
      ticking = false;
      var r = article.getBoundingClientRect(), total = r.height - window.innerHeight;
      var p = total > 0 ? Math.min(1, Math.max(0, -r.top / total)) : 1;
      var k = Math.round(p * tiles.length);
      tiles.forEach(function (t, i) { t.classList.toggle('on', i < k); });
      hand.classList.toggle('won', k === tiles.length);
    }
    window.addEventListener('scroll', function () { if (!ticking) { ticking = true; requestAnimationFrame(update); } }, { passive: true });
    window.addEventListener('resize', update);
    update();
  }
})();
