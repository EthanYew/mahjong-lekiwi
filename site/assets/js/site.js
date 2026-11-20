// Optional enhancements. The page is fully readable without this file.
(function () {
  var reduce = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;

  // Looping clips stand in for GIFs; with reduced motion, pause them and show controls.
  if (reduce) {
    document.querySelectorAll('video[loop]').forEach(function (v) { v.removeAttribute('autoplay'); v.pause(); v.controls = true; });
  }

  // Reveal on scroll, and start bar fills / count-ups when they arrive.
  var targets = document.querySelectorAll('.reveal');
  function arrive(el) {
    el.classList.add('is-in');
    el.querySelectorAll('[data-count]').forEach(countUp);
  }
  if (!('IntersectionObserver' in window) || reduce) {
    targets.forEach(function (el) { el.classList.add('is-in'); });
  } else {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) { if (e.isIntersecting) { arrive(e.target); io.unobserve(e.target); } });
    }, { threshold: 0.15 });
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

  // Turn steps: highlight one kind of work.
  var steps = document.getElementById('turn-steps');
  if (steps) {
    var chips = steps.querySelector('.chips');
    chips.hidden = false;
    var items = steps.querySelectorAll('.steps > li');
    chips.addEventListener('click', function (e) {
      var b = e.target.closest('.chip');
      if (!b) return;
      var kind = b.getAttribute('data-kind');
      chips.querySelectorAll('.chip').forEach(function (c) { c.setAttribute('aria-pressed', c === b ? 'true' : 'false'); });
      items.forEach(function (li) {
        var match = kind === 'all' || li.getAttribute('data-kinds').split(' ').indexOf(kind) > -1;
        li.classList.toggle('is-dim', !match);
      });
    });
  }

  // Timeline: a wall of face-down tiles. Turn one over to read it.
  var list = document.querySelector('[data-wall]');
  if (list) {
    var wall = document.querySelector('.wall'), detail = document.querySelector('.wall-detail');
    var btns = [];
    list.querySelectorAll('li').forEach(function (li, i) {
      var b = document.createElement('button');
      b.type = 'button'; b.className = 'tile-btn'; b.setAttribute('aria-pressed', 'false');
      b.setAttribute('aria-label', li.getAttribute('data-when') + ': ' + li.getAttribute('data-short'));
      var kind = li.getAttribute('data-kind');
      b.innerHTML = '<span class="flip"><span class="back">' + li.getAttribute('data-when') + '</span>' +
        '<span class="face"><span>' + li.getAttribute('data-short') + '</span><i class="k-' + kind + '"></i></span></span>';
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
    // Show the latest milestone to begin with, so the panel is never empty.
    btns[btns.length - 3].click();
  }
})();
