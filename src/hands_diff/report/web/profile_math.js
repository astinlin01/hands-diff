/*
 * Browser copy of the profile math, so the static dashboard can recompute everything when
 * the game-window / champion filters change. Each function mirrors a tested Python
 * function (named in its comment); tests/test_dashboard_math.py runs both on the same
 * inputs through Node and requires matching answers.
 *
 * Plain ES5-style JavaScript with no dependencies: build_dashboard.py inlines this file
 * into the page, and Node loads it via module.exports for the tests.
 */
(function (root) {
  'use strict';

  function isNum(v) {
    return v !== null && v !== undefined && typeof v === 'number' && isFinite(v);
  }

  function mean(values) {
    var nums = values.filter(isNum);
    if (!nums.length) return null;
    return nums.reduce(function (a, b) { return a + b; }, 0) / nums.length;
  }

  function sampleStd(values) {
    var nums = values.filter(isNum);
    if (nums.length < 2) return null;
    var m = mean(nums);
    var ss = nums.reduce(function (a, v) { return a + (v - m) * (v - m); }, 0);
    return Math.sqrt(ss / (nums.length - 1));
  }

  // Percentile of `value` among reference player averages, given only evenly spaced
  // percentile cutoffs (breakpoints[i] = the (100*i/(n-1))th percentile). Linear
  // between cutoffs. A value equal to a run of identical cutoffs (common for stats that
  // are often exactly 0) lands in the middle of the run -- ties count half, like
  // scipy's percentileofscore(kind="mean") on the Python side.
  function percentileFromBreakpoints(value, breakpoints) {
    var n = breakpoints.length;
    var step = 100 / (n - 1);
    if (value < breakpoints[0]) return 0;
    if (value > breakpoints[n - 1]) return 100;
    var first = -1, last = -1;
    for (var i = 0; i < n; i++) {
      if (breakpoints[i] === value) {
        if (first < 0) first = i;
        last = i;
      }
    }
    if (first >= 0) return ((first + last) / 2) * step;
    for (var j = 0; j < n - 1; j++) {
      if (value < breakpoints[j + 1]) {
        var lo = breakpoints[j], hi = breakpoints[j + 1];
        return (j + (value - lo) / (hi - lo)) * step;
      }
    }
    return 100;
  }

  // One player's average of each area over the games that have it, plus how many games
  // that was (performance_shape.player_area_averages, for one player).
  function areaProfile(games, areas) {
    var profile = {}, counts = {};
    areas.forEach(function (area) {
      var values = games.map(function (g) { return g[area]; }).filter(isNum);
      counts[area] = values.length;
      profile[area] = values.length ? mean(values) : null;
    });
    return { profile: profile, counts: counts };
  }

  function ownSampleStatus(games) {
    if (games < 5) return 'insufficient';
    return games < 10 ? 'low' : 'ok';
  }

  // champion_pool.effective_champions: 1 / unbiased HHI, capped at champions played.
  function effectiveChampions(counts) {
    var values = Object.keys(counts).map(function (k) { return counts[k]; }).filter(function (v) { return v > 0; });
    var total = values.reduce(function (a, b) { return a + b; }, 0);
    if (total < 2) return null;
    var samePairs = values.reduce(function (a, n) { return a + n * (n - 1); }, 0);
    if (samePairs === 0) return values.length;
    return Math.min(total * (total - 1) / samePairs, values.length);
  }

  // champion_pool.top_champion: [name, share]; first-listed wins ties.
  function topChampion(counts) {
    var best = null, total = 0;
    Object.keys(counts).forEach(function (name) {
      total += counts[name];
      if (best === null || counts[name] > counts[best]) best = name;
    });
    return best === null ? null : [best, counts[best] / total];
  }

  // champion_pool.class_mix, from each game's already-resolved class label.
  function classMix(labels) {
    var mix = {};
    labels.forEach(function (label) { mix[label] = (mix[label] || 0) + 1; });
    Object.keys(mix).forEach(function (label) { mix[label] /= labels.length; });
    return mix;
  }

  // performance.biggest_gaps. curves[stat] = {breakpoints, median} of the reference
  // players' averages; gapFeatures[stat] = [theme, direction].
  var GAP_INTERVAL_Z = 1.2816;  // 80% two-sided
  function biggestGaps(games, curves, gapFeatures, top) {
    var bestPerTheme = {}, themeOrder = [];
    Object.keys(gapFeatures).forEach(function (feature) {
      var theme = gapFeatures[feature][0], direction = gapFeatures[feature][1];
      var curve = curves[feature];
      if (!curve) return;
      var values = games.map(function (g) { return g[feature]; }).filter(isNum);
      if (!values.length) return;
      var avg = mean(values);
      var raw = percentileFromBreakpoints(avg, curve.breakpoints);
      var percentile = direction > 0 ? raw : 100 - raw;
      if (percentile >= 50) return;
      var sd = sampleStd(values);
      var se = sd === null ? Infinity : sd / Math.sqrt(values.length);
      if (Math.abs(avg - curve.median) <= GAP_INTERVAL_Z * se) return;  // within noise
      var gap = {
        feature: feature, theme: theme, percentile: percentile,
        your_avg: avg, reference_median: curve.median, n_games: values.length
      };
      if (!(theme in bestPerTheme)) themeOrder.push(theme);
      if (!(theme in bestPerTheme) || percentile < bestPerTheme[theme].percentile) bestPerTheme[theme] = gap;
    });
    var gaps = themeOrder.map(function (t) { return bestPerTheme[t]; });
    gaps.sort(function (a, b) { return a.percentile - b.percentile; });
    return gaps.slice(0, top === undefined ? 3 : top);
  }

  // performance.gap_size: a gap in game terms. Per-minute stats -> per game of the role's
  // typical length; 0-1 shares -> percentage points; else the plain difference.
  function gapSize(feature, yourAvg, median, gameMinutes, percentFeatures) {
    var diff = Math.abs(median - yourAvg);
    if (/_per_min(ute)?$/.test(feature)) return { amount: diff * gameMinutes, per_game: true };
    if ((percentFeatures || []).indexOf(feature) !== -1) return { amount: diff * 100, per_game: false };
    return { amount: diff, per_game: false };
  }

  // ---------- Performance shape ----------
  // How much a player's champions move each area (mirrors performance_shape.champion_effect).
  // games/hands: per-game {area: score} in the two views, same order as champions.
  // effect_sd: mean per-game (games - hands), in units of the reference players' spread
  // (sd of player averages) -- SD rather than percentile points, which squeeze near 0/100.
  // driver: champion pushing hardest in the direction of the total; driver_share can
  // exceed 1 when other champions pull the other way.
  function championEffect(games, hands, champions, sdByArea, areas) {
    var out = {};
    areas.forEach(function (area) {
      var diffs = [], byChampion = {}, total = 0;
      games.forEach(function (g, i) {
        var a = g[area], b = hands[i][area];
        if (!isNum(a) || !isNum(b)) return;
        var d = a - b;
        diffs.push(d); total += d;
        byChampion[champions[i]] = (byChampion[champions[i]] || 0) + d;
      });
      var sd = sdByArea[area];
      var effect = diffs.length && isNum(sd) && sd ? mean(diffs) / sd : null;
      var driver = null, share = null;
      if (diffs.length && total !== 0) {
        var sign = total > 0 ? 1 : -1;
        Object.keys(byChampion).forEach(function (c) {
          if (driver === null || byChampion[c] * sign > byChampion[driver] * sign) driver = c;
        });
        share = byChampion[driver] * sign / Math.abs(total);
      }
      out[area] = { effect_sd: effect, driver: driver, driver_share: share };
    });
    return out;
  }

  // "Top X%" for a percentile, rounded once so the label, band and verdict always agree
  // (89.8 is "top 10%" everywhere, never "top 10%" on the chart and "top 25%" in the text).
  function topPct(percentile) { return Math.max(1, Math.round(100 - percentile)); }
  function shapeBand(percentile) {
    var t = topPct(percentile);
    return t <= 10 ? 'top 10%' : t <= 25 ? 'top 25%' : t < 75 ? 'middle' : t < 90 ? 'bottom 25%' : 'bottom 10%';
  }
  // Callouts only call something champion-driven when the champions change this verdict.
  function shapeVerdict(percentile) {
    var t = topPct(percentile);
    return t <= 25 ? 'strength' : t >= 75 ? 'weakness' : 'middle';
  }
  // Risk-for-damage quadrant (percentiles vs your comparison tier). The median player sits
  // on the corner; ties count as up/right.
  function quadrantLabel(damage, survival) {
    var hiD = damage >= 50, hiS = survival >= 50;
    return hiD && hiS ? 'Efficient carry' : hiS ? 'Safe, low impact' : hiD ? 'Glass cannon' : 'Struggling';
  }

  var ProfileMath = {
    championEffect: championEffect,
    topPct: topPct,
    shapeBand: shapeBand,
    shapeVerdict: shapeVerdict,
    quadrantLabel: quadrantLabel,
    mean: mean,
    percentileFromBreakpoints: percentileFromBreakpoints,
    areaProfile: areaProfile,
    ownSampleStatus: ownSampleStatus,
    effectiveChampions: effectiveChampions,
    topChampion: topChampion,
    classMix: classMix,
    biggestGaps: biggestGaps,
    gapSize: gapSize
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = ProfileMath;
  else root.ProfileMath = ProfileMath;
})(this);
