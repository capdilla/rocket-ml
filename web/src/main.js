import * as THREE from '../vendor/three.module.js';
import { RocketSimulation, DT, PAD_HALF_WIDTH, uprightAngle } from './physics.js';
import { PpoPilot } from './agent.js';

const app = document.querySelector('#app');
app.innerHTML = `
  <div class="shell">
    <header class="topbar">
      <div class="brand"><span class="brand-mark">✦</span><div><strong>ROCKET<span>LAB</span></strong><small>FLIGHT SIMULATOR / 01</small></div></div>
      <div class="topbar-right"><span class="live-dot"></span><span>REAL-TIME SIMULATION</span><span class="top-divider"></span><span id="top-clock">T+ 00:00.0</span><button type="button" id="settings-toggle" class="hud-settings-toggle" aria-controls="mission-panel" aria-expanded="false">SETTINGS</button></div>
    </header>

    <button type="button" id="settings-backdrop" class="settings-backdrop" aria-label="Close settings" tabindex="-1" hidden></button>

    <main class="layout">
      <section class="flight-panel" aria-label="Flight simulator">
        <div class="stage-header"><div><span class="eyebrow">MISSION / LANDING</span><h1>Gravity does not forgive.<br><em>You are in control.</em></h1></div><div class="stage-index">01 <span>/ 03</span></div></div>
        <div class="viewport" id="viewport">
          <div class="scene" id="scene" aria-label="Rocket and landing pad"></div>
          <div class="viewport-top"><span class="scene-label"><span class="crosshair">⊕</span> FLIGHT VIEW</span><span class="scene-label" id="scene-mode">MOON · XY-ANGLE CHALLENGE</span></div>
          <div class="reticle reticle-a"></div><div class="reticle reticle-b"></div>
          <div class="altitude-badge"><span>ALTITUDE</span><strong id="altitude">9.00</strong><small>METERS</small></div>
          <div class="status-overlay" id="status-overlay" hidden><span id="status-kicker">MISSION ENDED</span><strong id="status-title">LANDING</strong><p id="status-copy"></p><button type="button" id="overlay-restart" class="overlay-button">REPEAT FLIGHT</button></div>
          <div class="viewport-bottom"><span><i class="legend-line"></i> LANDING PAD</span><span>60 STEPS / SEC</span></div>
        </div>
        <div class="instrument-strip">
          <div class="instrument"><span><span class="desktop-label">VERTICAL SPEED</span><span class="hud-label">V. VERT.</span></span><strong id="vertical-speed">−0.40 <small>m/s</small></strong><div class="instrument-bar"><i id="vertical-bar"></i></div></div>
          <div class="instrument"><span><span class="desktop-label">HORIZONTAL SPEED</span><span class="hud-label">V. HORIZ.</span></span><strong id="horizontal-speed">+0.00 <small>m/s</small></strong><div class="instrument-bar"><i id="horizontal-bar"></i></div></div>
          <div class="instrument"><span><span class="desktop-label">TILT</span><span class="hud-label">ANGLE</span></span><strong id="angle">+0.0 <small>°</small></strong><div class="instrument-bar"><i id="angle-bar"></i></div></div>
          <div class="instrument position"><span><span class="desktop-label">X POSITION</span><span class="hud-label">POS. X</span></span><strong id="position-x">+0.00 <small>m</small></strong><div class="position-detail">TARGET <span id="target-x">+0.00 m</span></div></div>
        </div>
        <div class="mobile-play-panel">
          <div class="touch-controls" role="group" aria-label="Touch flight controls">
            <button type="button" class="touch-button lateral" data-control="lateralLeft" aria-label="Left lateral thruster"><span aria-hidden="true">⇤</span><small>LATERAL</small></button>
            <button type="button" class="touch-button" data-control="left" aria-label="Rotate left"><span aria-hidden="true">↶</span><small>ROTATE</small></button>
            <button type="button" class="touch-button thrust" data-control="engine" aria-label="Activate engine"><span aria-hidden="true">↑</span><small>ENGINE</small></button>
            <button type="button" class="touch-button" data-control="right" aria-label="Rotate right"><span aria-hidden="true">↷</span><small>ROTATE</small></button>
            <button type="button" class="touch-button lateral" data-control="lateralRight" aria-label="Right lateral thruster"><span aria-hidden="true">⇥</span><small>LATERAL</small></button>
          </div>
          <div class="mobile-flight-actions" role="group" aria-label="Flight actions">
            <button type="button" id="mobile-restart">REPEAT</button>
            <button type="button" id="mobile-new-flight">NEW FLIGHT</button>
            <button type="button" id="mobile-pause">PAUSE</button>
            <button type="button" id="mobile-ai">AI</button>
          </div>
        </div>
      </section>

      <aside class="mission-panel" id="mission-panel" aria-labelledby="settings-title">
        <div class="panel-heading"><div><span class="eyebrow">COMMAND CENTER</span><h2 id="settings-title">Configure your flight</h2></div><button type="button" id="settings-close" class="settings-close" aria-label="Close settings">✕</button><span class="panel-icon">✳</span></div>
        <div class="setting"><div class="setting-head"><label>01 / GRAVITY</label><span>ENVIRONMENT</span></div><div class="segmented" role="group" aria-label="Gravity"><button class="active" data-planet="moon" aria-pressed="true"><span>◕</span> Moon</button><button data-planet="earth" aria-pressed="false"><span>◉</span> Earth</button></div><p id="planet-description">Gravity: 1.62 m/s² · Thrust: 4.50 m/s²</p></div>
        <div class="setting pad-setting"><div class="setting-head"><label for="pad-range">02 / RANDOM PAD</label><strong id="pad-range-value">0 m</strong></div><input type="range" id="pad-range" min="0" max="5" step="0.5" value="0" /><p>Maximum landing-pad offset for a new flight.</p></div>
        <div class="action-group"><button id="restart" class="primary-action">REPEAT FLIGHT</button><button id="new-flight" class="secondary-action">↻ &nbsp; NEW SCENARIO</button><button id="pause" class="pause-action">Ⅱ &nbsp; PAUSE SIMULATION</button><button id="ai-toggle" class="secondary-action">✧ &nbsp; AI PILOT</button></div>
        <div class="control-card"><div class="control-head"><span>FLIGHT CONTROLS</span><span>KEYBOARD</span></div><div class="control-row"><span>Main engine</span><span><kbd>↑</kbd><kbd>W</kbd></span></div><div class="control-row"><span>Rotate</span><span><kbd>←</kbd><kbd>Q</kbd><kbd>→</kbd><kbd>D</kbd></span></div><div class="control-row"><span>Lateral thrusters</span><span><kbd>J</kbd><kbd>L</kbd></span></div><div class="control-row"><span>AI pilot</span><span><kbd>A</kbd></span></div><div class="control-row"><span>Restart / new flight</span><span><kbd>R</kbd><kbd>⇧ R</kbd></span></div><div class="control-row"><span>Pause</span><span><kbd>P</kbd></span></div></div>
        <button id="test-lateral-audio" class="sound-test" type="button">TEST LATERAL THRUSTER SOUND</button>
        <p id="audio-warning" class="audio-warning" role="status" hidden>The browser could not play the sound. Check the tab volume and press “Test lateral sound” again.</p>
        <p class="footnote">Goal: land inside the green line, nearly upright, at less than 1 m/s.</p>
      </aside>
    </main>
    <footer><span>ROCKET LAB <b>©</b> MANUAL FLIGHT SIMULATOR</span><span>PHYSICS FROM <code>play.py</code> · THREE.JS GRAPHICS · <a href="https://freesound.org/people/daveincamas/sounds/25709/" target="_blank" rel="noopener noreferrer">AIR: DAVEINCAMAS / FREESOUND (CC BY 4.0)</a></span></footer>
  </div>`;

const $ = (selector) => document.querySelector(selector);
const sim = new RocketSimulation({ mode: 'xy-angle' });
const pilot = new PpoPilot();
const held = new Set();
const setEngineAudio = createHoldAudio('../assets/mixkit-space-rocket-full-power-turbine-1720.wav', 0.65);
const setLateralAudio = createHoldAudio('../assets/air-valve-lateral.wav', 0.75);
const lateralPreview = new Audio(new URL('../assets/air-valve-lateral.wav', import.meta.url));
lateralPreview.volume = 0.75;
let awaitingStart = true;
let paused = true;
let accumulator = 0;
let previousTime = 0;
let lastTrailStep = -1;
let aiEnabled = false;
let aiMessage = '';
const trailPoints = [];

function createHoldAudio(path, volume) {
  const audio = new Audio(new URL(path, import.meta.url));
  audio.loop = true;
  audio.preload = 'auto';
  audio.volume = volume;
  let active = false;
  return (shouldPlay) => {
    if (shouldPlay === active) return;
    active = shouldPlay;
    if (shouldPlay) {
      audio.currentTime = 0;
      audio.play().catch(() => { $('#audio-warning').hidden = false; });
    } else {
      audio.pause();
      audio.currentTime = 0;
    }
  };
}

function syncAudio() {
  const running = !paused && !sim.finished && !document.hidden;
  const lateral = Number(held.has('lateralRight')) - Number(held.has('lateralLeft'));
  setEngineAudio(running && held.has('engine'));
  setLateralAudio(running && lateral !== 0);
}

const scene = new THREE.Scene();
const camera = new THREE.OrthographicCamera(-8, 8, 8, -8, 0.1, 100);
camera.position.set(0, 5, 30);
camera.lookAt(0, 5, 0);
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
$('#scene').append(renderer.domElement);

function shape(points, color, z = 0) {
  const outline = new THREE.Shape();
  points.forEach(([x, y], index) => index ? outline.lineTo(x, y) : outline.moveTo(x, y));
  outline.closePath();
  const mesh = new THREE.Mesh(new THREE.ShapeGeometry(outline), new THREE.MeshBasicMaterial({ color, side: THREE.DoubleSide }));
  mesh.position.z = z;
  return mesh;
}

function line(points, color, width = 1, opacity = 1) {
  const geometry = new THREE.BufferGeometry().setFromPoints(points.map(([x, y, z = 0]) => new THREE.Vector3(x, y, z)));
  const material = new THREE.LineBasicMaterial({ color, linewidth: width, transparent: opacity < 1, opacity });
  return new THREE.Line(geometry, material);
}

// Fixed scenery lives in world metres, with the same x/y frame as play.py.
scene.add(shape([[-30, -3], [30, -3], [30, -0.035], [-30, -0.035]], 0x0d1d2b, -2));
scene.add(line([[-30, 0, -1], [30, 0, -1]], 0x335470));
for (let x = -28; x <= 28; x += 2) {
  scene.add(line([[x, -0.12, -0.9], [x, -0.38, -0.9]], 0x2d455c));
}
for (let y = 2; y <= 14; y += 2) {
  const guide = line([[-30, y, -5], [30, y, -5]], 0x4b6d86, 1, 0.09);
  scene.add(guide);
}
const stars = new THREE.Group();
let starSeed = 1347;
function randomStar() { starSeed = (starSeed * 1664525 + 1013904223) >>> 0; return starSeed / 4294967296; }
for (let i = 0; i < 90; i++) {
  const x = (randomStar() - 0.5) * 55;
  const y = randomStar() * 18 + 1;
  const star = new THREE.Mesh(new THREE.CircleGeometry(randomStar() > 0.85 ? 0.026 : 0.013, 6), new THREE.MeshBasicMaterial({ color: 0xa7d3ef, transparent: true, opacity: randomStar() * 0.5 + 0.2 }));
  star.position.set(x, y, -7);
  stars.add(star);
}
scene.add(stars);

const pad = new THREE.Group();
pad.add(shape([[-PAD_HALF_WIDTH, -0.04], [PAD_HALF_WIDTH, -0.04], [PAD_HALF_WIDTH, 0.08], [-PAD_HALF_WIDTH, 0.08]], 0x59e5af, 1));
pad.add(shape([[-PAD_HALF_WIDTH - 0.13, -0.13], [PAD_HALF_WIDTH + 0.13, -0.13], [PAD_HALF_WIDTH + 0.13, -0.04], [-PAD_HALF_WIDTH - 0.13, -0.04]], 0x256955, 0.5));
pad.add(line([[0, 0.08, -0.2], [0, 14, -0.2]], 0x6ae7bd, 1, 0.22));
scene.add(pad);

const rocket = new THREE.Group();
rocket.add(shape([[-0.24, 0.08], [0.24, 0.08], [0.22, 0.86], [0, 1.30], [-0.22, 0.86]], 0xeaf3fa, 2));
rocket.add(shape([[-0.22, 0.85], [0, 1.30], [0.22, 0.85]], 0xf18c5d, 2.2));
rocket.add(shape([[-0.28, 0.14], [0.28, 0.14], [0.22, 0.38], [-0.22, 0.38]], 0x92b9cc, 2.1));
rocket.add(shape([[-0.20, 0.28], [-0.57, 0.03], [-0.46, 0.03], [-0.18, 0.17]], 0x91a9bf, 1.9));
rocket.add(shape([[0.20, 0.28], [0.57, 0.03], [0.46, 0.03], [0.18, 0.17]], 0x91a9bf, 1.9));
rocket.add(shape([[-0.68, 0.015], [-0.42, 0.015], [-0.42, 0.07], [-0.68, 0.07]], 0x91a9bf, 2));
rocket.add(shape([[0.42, 0.015], [0.68, 0.015], [0.68, 0.07], [0.42, 0.07]], 0x91a9bf, 2));
const windowMesh = new THREE.Mesh(new THREE.CircleGeometry(0.115, 24), new THREE.MeshBasicMaterial({ color: 0x3ac6f2 }));
windowMesh.position.set(0, 0.72, 2.4);
rocket.add(windowMesh);
const mainFlame = shape([[-0.12, 0.07], [0.12, 0.07], [0, -0.61]], 0xffb05a, 1.8);
rocket.add(mainFlame);
const leftFlame = shape([[-0.25, 0.53], [-0.25, 0.70], [-0.52, 0.60]], 0x5cc4f4, 2);
const rightFlame = shape([[0.25, 0.53], [0.25, 0.70], [0.52, 0.60]], 0x5cc4f4, 2);
rocket.add(leftFlame, rightFlame);
scene.add(rocket);
const trail = line([[0, 9, 0]], 0x75d7e9, 1, 0.42);
trail.visible = false;
scene.add(trail);

function resize() {
  const bounds = $('#scene').getBoundingClientRect();
  if (!bounds.width || !bounds.height) return;
  renderer.setSize(bounds.width, bounds.height, false);
  const aspect = bounds.width / bounds.height;
  const height = Math.max(15.4, 11.5 / aspect);
  camera.top = height / 2;
  camera.bottom = -height / 2;
  camera.left = -height * aspect / 2;
  camera.right = height * aspect / 2;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe($('#scene'));
resize();

function signed(value, digits = 2) {
  return `${value >= 0 ? '+' : '−'}${Math.abs(value).toFixed(digits)}`;
}

function updateHud() {
  const s = sim.state;
  const angleDeg = uprightAngle(s.angle) * 180 / Math.PI;
  $('#altitude').textContent = Math.max(0, s.y).toFixed(2);
  $('#vertical-speed').innerHTML = `${signed(s.vy)} <small>m/s</small>`;
  $('#horizontal-speed').innerHTML = `${signed(s.vx)} <small>m/s</small>`;
  $('#angle').innerHTML = `${signed(angleDeg, 1)} <small>°</small>`;
  $('#position-x').innerHTML = `${signed(s.x)} <small>m</small>`;
  $('#target-x').textContent = `${signed(sim.padX)} m`;
  $('#vertical-bar').style.width = `${Math.min(100, Math.abs(s.vy) / 5 * 100)}%`;
  $('#horizontal-bar').style.width = `${Math.min(100, Math.abs(s.vx) / 3 * 100)}%`;
  $('#angle-bar').style.width = `${Math.min(100, Math.abs(angleDeg) / 45 * 100)}%`;
  $('#vertical-bar').classList.toggle('danger', Math.abs(s.vy) > 1);
  $('#horizontal-bar').classList.toggle('danger', Math.abs(s.vx) > 0.7);
  $('#angle-bar').classList.toggle('danger', Math.abs(angleDeg) > 12);
  const seconds = sim.steps * DT;
  $('#top-clock').textContent = `T+ ${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(Math.floor(seconds % 60)).padStart(2, '0')}.${Math.floor(seconds % 1 * 10)}`;
  const pilotLabel = aiEnabled ? ' · AI PILOT' : (aiMessage ? ` · ${aiMessage}` : '');
  $('#scene-mode').textContent = `${sim.planet === 'moon' ? 'MOON' : 'EARTH'} · XY-ANGLE CHALLENGE${pilotLabel}`;
  $('#planet-description').textContent = `Gravity: ${sim.gravity.toFixed(2)} m/s² · Thrust: ${sim.engineAccel.toFixed(2)} m/s²`;
  $('#status-overlay').hidden = !sim.finished && !paused;
  if (sim.finished) {
    const landed = sim.outcome === 'LANDED';
    $('#status-kicker').textContent = landed ? 'MISSION COMPLETE' : 'MISSION ENDED';
    $('#status-title').textContent = landed ? 'LANDING' : sim.outcome === 'TIMEOUT' ? 'TIME EXPIRED' : 'IMPACT';
    $('#status-title').classList.toggle('success', landed);
    $('#status-copy').textContent = landed ? 'Soft landing. The pad is yours.' : 'Adjust thrust, tilt, and speed, then try again.';
    $('#overlay-restart').textContent = 'REPEAT FLIGHT';
  } else if (paused) {
    $('#status-kicker').textContent = awaitingStart ? 'READY FOR FLIGHT' : 'SIMULATION PAUSED';
    $('#status-title').textContent = awaitingStart ? 'XY-ANGLE CHALLENGE' : 'PAUSE';
    $('#status-title').classList.remove('success');
    $('#status-copy').textContent = awaitingStart
      ? 'Hold ENGINE to slow down. Combine it with ROTATE and LATERAL.'
      : 'Press P or the button below to resume.';
    $('#overlay-restart').textContent = awaitingStart ? 'START FLIGHT' : 'RESUME';
  }
  $('#pause').innerHTML = awaitingStart ? '▶ &nbsp; START SIMULATION' : paused ? '▶ &nbsp; RESUME SIMULATION' : 'Ⅱ &nbsp; PAUSE SIMULATION';
  $('#mobile-pause').textContent = awaitingStart ? 'START' : paused ? 'RESUME' : 'PAUSE';
}

function resetTrail() {
  trailPoints.length = 0;
  lastTrailStep = -1;
  trail.visible = false;
}

function restart(newScenario = false) {
  if (newScenario) sim.reset(true);
  else sim.restart();
  awaitingStart = true;
  paused = true;
  accumulator = 0;
  resetTrail();
  syncAudio();
  updateHud();
  if (aiEnabled) togglePause();
}

function setPlanet(planet) {
  sim.planet = planet;
  document.querySelectorAll('[data-planet]').forEach((button) => {
    const active = button.dataset.planet === planet;
    button.classList.toggle('active', active);
    button.setAttribute('aria-pressed', String(active));
  });
  restart(false);
}

function togglePause() {
  if (sim.finished) return;
  paused = !paused;
  awaitingStart = false;
  accumulator = 0;
  syncAudio();
  updateHud();
}

function manualControls() {
  return {
    engine: held.has('engine'),
    turn: Number(held.has('left')) - Number(held.has('right')),
    lateral: Number(held.has('lateralRight')) - Number(held.has('lateralLeft')),
  };
}

async function toggleAI() {
  if (aiEnabled) {
    aiEnabled = false;
    aiMessage = '';
  } else if (!pilot.ready) {
    aiMessage = 'CARGANDO IA…';
    try {
      await pilot.load();
    } catch {
      aiMessage = 'NO SE PUDO CARGAR LA IA';
    }
    if (pilot.ready) {
      aiEnabled = true;
      aiMessage = 'AI PILOT';
      if (awaitingStart) togglePause();
    }
  } else {
    aiEnabled = true;
    aiMessage = 'AI PILOT';
    if (awaitingStart) togglePause();
  }
  syncAudio();
  updateHud();
}

function openSettings() {
  if (!paused && !sim.finished) togglePause();
  releaseControls();
  $('.shell').classList.add('settings-open');
  $('#settings-backdrop').hidden = false;
  $('#settings-toggle').setAttribute('aria-expanded', 'true');
  $('#mission-panel').setAttribute('role', 'dialog');
  $('#mission-panel').setAttribute('aria-modal', 'true');
  $('.flight-panel').inert = true;
  $('.topbar').inert = true;
  $('#settings-close').focus();
}

function closeSettings() {
  if (!$('.shell').classList.contains('settings-open')) return;
  $('.shell').classList.remove('settings-open');
  $('#settings-backdrop').hidden = true;
  $('#settings-toggle').setAttribute('aria-expanded', 'false');
  $('#mission-panel').removeAttribute('role');
  $('#mission-panel').removeAttribute('aria-modal');
  $('.flight-panel').inert = false;
  $('.topbar').inert = false;
  $('#settings-toggle').focus();
}

$('#settings-toggle').addEventListener('click', openSettings);
$('#settings-close').addEventListener('click', closeSettings);
$('#settings-backdrop').addEventListener('click', closeSettings);
document.querySelectorAll('[data-planet]').forEach((button) => button.addEventListener('click', () => setPlanet(button.dataset.planet)));
$('#pad-range').addEventListener('input', (event) => { $('#pad-range-value').textContent = `${event.target.value} m`; });
$('#pad-range').addEventListener('change', (event) => { sim.padRange = Number(event.target.value); sim.reset(false); awaitingStart = true; paused = true; accumulator = 0; resetTrail(); syncAudio(); updateHud(); event.target.blur(); });
$('#restart').addEventListener('click', () => restart());
$('#new-flight').addEventListener('click', () => restart(true));
$('#pause').addEventListener('click', togglePause);
$('#mobile-restart').addEventListener('click', () => restart());
$('#mobile-new-flight').addEventListener('click', () => restart(true));
$('#mobile-pause').addEventListener('click', togglePause);
$('#ai-toggle').addEventListener('click', toggleAI);
$('#mobile-ai').addEventListener('click', toggleAI);
$('#overlay-restart').addEventListener('click', () => paused ? togglePause() : restart());
$('#test-lateral-audio').addEventListener('click', () => {
  lateralPreview.pause();
  lateralPreview.currentTime = 0;
  lateralPreview.play().then(() => { $('#audio-warning').hidden = true; }).catch(() => { $('#audio-warning').hidden = false; });
});

const keyControls = new Map([
  ['ArrowUp', 'engine'], ['KeyW', 'engine'],
  ['ArrowLeft', 'left'], ['KeyQ', 'left'],
  ['ArrowRight', 'right'], ['KeyD', 'right'],
  ['KeyJ', 'lateralLeft'], ['KeyL', 'lateralRight'],
]);
const activeKeys = new Set();
const activePointers = new Map();
const activeButtonControls = new Set();

function isTouchControlActive(control) {
  return activeButtonControls.has(control) || [...activePointers.values()].includes(control);
}

function syncControls() {
  held.clear();
  for (const code of activeKeys) held.add(keyControls.get(code));
  for (const control of activePointers.values()) held.add(control);
  for (const control of activeButtonControls) held.add(control);
  syncAudio();
}
document.addEventListener('keydown', (event) => {
  if ($('.shell').classList.contains('settings-open')) {
    if (event.code === 'Escape') { event.preventDefault(); closeSettings(); }
    if (event.code === 'Tab') {
      const controls = [...$('#mission-panel').querySelectorAll('button, input, a[href]')].filter((element) => element.getClientRects().length);
      const first = controls[0];
      const last = controls.at(-1);
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
    }
    return;
  }
  if (['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement?.tagName)) return;
  if (keyControls.has(event.code)) {
    event.preventDefault();
    if (sim.finished && !event.repeat) restart();
    if (awaitingStart) togglePause();
    activeKeys.add(event.code);
    syncControls();
  } else if (!event.repeat && event.code === 'KeyR') {
    event.preventDefault();
    restart(event.shiftKey);
  } else if (!event.repeat && (event.code === 'KeyP' || event.code === 'Escape')) {
    event.preventDefault();
    togglePause();
  } else if (!event.repeat && event.code === 'KeyA') {
    event.preventDefault();
    toggleAI();
  }
});
document.addEventListener('keyup', (event) => {
  if (keyControls.has(event.code)) { activeKeys.delete(event.code); syncControls(); }
});
function releaseControls() {
  activeKeys.clear();
  activePointers.clear();
  activeButtonControls.clear();
  document.querySelectorAll('.touch-button.pressed').forEach((button) => button.classList.remove('pressed'));
  syncControls();
}
window.addEventListener('blur', releaseControls);
document.addEventListener('visibilitychange', () => { if (document.hidden) releaseControls(); });
document.querySelectorAll('.touch-button').forEach((button) => {
  button.addEventListener('pointerdown', (event) => {
    event.preventDefault();
    if (sim.finished) restart();
    if (awaitingStart) togglePause();
    button.setPointerCapture(event.pointerId);
    activePointers.set(event.pointerId, button.dataset.control);
    button.classList.add('pressed');
    syncControls();
  });
  const release = (event) => {
    activePointers.delete(event.pointerId);
    button.classList.toggle('pressed', isTouchControlActive(button.dataset.control));
    syncControls();
  };
  button.addEventListener('pointerup', release);
  button.addEventListener('pointercancel', release);
  button.addEventListener('lostpointercapture', release);
  button.addEventListener('keydown', (event) => {
    if (event.key !== ' ' && event.key !== 'Enter') return;
    event.preventDefault();
    if (sim.finished) restart();
    if (awaitingStart) togglePause();
    activeButtonControls.add(button.dataset.control);
    button.classList.add('pressed');
    syncControls();
  });
  button.addEventListener('keyup', (event) => {
    if (event.key !== ' ' && event.key !== 'Enter') return;
    event.preventDefault();
    activeButtonControls.delete(button.dataset.control);
    button.classList.toggle('pressed', isTouchControlActive(button.dataset.control));
    syncControls();
  });
  button.addEventListener('blur', () => {
    if (!activeButtonControls.delete(button.dataset.control)) return;
    button.classList.toggle('pressed', isTouchControlActive(button.dataset.control));
    syncControls();
  });
});

async function render(time) {
  if (!previousTime) previousTime = time;
  const elapsed = Math.min((time - previousTime) / 1000, 0.1);
  previousTime = time;
  if (!paused && !sim.finished) {
    let controls;
    if (aiEnabled && pilot.ready) {
      try {
        controls = await pilot.selectAction(sim);
      } catch {
        controls = manualControls();
      }
    } else {
      controls = manualControls();
    }
    accumulator += elapsed;
    while (accumulator >= DT && !sim.finished) {
      sim.step(controls);
      accumulator -= DT;
    }
  }
  syncAudio();
  const s = sim.state;
  rocket.position.set(s.x, s.y, 0);
  rocket.rotation.z = s.angle;
  pad.position.x = sim.padX;
  const flameActive = !paused && !sim.finished;
  mainFlame.visible = flameActive && sim.lastInput.engine;
  leftFlame.visible = flameActive && sim.lastInput.lateral > 0;
  rightFlame.visible = flameActive && sim.lastInput.lateral < 0;
  mainFlame.scale.y = 0.84 + Math.sin(time * 0.035) * 0.13;
  if (sim.steps !== lastTrailStep && sim.steps % 6 === 0) {
    trailPoints.push(new THREE.Vector3(s.x, s.y + 0.52, -0.6));
    if (trailPoints.length > 160) trailPoints.shift();
    if (trailPoints.length >= 2) { trail.geometry.setFromPoints(trailPoints); trail.visible = true; }
    lastTrailStep = sim.steps;
  }
  const targetX = Math.max(-3.5, Math.min(3.5, (s.x + sim.padX) * 0.24));
  camera.position.x += (targetX - camera.position.x) * 0.035;
  camera.lookAt(camera.position.x, 5, 0);
  updateHud();
  renderer.render(scene, camera);
  requestAnimationFrame(render);
}

updateHud();
requestAnimationFrame(render);
