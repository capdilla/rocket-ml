// Port of rocket.py's fixed-step, semi-implicit Euler simulation.
export const DT = 1 / 60;
export const GRAVITY = 1.62;
export const EARTH_GRAVITY = 9.81;
export const ENGINE_ACCEL = 4.5;
export const EARTH_ENGINE_ACCEL = ENGINE_ACCEL * EARTH_GRAVITY / GRAVITY;
export const SIDE_ACCEL = 2.0;
export const ROTATION_ACCEL = 4.0;
export const ROTATION_DAMPING = 0.985;
export const PAD_HALF_WIDTH = 1.5;
export const SAFE_VERTICAL_SPEED = 1.0;
export const SAFE_HORIZONTAL_SPEED = 0.7;
export const SAFE_ANGLE = 12 * Math.PI / 180;
export const MAX_STEPS = 60 * 30;

export function uprightAngle(angle) {
  return ((angle + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
}

function mulberry32(seed) {
  let value = seed >>> 0;
  return () => {
    value += 0x6D2B79F5;
    let t = value;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export class RocketSimulation {
  constructor({ mode = 'y', planet = 'moon', seed = 42, padRange = 0 } = {}) {
    this.mode = mode;
    this.planet = planet;
    this.padRange = padRange;
    this.random = mulberry32(seed);
    this.reset(false);
  }

  get gravity() {
    return this.planet === 'earth' ? EARTH_GRAVITY : GRAVITY;
  }

  get engineAccel() {
    return this.planet === 'earth' ? EARTH_ENGINE_ACCEL : ENGINE_ACCEL;
  }

  between(min, max) {
    return min + (max - min) * this.random();
  }

  reset(randomize = false) {
    this.padX = this.padRange ? this.between(-this.padRange, this.padRange) : 0;
    if (this.mode === 'y' && !randomize) {
      this.state = { x: 0, y: 9, vx: 0, vy: -0.4, angle: 0, angularVelocity: 0 };
    } else if (this.mode === 'xy' || this.mode === 'xy-angle') {
      this.state = {
        x: (this.random() < 0.5 ? -1 : 1) * this.between(0.25, 3.5),
        y: 9,
        vx: this.between(-0.5, 0.5),
        vy: -0.4,
        angle: this.mode === 'xy-angle' ? this.between(-0.45, 0.45) : 0,
        angularVelocity: this.mode === 'xy-angle' ? this.between(-0.3, 0.3) : 0,
      };
    } else {
      this.state = {
        x: this.between(-3, 3),
        y: this.between(7, 10),
        vx: this.between(-0.5, 0.5),
        vy: this.between(-0.8, 0.1),
        angle: this.between(-0.12, 0.12),
        angularVelocity: 0,
      };
    }
    this.initialState = { ...this.state };
    this.initialPadX = this.padX;
    this.steps = 0;
    this.finished = false;
    this.outcome = null;
    this.lastInput = { engine: false, turn: 0, lateral: 0 };
    this.lastReward = 0;
  }

  restart() {
    this.state = { ...this.initialState };
    this.padX = this.initialPadX;
    this.steps = 0;
    this.finished = false;
    this.outcome = null;
    this.lastInput = { engine: false, turn: 0, lateral: 0 };
    this.lastReward = 0;
  }

  step({ engine = false, turn = 0, lateral = 0 } = {}) {
    if (this.finished) return;
    if (this.mode === 'y') lateral = 0;
    const s = this.state;
    this.lastInput = { engine: Boolean(engine), turn, lateral };

    s.angularVelocity = (s.angularVelocity + turn * ROTATION_ACCEL * DT) * ROTATION_DAMPING;
    s.angle += s.angularVelocity * DT;
    const ax = -Math.sin(s.angle) * this.engineAccel * Number(engine) + lateral * SIDE_ACCEL;
    const ay = Math.cos(s.angle) * this.engineAccel * Number(engine) - this.gravity;
    s.vx += ax * DT;
    s.vy += ay * DT;
    s.x += s.vx * DT;
    s.y += s.vy * DT;
    this.steps += 1;
    this.lastReward = -0.01;

    if (s.y <= 0) {
      s.y = 0;
      const safe = Math.abs(s.x - this.padX) <= PAD_HALF_WIDTH
        && Math.abs(s.vx) <= SAFE_HORIZONTAL_SPEED
        && Math.abs(s.vy) <= SAFE_VERTICAL_SPEED
        && Math.abs(uprightAngle(s.angle)) <= SAFE_ANGLE
        && Math.abs(s.angularVelocity) <= 0.35;
      this.outcome = safe ? 'LANDED' : 'CRASHED';
      this.lastReward = safe ? 100 : -100;
    } else if (['xy', 'xy-angle'].includes(this.mode)
      && (Math.abs(s.x - this.padX) > 8 || s.y > 14)) {
      this.outcome = 'CRASHED';
      this.lastReward = -100;
    } else if (this.steps >= MAX_STEPS) {
      this.outcome = 'TIMEOUT';
    }
    this.finished = Boolean(this.outcome);
  }
}
