// Carga el modelo PPO (ONNX) y decide la acción del piloto en el navegador.
// La red recibe 8 valores crudos: x, y, vx, vy, angle, angular_velocity, dx, dy.
import * as ort from '../vendor/ort/ort.min.mjs';
import { uprightAngle } from './physics.js';

const MODEL_URL = new URL('../models/ppo-xy-angle-both.onnx', import.meta.url).href;
const WASM_PATH = new URL('../vendor/ort/', import.meta.url).href;

// Índice de logit -> {engine, turn, lateral} para las acciones xy-angle
// (0, 1, 6, 7, 8, 9, 2, 3, 4, 5) de rocket_lab/physics.py.
const ACTION_CONTROLS = [
  { engine: false, turn: 0, lateral: 0 },   // 0 coast
  { engine: true, turn: 0, lateral: 0 },    // 1 main engine
  { engine: false, turn: 1, lateral: 0 },   // 2 rotate left
  { engine: false, turn: -1, lateral: 0 },  // 3 rotate right
  { engine: true, turn: 1, lateral: 0 },    // 4 engine + left
  { engine: true, turn: -1, lateral: 0 },   // 5 engine + right
  { engine: false, turn: 0, lateral: 1 },   // 6 lateral right
  { engine: false, turn: 0, lateral: -1 },  // 7 lateral left
  { engine: true, turn: 0, lateral: 1 },    // 8 engine + lateral right
  { engine: true, turn: 0, lateral: -1 },   // 9 engine + lateral left
];

export function observationFrom(sim) {
  const s = sim.state;
  return [
    s.x, s.y, s.vx, s.vy,
    uprightAngle(s.angle),
    s.angularVelocity,
    sim.padX - s.x,
    0 - s.y,
  ];
}

export class PpoPilot {
  constructor() {
    this.session = null;
    this.ready = false;
    this.error = null;
    this.loadPromise = null;
  }

  load() {
    if (!this.loadPromise) {
      this.loadPromise = (async () => {
        ort.env.wasm.numThreads = 1;
        ort.env.wasm.wasmPaths = WASM_PATH;
        this.session = await ort.InferenceSession.create(MODEL_URL);
        this.ready = true;
      })();
      this.loadPromise.catch((error) => { this.error = error; });
    }
    return this.loadPromise;
  }

  async selectAction(sim) {
    const observation = new Float32Array(observationFrom(sim));
    const feeds = { observation: new ort.Tensor('float32', observation, [1, 8]) };
    const { logits } = await this.session.run(feeds);
    const data = logits.data;
    let best = 0;
    for (let index = 1; index < data.length; index += 1) {
      if (data[index] > data[best]) best = index;
    }
    return ACTION_CONTROLS[best];
  }
}
