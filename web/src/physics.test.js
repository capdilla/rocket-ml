import { test, expect } from 'bun:test';
import { RocketSimulation } from './physics.js';

function close(actual, expected) {
  expect(Math.abs(actual - expected)).toBeLessThan(1e-10);
}

test('fixed-step thrust and coast match rocket.py', () => {
  const sim = new RocketSimulation({ mode: 'y' });
  for (let i = 0; i < 60; i++) sim.step({ engine: true });
  close(sim.state.y, 10.064);
  close(sim.state.vy, 2.4800000000000013);
  for (let i = 0; i < 30; i++) sim.step();
  close(sim.state.y, 11.09475);
  close(sim.state.vy, 1.6700000000000002);
});

test('rotation and angled thrust match rocket.py', () => {
  const sim = new RocketSimulation({ mode: 'y' });
  for (let i = 0; i < 60; i++) sim.step({ engine: true, turn: 1 });
  for (let i = 0; i < 15; i++) sim.step({ engine: true, turn: -1 });
  for (let i = 0; i < 20; i++) sim.step();
  const expected = [-2.368558396042541, 10.287929526494915, -3.198288854129059, 0.21976954330684845, 2.3186085256770834, 0.881474443543783];
  [sim.state.x, sim.state.y, sim.state.vx, sim.state.vy, sim.state.angle, sim.state.angularVelocity]
    .forEach((value, index) => close(value, expected[index]));
});

test('landing requires the pad and safe velocity', () => {
  const landed = new RocketSimulation();
  Object.assign(landed.state, { y: 0.005, vy: -0.5 });
  landed.step();
  expect(landed.outcome).toBe('LANDED');
  expect(landed.lastReward).toBe(100);

  const crashed = new RocketSimulation();
  Object.assign(crashed.state, { x: 2, y: 0.005, vy: -0.5 });
  crashed.step();
  expect(crashed.outcome).toBe('CRASHED');
  expect(crashed.lastReward).toBe(-100);
});

test('restart repeats the selected scenario', () => {
  const sim = new RocketSimulation({ mode: 'xy-angle', padRange: 2, seed: 12 });
  const initialState = { ...sim.state };
  const initialPad = sim.padX;
  sim.step({ engine: true, turn: 1, lateral: -1 });
  sim.restart();
  expect(sim.state).toEqual(initialState);
  expect(sim.padX).toBe(initialPad);
  expect(sim.steps).toBe(0);
});
