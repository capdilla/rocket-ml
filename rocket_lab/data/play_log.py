"""Durable, per-episode flight logs from the interactive Pygame controller."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

from ..physics import GRAVITY, ENGINE_ACCEL


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PlayLogger:
    """Write one JSONL event per step so interrupted flights remain inspectable."""

    def __init__(self, root: str | Path):
        session_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.directory = Path(root) / f"{session_name}-{uuid4().hex[:8]}"
        self.directory.mkdir(parents=True)
        self.episode_number = 0
        self.path: Path | None = None
        self.file = None
        self.active = False
        self.needs_review = False

    def _write(self, event: dict):
        line = json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
        if self.file is not None:
            self.file.write(line)
            self.file.flush()
        elif self.path is not None:
            with self.path.open("a", encoding="utf-8") as file:
                file.write(line)

    def start(self, env, mode: str, seed: int, model_path: str | None):
        if self.active:
            raise RuntimeError("The previous flight is still open")
        self.episode_number += 1
        self.path = self.directory / f"episode_{self.episode_number:04d}.jsonl"
        self.file = self.path.open("x", encoding="utf-8")
        self.active = True
        self.needs_review = False
        self.episode_step = 0
        self.total_reward = 0.0
        self._write({
            "type": "start", "schema_version": 1, "timestamp": utc_now(),
            "mode": mode, "seed": seed, "model_path": model_path,
            "pad_x": env.pad_x, "pad_y": env.pad_y,
            "gravity": env.gravity, "engine_accel": env.engine_accel,
            "max_horizontal_distance": env.max_horizontal_distance,
            "max_height": env.max_height,
            "initial_observation": list(env.observation()),
        })

    def record(self, state, action: int, reward: float, next_state,
               terminated: bool, truncated: bool, info: dict,
               controller: str, ai_action: int | None, model_mode: str | None):
        if not self.active:
            raise RuntimeError("There is no open flight")
        self._write({
            "type": "step", "step": self.episode_step + 1,
            "state": list(state), "action": action, "reward": reward,
            "next_state": list(next_state), "terminated": terminated,
            "truncated": truncated, "outcome": info["outcome"],
            "controller": controller, "ai_action": ai_action,
            "model_mode": model_mode,
        })
        self.episode_step += 1
        self.total_reward += reward

    def finish(self, env, reason: str, model_mode: str | None = None):
        if not self.active:
            return
        self._write({
            "type": "end", "timestamp": utc_now(), "reason": reason,
            "outcome": env.last_outcome or "INTERRUPTED",
            "duration_steps": self.episode_step,
            "total_reward": self.total_reward,
            "final_observation": list(env.observation()),
            "model_mode": model_mode,
            "needs_review": self.needs_review,
        })
        self.file.close()
        self.file = None
        self.active = False

    def mark_for_review(self):
        if self.path is None or self.needs_review:
            return
        self.needs_review = True
        self._write({"type": "review", "timestamp": utc_now(), "needs_review": True})


def load_play_episode(path: str | Path) -> dict:
    """Reconstruct a flight for analysis or experience_store.replay_episode."""
    with Path(path).open(encoding="utf-8") as file:
        events = [json.loads(line) for line in file if line.strip()]
    if not events or events[0].get("type") != "start":
        raise ValueError(f"{path} does not start with a start event")
    start = events[0]
    steps = [event for event in events[1:] if event.get("type") == "step"]
    endings = [event for event in events[1:] if event.get("type") == "end"]
    end = endings[-1] if endings else None
    return {
        "schema_version": start["schema_version"],
        "mode": start["mode"], "seed": start["seed"],
        "model_path": start["model_path"],
        "pad_x": start["pad_x"], "pad_y": start["pad_y"],
        "gravity": start.get("gravity", GRAVITY),
        "engine_accel": start.get("engine_accel", ENGINE_ACCEL),
        "initial_observation": start["initial_observation"],
        "transitions": steps,
        "outcome": end["outcome"] if end else "INCOMPLETE",
        "end_reason": end["reason"] if end else None,
        "duration_steps": len(steps),
        "total_reward": end["total_reward"] if end else sum(step["reward"] for step in steps),
        "needs_review": any(event.get("needs_review") for event in events[1:]),
    }
