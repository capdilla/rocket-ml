"""Exporta un modelo PPO a ONNX para inferencia en el navegador.

El escalado de observaciones queda horneado dentro del grafo, así que el
navegador solo pasa los 8 valores crudos (con el ángulo normalizado a
``upright_angle`` antes, igual que ``policy_observation``). La salida es el
vector de logits; el argmax se calcula en JavaScript.
"""
import argparse
from pathlib import Path

import torch

from rocket_lab.ml.ppo_agent import PPOAgent
from rocket_lab.physics import OBSERVATION_SIZE


class ActorOnly(torch.nn.Module):
    def __init__(self, agent: PPOAgent):
        super().__init__()
        self.agent = agent

    def forward(self, observation):
        logits, _ = self.agent(observation)
        return logits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Modelo PPO .pt a exportar")
    parser.add_argument("--output", help="Ruta del .onnx (por defecto web/models/<nombre>.onnx)")
    args = parser.parse_args()

    agent, mode = PPOAgent.from_model(args.model)
    agent.eval()
    model_path = Path(args.model)
    default_name = model_path.parent.name if model_path.parent.name != "runs" else model_path.stem
    output = Path(args.output) if args.output else Path("web/models") / f"{default_name}.onnx"
    output.parent.mkdir(parents=True, exist_ok=True)

    wrapper = ActorOnly(agent)
    torch.onnx.export(
        wrapper,
        torch.zeros(1, OBSERVATION_SIZE, dtype=torch.float32),
        output,
        input_names=["observation"],
        output_names=["logits"],
        opset_version=17,
        external_data=False,
        dynamic_axes={"observation": {0: "batch"}},
    )
    print(f"Exportado {args.model} -> {output} (mode={mode}, planet={agent.planet}, "
          f"actions={agent.allowed_actions})")


if __name__ == "__main__":
    main()
