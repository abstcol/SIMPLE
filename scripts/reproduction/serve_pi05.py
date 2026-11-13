"""Serve the official 36-D SIMPLE pi0.5 checkpoint using its original transforms."""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import safetensors.torch
import torch
from openpi import transforms
from openpi.models.pi0_config import Pi0Config
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.policies.policy import Policy
from openpi.policies.psi_policy import HfmInputs, HfmOutputs
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.shared.normalize import NormStats

from pi05_common import sha256


def load_stats(path):
    # Official rewrite_norm_stats.py renames these fields without recomputing them.
    raw = json.loads(path.read_text())
    stats = {dst: NormStats(**{key: np.asarray(raw[src][key])
                              for key in ('mean', 'std', 'q01', 'q99')})
             for src, dst in [('states', 'state'), ('action', 'actions')]}
    for key, width in [('state', 32), ('actions', 36)]:
        values = stats[key]
        if any(value.shape != (width,) or not np.isfinite(value).all()
               for value in (values.mean, values.std, values.q01, values.q99)):
            raise ValueError(f'Invalid {key} normalization statistics')
        if np.any(values.q99 < values.q01):
            raise ValueError(f'Reversed {key} normalization quantiles')
    return stats


def policy_transforms(config, stats):
    return ([HfmInputs(config.model_type),
             transforms.Normalize(stats, use_quantiles=True),
             transforms.ResizeImages(*config.image_resolution),
             transforms.TokenizePrompt(PaligemmaTokenizer(config.max_token_len),
                                       discrete_state_input=config.discrete_state_input),
             transforms.PadStatesAndActions(config.action_dim)],
            [transforms.Unnormalize(stats, use_quantiles=True), HfmOutputs()])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--stats', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8015)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--task', default='G1WholebodyBendPickMP-v0')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, force=True)
    config = Pi0Config(pi05=True, action_dim=36, action_horizon=30, max_token_len=250)
    model = PI0Pytorch(config)
    safetensors.torch.load_model(model, str(args.checkpoint), strict=True)
    logging.info('Strict checkpoint restore passed; parameters=%d',
                 sum(parameter.numel() for parameter in model.parameters()))
    model.paligemma_with_expert.to_bfloat16_for_selected_params('bfloat16')
    inputs, outputs = policy_transforms(config, load_stats(args.stats))

    class SeededPolicy(Policy):
        def infer(self, obs, **kwargs):
            if obs.get('reset', False):
                torch.manual_seed(int(obs.get('model_seed', args.seed)))
            return super().infer(obs, **kwargs)

    policy = SeededPolicy(
        model,
        transforms=inputs, output_transforms=outputs,
        is_pytorch=True, pytorch_device='cuda',
        metadata={'task': args.task, 'action_dim': 36,
                  'action_horizon': 30, 'seed': args.seed,
                  'stats_sha256': sha256(args.stats.read_bytes()),
                  'normalization': 'published_training_quantiles',
                  'image_resolution': list(config.image_resolution),
                  'stats_source': str(args.stats), 'checkpoint': str(args.checkpoint)},
    )
    # The official server appends the previous torso command to the 28-D input.
    WebsocketPolicyServer(policy, host='127.0.0.1', port=args.port,
                          metadata=policy.metadata).serve_forever()


if __name__ == '__main__':
    main()
