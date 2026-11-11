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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--stats', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8015)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    config = Pi0Config(pi05=True, action_dim=36, action_horizon=30, max_token_len=250)
    model = PI0Pytorch(config)
    safetensors.torch.load_model(model, str(args.checkpoint), strict=True)
    model.paligemma_with_expert.to_bfloat16_for_selected_params('bfloat16')
    # Official rewrite_norm_stats.py maps these two training-statistics fields.
    raw = json.loads(args.stats.read_text())
    stats = {dst: NormStats(**{key: np.asarray(raw[src][key])
                              for key in ('mean', 'std', 'q01', 'q99')})
             for src, dst in [('states', 'state'), ('action', 'actions')]}

    class SeededPolicy(Policy):
        def infer(self, obs, **kwargs):
            if obs.get('reset', False):
                torch.manual_seed(args.seed)
            return super().infer(obs, **kwargs)

    policy = SeededPolicy(
        model,
        transforms=[HfmInputs(config.model_type),
                    transforms.Normalize(stats, use_quantiles=True),
                    transforms.ResizeImages(*config.image_resolution),
                    transforms.TokenizePrompt(PaligemmaTokenizer(config.max_token_len),
                                              discrete_state_input=True),
                    transforms.PadStatesAndActions(config.action_dim)],
        output_transforms=[transforms.Unnormalize(stats, use_quantiles=True), HfmOutputs()],
        is_pytorch=True, pytorch_device='cuda',
        metadata={'task': 'G1WholebodyBendPickMP-v0', 'action_dim': 36,
                  'action_horizon': 30, 'seed': args.seed,
                  'stats_source': str(args.stats), 'checkpoint': str(args.checkpoint)},
    )
    # The official server appends the previous torso command to the 28-D input.
    WebsocketPolicyServer(policy, host='127.0.0.1', port=args.port,
                          metadata=policy.metadata).serve_forever()


if __name__ == '__main__':
    main()
