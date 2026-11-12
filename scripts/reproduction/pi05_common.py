"""Shared pi0.5 manifest validation and serialization (standard library only)."""
from copy import deepcopy
import hashlib
import json
import math
import re


def validate_manifest(records):
    if not isinstance(records, list) or not records:
        raise ValueError('Manifest must be a nonempty JSON list of trials')
    seen, contracts = set(), set()
    for trial in records:
        required = {'trial_id', 'env_id', 'controller', 'episode_index',
                    'model_seed', 'sim_seed', 'environment_config',
                    'max_steps', 'success_criteria'}
        if not isinstance(trial, dict) or required - trial.keys():
            raise ValueError(f'Trial is missing required fields: {trial!r}')
        name = trial['trial_id']
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name):
            raise ValueError(f'Unsafe trial_id: {name!r}')
        if name in seen:
            raise ValueError(f'Duplicate trial_id: {name}')
        seen.add(name)
        for key in ('episode_index', 'model_seed', 'sim_seed', 'max_steps'):
            value = trial[key]
            if type(value) is not int or value < (1 if key == 'max_steps' else 0):
                raise ValueError(f'{name}: invalid {key}: {value!r}')
        if trial['sim_seed'] >= 2**32 or trial['model_seed'] >= 2**63:
            raise ValueError(f'{name}: seed is out of range')
        if trial['controller'] not in ('amo', 'decoupled'):
            raise ValueError(f'{name}: unsupported controller')
        if not isinstance(trial['env_id'], str) or not trial['env_id'].startswith('simple/'):
            raise ValueError(f'{name}: expected a registered simple/ environment')
        threshold = trial['success_criteria']
        if type(threshold) not in (int, float) or not math.isfinite(threshold):
            raise ValueError(f'{name}: invalid success_criteria')
        decode_environment_config(trial['environment_config'])
        contracts.add((trial['env_id'], trial['controller'], trial['max_steps'],
                       threshold, trial.get('weight_task')))
    if len(contracts) != 1:
        raise ValueError('Run one task/controller/checkpoint contract per process')
    return records


def decode_environment_config(value):
    # Same historical scene UID migration as get_episode_lerobot, on a copy.
    if isinstance(value, str):
        value = json.loads(value.replace('\\"', '"'))
    value = deepcopy(value)
    if not isinstance(value, dict):
        raise ValueError('environment_config must decode to an object')
    scene = value.get('dr_state_dict', {}).get('scene')
    if isinstance(scene, dict) and isinstance(scene.get('uid'), str):
        scene['uid'] = scene['uid'].replace('102344280', 'scene3')
    return value


def json_default(value):
    if hasattr(value, 'tolist'):
        return value.tolist()
    raise TypeError(f'Cannot serialize {type(value).__name__}')


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=json_default, allow_nan=False) + '\n')
    temporary.replace(path)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def config_hash(config):
    return sha256(json.dumps(config, sort_keys=True, separators=(',', ':'),
                             ensure_ascii=True, allow_nan=False).encode())


def checked_read(path, expected_sha):
    content = path.read_bytes()
    actual_sha = sha256(content)
    if actual_sha != expected_sha:
        raise ValueError(f'Frozen metadata mismatch: {path}: {actual_sha} != {expected_sha}')
    return content
