"""Freeze six pi0.5 manifests: 10 official Level-0 initial conditions x 2 seeds.

Input layout: DATA_ROOT/{eval,train}/ENV_NAME/.../meta. Metadata must match
the frozen official release in pi05-tasks.json. --pilot emits the first initial
condition with both model seeds per task; these trials never count toward the main set.
"""
import argparse
import hashlib
import json
from pathlib import Path

from run_pi05_suite import decode_environment_config, validate_manifest, write_json


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


def prepare_task(contract, catalog, data_root, pilot):
    task_name = contract['env_id'].removeprefix('simple/')
    source = contract['metadata_source']
    eval_path = Path('eval') / task_name / 'dr-level-0/meta/episodes.jsonl'
    stats_path = Path('train') / task_name / 'meta/stats.json'
    rows = [json.loads(line) for line in checked_read(
        data_root / eval_path, source['episodes_sha256']).decode().splitlines() if line.strip()]
    stats = json.loads(checked_read(data_root / stats_path, source['train_stats_sha256']))
    for field, width in [('states', 32), ('action', 36)]:
        if any(len(stats[field][key]) != width for key in ('mean', 'std', 'q01', 'q99')):
            raise ValueError(f'{task_name}: invalid {field} normalization dimensions')
    indices = [row['episode_index'] for row in rows]
    if sorted(indices) != list(range(10)):
        raise ValueError(f'{task_name}: expected exactly official episode indices 0..9, got {indices}')
    rows.sort(key=lambda row: row['episode_index'])
    configs = [decode_environment_config(row['environment_config']) for row in rows]
    hashes = [config_hash(config) for config in configs]
    if len(set(hashes)) != 10:
        raise ValueError(f'{task_name}: Level-0 initial configurations are not 10 distinct conditions')
    for config in configs:
        if config['uid'] != contract['task_uid'] or config['metadata']['dr_level'] != 0:
            raise ValueError(f'{task_name}: mismatched task UID or DR level')
    metadata_source = {'repo': catalog['data_repo'], 'revision': catalog['data_revision'],
                       **source, 'runtime_episodes_path': eval_path.as_posix(),
                       'runtime_train_stats_path': stats_path.as_posix()}
    records = []
    for row, config, digest in zip(rows, configs, hashes):
        if pilot and row['episode_index'] != 0:
            continue
        original = row['environment_config']
        if isinstance(original, str):
            original = json.loads(original.replace('\\"', '"'))
        scene = config['dr_state_dict']['scene']
        for seed in [0, 1]:
            records.append({
                'trial_id': f'{"pilot_" if pilot else ""}episode_{row["episode_index"]:03d}_seed_{seed}',
                'env_id': contract['env_id'], 'controller': contract['controller'],
                'weight_task': contract['weight_task'], 'episode_index': row['episode_index'],
                'model_seed': seed, 'sim_seed': 0, 'dr_level': 0,
                'max_steps': contract['max_steps'], 'success_criteria': contract['success_criteria'],
                'environment_config': row['environment_config'],
                'environment_config_sha256': digest,
                'source_environment_config_sha256': config_hash(original),
                'scene_uid': scene['uid'], 'scene_config_sha256': config_hash(scene),
                'source_episode': {key: value for key, value in row.items() if key != 'environment_config'},
                'metadata_source': metadata_source,
                'model_source': {'repo': catalog['model_repo'], 'revision': catalog['model_revision'],
                                 'checkpoint_subdir': contract['checkpoint_subdir']},
                'protocol': 'pilot' if pilot else 'main', 'counts_toward_main': not pilot,
            })
    validate_manifest(records)
    return records, {'env_id': contract['env_id'], 'weight_task': contract['weight_task'],
                     'trials': len(records), 'unique_initial_conditions': 1 if pilot else 10,
                     'available_initial_conditions': 10,
                     'scene_uids': sorted({config['dr_state_dict']['scene']['uid'] for config in configs}),
                     'controller': contract['controller'], 'max_steps': contract['max_steps'],
                     'success_criteria': contract['success_criteria'], 'metadata_source': metadata_source}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    catalog_path = Path(__file__).with_name('pi05-tasks.json')
    catalog = json.loads(catalog_path.read_text())
    if len(catalog['contracts']) != 6:
        raise ValueError('Expected the six audited pi0.5 task contracts')
    prepared = [prepare_task(contract, catalog, args.data_root, args.pilot)
                for contract in catalog['contracts']]
    # Validate every input before creating any output. Never overwrite a run.
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'tasks.json', catalog)
    summary = {'protocol': 'pilot' if args.pilot else 'main', 'counts_toward_main': not args.pilot,
               'total_trials': sum(len(records) for records, _ in prepared),
               'model_seeds': [0, 1], 'sim_seed': 0, 'dr_level': 0,
               'catalog_sha256': sha256(catalog_path.read_bytes()),
               'note': 'Initial conditions vary within one scene UID per task; these are not ten different rooms.',
               'tasks': []}
    for records, task_summary in prepared:
        filename = task_summary['env_id'].removeprefix('simple/') + '.json'
        path = args.output / filename
        write_json(path, records)
        summary['tasks'].append({**task_summary, 'manifest': filename, 'manifest_sha256': sha256(path.read_bytes())})
    write_json(args.output / 'summary.json', summary)
    print(f'Prepared {summary["total_trials"]} {summary["protocol"]} trials in {args.output}')


if __name__ == '__main__':
    main()
