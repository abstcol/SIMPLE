"""Run prepared pi0.5 task manifests sequentially with isolated model services."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

from pi05_common import checked_read, validate_manifest, write_json

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / 'scripts/reproduction'


def stop(process):
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def interrupted(signum, _frame):
    raise SystemExit(f'Received signal {signum}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifests', 'data-root', 'weights-root', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--port', type=int, help='Defaults to 8015 + GPU index')
    parser.add_argument('--gpu', type=int, help='Physical nvidia-smi GPU index for this independent batch')
    parser.add_argument('--tasks', nargs='+', help='Environment name suffixes; omitted means all six')
    parser.add_argument('--sim-mode', choices=('mujoco', 'mujoco_isaac'), default='mujoco')
    parser.add_argument('--isaac-session', choices=('trial', 'task'), default='trial')
    args = parser.parse_args()
    if args.gpu is not None and args.gpu < 0:
        parser.error('--gpu must be nonnegative')
    if args.isaac_session == 'task' and args.sim_mode != 'mujoco_isaac':
        parser.error('--isaac-session task requires --sim-mode mujoco_isaac')
    if args.port is None:
        args.port = 8015 + (args.gpu or 0)
    args.output = args.output.resolve()
    summary = json.loads((args.manifests / 'summary.json').read_text())
    available = {item['env_id'].removeprefix('simple/') for item in summary['tasks']}
    if args.tasks and set(args.tasks) - available:
        parser.error(f'Unknown tasks: {set(args.tasks) - available}')
    tasks = []
    sim_python = ROOT / ('.venv-isaac/bin/python' if args.sim_mode == 'mujoco_isaac' else '.venv-sonic/bin/python')
    for item in summary['tasks']:
        name = item['env_id'].removeprefix('simple/')
        if args.tasks and name not in args.tasks:
            continue
        manifest = (args.manifests / item['manifest']).resolve()
        records = validate_manifest(json.loads(checked_read(manifest, item['manifest_sha256'])))
        checkpoint = (args.weights_root / records[0]['weight_task'] / 'model.safetensors').resolve()
        stats = (args.data_root / 'train' / name / 'meta/stats.json').resolve()
        for path in (checkpoint, stats, ROOT / '.venv-pi05/bin/python', sim_python):
            if not path.is_file():
                raise FileNotFoundError(path)
        checked_read(stats, records[0]['metadata_source']['train_stats_sha256'])
        folder = args.output / name
        tasks.append({'task': name, 'status': 'unrun', 'server_command': [str(ROOT / '.venv-pi05/bin/python'), '-u',
                      str(SCRIPTS / 'serve_pi05.py'), '--checkpoint', str(checkpoint), '--stats', str(stats),
                      '--task', records[0]['weight_task'], '--port', str(args.port)],
                      'eval_command': [str(sim_python), '-u', str(SCRIPTS / 'run_pi05_suite.py'),
                      '--manifest', str(manifest), '--output', str(folder / 'evaluations'), '--port', str(args.port),
                      '--sim-mode', args.sim_mode, '--isaac-session', args.isaac_session]})
    if not tasks:
        raise ValueError('No task manifests selected')
    # Refuse an occupied port without sending any request to an existing service.
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', args.port))
    common = dict(MUJOCO_GL='egl', JAX_PLATFORMS='cpu', TORCH_COMPILE_DISABLE='1',
                  HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                  OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false')
    if args.gpu is not None:
        gpu_uuid = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu), '--query-gpu=uuid',
                                           '--format=csv,noheader'], text=True).strip()
        if not gpu_uuid.startswith('GPU-') or '\n' in gpu_uuid:
            raise ValueError(f'Expected one physical GPU, got {gpu_uuid!r}')
        common.update(CUDA_VISIBLE_DEVICES=gpu_uuid, SIMPLE_ISAAC_GPU=str(args.gpu),
                      MUJOCO_EGL_DEVICE_ID=str(args.gpu), SIMPLE_ISAAC_CPU_THREADS='8')
    for key, path in {'HF_HOME': 'huggingface', 'HF_HUB_CACHE': 'huggingface/hub',
                      'HUGGINGFACE_HUB_CACHE': 'huggingface/hub', 'HF_DATASETS_CACHE': 'huggingface/datasets',
                      'XDG_CACHE_HOME': 'runtime', 'TORCH_HOME': 'torch', 'OPENPI_DATA_HOME': 'openpi',
                      'TORCH_EXTENSIONS_DIR': 'torch-extensions', 'TORCHINDUCTOR_CACHE_DIR': 'torchinductor-pi05',
                      'TRITON_CACHE_DIR': 'triton-pi05'}.items():
        common[key] = str(ROOT / '.cache' / path)
    source = '.tools/psi0-source/Psi0/src'
    server_path = ':'.join(str(ROOT / path) for path in (source, source + '/openpi/openpi-client/src'))
    sim_path = ':'.join(str(ROOT / path) for path in ('src', 'third_party', 'third_party/GR00T-WholeBodyControl',
        'third_party/unitree_sdk2_python', source + '/openpi/openpi-client/src', '.tools/pi05-suite-deps',
        '.venv-pi05-eval/lib/python3.10/site-packages', '.venv/lib/python3.10/site-packages'))
    if args.sim_mode == 'mujoco_isaac':
        sim_path = ':'.join(str(ROOT / path) for path in ('src', 'third_party',
            'third_party/GR00T-WholeBodyControl', 'third_party/unitree_sdk2_python',
            source + '/openpi/openpi-client/src'))
        for key, path in {'XDG_CACHE_HOME': 'isaac-runtime', 'XDG_DATA_HOME': 'isaac-data',
                          'CUDA_CACHE_PATH': 'cuda-isaac', 'TORCH_EXTENSIONS_DIR': 'torch-extensions-isaac',
                          'SIMPLE_ISAAC_PORTABLE_ROOT': 'isaac-kit'}.items():
            common[key] = str(ROOT / '.cache' / path)
        if args.gpu is not None:
            for key in ('XDG_CACHE_HOME', 'XDG_DATA_HOME', 'SIMPLE_ISAAC_PORTABLE_ROOT'):
                common[key] = str(Path(common[key]) / f'gpu-{args.gpu}')
    base_env = {key: value for key, value in os.environ.items()
                if key not in ('PYTHONHOME', 'PYTHONPATH', 'VIRTUAL_ENV', 'CONDA_PREFIX', 'TRANSFORMERS_CACHE')}
    state = {'git_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
             'status': 'running', 'sim_mode': args.sim_mode, 'isaac_session': args.isaac_session,
             'gpu': args.gpu, 'environment_overrides': common, 'server_pythonpath': server_path,
             'sim_pythonpath': sim_path, 'tasks': tasks}
    args.output.mkdir(parents=True, exist_ok=False)
    signal.signal(signal.SIGTERM, interrupted)
    write_json(args.output / 'launch.json', state)
    try:
        for task in tasks:
            folder = args.output / task['task']
            folder.mkdir()
            server = evaluator = None
            started = time.monotonic()
            try:
                task['status'] = 'starting_server'
                with (folder / 'service.log').open('x') as service_log, (folder / 'eval.log').open('x') as eval_log:
                    server = subprocess.Popen(task['server_command'], cwd=ROOT, stdout=service_log, stderr=subprocess.STDOUT,
                                              env={**base_env, **common, 'PYTHONPATH': server_path})
                    task['server_pid'] = server.pid
                    write_json(args.output / 'launch.json', state)
                    while 'server listening' not in (folder / 'service.log').read_text(errors='replace').lower():
                        if server.poll() is not None:
                            raise RuntimeError(f'Model server exited: {server.returncode}; see service.log')
                        if time.monotonic() - started > 900:
                            raise TimeoutError('Model server was not listening within 15 minutes')
                        time.sleep(1)
                    task['status'] = 'evaluating'
                    evaluator = subprocess.Popen(task['eval_command'], cwd=ROOT, stdout=eval_log, stderr=subprocess.STDOUT,
                                                 env={**base_env, **common, 'PYTHONPATH': sim_path})
                    task['eval_pid'] = evaluator.pid
                    write_json(args.output / 'launch.json', state)
                    while evaluator.poll() is None:
                        if server.poll() is not None:
                            raise RuntimeError(f'Model server exited during evaluation: {server.returncode}')
                        time.sleep(1)
                    if evaluator.returncode:
                        raise RuntimeError(f'Evaluator exited: {evaluator.returncode}; see eval.log')
                    result = json.loads((folder / 'evaluations/summary.json').read_text())
                    if result['status'] != 'complete' or result['completed'] != result['requested']:
                        raise RuntimeError('Evaluator did not complete every requested trial')
                    task.update(status='complete', completed=result['completed'], successes=result['successes'])
            except BaseException as error:
                task.update(status='error', error=repr(error))
                raise
            finally:
                stop(evaluator)
                stop(server)
                task.update(wall_seconds=time.monotonic() - started,
                            server_exit_code=server.returncode if server else None,
                            eval_exit_code=evaluator.returncode if evaluator else None)
                write_json(args.output / 'launch.json', state)
        state['status'] = 'complete'
    except BaseException as error:
        state.update(status='error', error=repr(error))
        raise
    finally:
        write_json(args.output / 'launch.json', state)


if __name__ == '__main__':
    main()
