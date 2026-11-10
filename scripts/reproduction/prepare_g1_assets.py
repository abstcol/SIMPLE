"""Install verified upstream G1/scene/grasp assets and materialize AMO LFS files."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[2]
FILES = {
    'robots_g1.zip': '4fad7f31969a1ce0bf53f73b8631d4d77b1d826cb983192fe7b9bc76d2ee337c',
    'scenes_hssd_107734119_175999932.zip': 'fd04f1c3612b249ddb43e7e420ff46a0fe625cdc6aa22cd3b6451623d44e3448',
    'assets_graspnet_dex_grasp_dex3_right_0.zip': 'b93bfbe7abf38fbb0c49ccd003ad97bce3477a6c9f9a5656d03e8438419627c4',
    'amo_jit.pt': '6d867ed2dd2261d0f02a5e81d2b7f92802be30f3d36570fb7c4b18707649ef3f',
    'adapter_jit.pt': '159c5f691e55f68c68e2d98e287f4f87baa54e79759c04e06af3bdbc8f8edc98',
    'adapter_norm_stats.pt': '5adc634a7c5928b7d0a4ede15dbed4371b665b6972d706f32416350028e1956a',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    args = parser.parse_args()
    for name, expected in FILES.items():
        source = args.inputs/name
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected, name
    for name in FILES:
        source = args.inputs/name
        if name.endswith('.zip'):
            target = ROOT/'data'
            version = 2 if name == 'robots_g1.zip' else 1
            expected = json.loads((target/'_versions.json').read_text()).get(name, 1)
            assert expected <= version, f'{name}: pinned archive is older than repository metadata'
            with zipfile.ZipFile(source) as archive:
                for member in archive.namelist():
                    assert (target/member).resolve().is_relative_to(target.resolve()), member
                archive.extractall(target)
            (target/f'.{name}.version').write_text(f'{version}\n')
        else:
            shutil.copy2(source, ROOT/'src/simple/robots/policy'/name)
        print('Verified and installed:', name, flush=True)


if __name__ == '__main__':
    main()
