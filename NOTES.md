# NOTES — fork + pixify log (raw material for a future `fork-and-pixify` skill)

Upstream: TomTomTommi/LiteAnyStereo @ 8c97bd4 (MIT). `main` on this fork is that commit, frozen.
Work branch: `pixi` (default branch of the fork).

## Decisions
- Pins copied from rerun-io/examples-monorepo (python 3.12, cuda-version 13.0.*, conda pytorch-gpu 2.12, rerun-sdk 0.36.2, timm >=1.0,<2, tyro 0.9.x) so the later monorepo port has no env surprises (5090 needs CUDA 13).
- `simplecv` as a git dep on monorepo `main` → `RerunTyroConfig` + `log_rig_static` (exoego:v2 rig layout) in `demo_rerun.py`.
- Change set kept to: pixi.toml, pixi.lock, .gitignore, demo_rerun.py, NOTES.md, README "Run with pixi". Upstream code untouched unless required to run (see "Upstream edits").
- Checkpoints via upstream `download_checkpoint.py` (HF tomtomtommi/LiteAnyStereoV2, M + H only). Data = one ETH3D scene (`playground_1l`) re-hosted at HF `pablovela5620/monoprior-example/stereo/eth3d` in upstream's exact `core/stereo_datasets.py` layout, so upstream `evaluate_stereo.py --dataset eth3d` works unchanged on a single scene.
- Download tasks are guarded with shell `test -f` (pixi `inputs`/`outputs` skip gitignored files).
- ONNX export/profiling deps dropped from the env (out of scope).

## Commands
- fork: `GH_TOKEN=$(gh auth token --user pablovela5620) gh repo fork TomTomTommi/LiteAnyStereo --clone=false`
- sample: ETH3D `two_view_training.7z` + `two_view_training_gt.7z` (14 MB each), `pixi exec -s 7zip -- 7z x`, `hf upload pablovela5620/monoprior-example <dir> stereo --repo-type dataset`

## Upstream edits
(filled in from the Codex report)

## Reproduction
(filled in after eval)
