# Exact GPU ObstructedMaze backend

`gym_adapter.evolve_maze_gpu` runs `MiniGrid-ObstructedMaze-Full-v1` with maze transitions, visibility, symbolic observations, and sensory encoding on CUDA. Brains use the same neural kernels as the existing CUDA evaluator.

## What remains unchanged

- Partial 7×7 symbolic observations, all seven actions, 3,600-step native horizon.
- Movement, object pickup/drop, box contents, matching keys, door state and opacity.
- Target **object identity**, rewards, termination and truncation.
- Episode seeds and per-episode neural resets.
- NumPy float32 sensory encoding, implemented through an exact lookup table.
- NumPy action tie-breaking and its independent per-episode RNG stream.
- Fresh training-only selection, breeding, mutation, population size and stopping goals.
- Fixed report-only validation and fresh stopping-only audits use reference MiniGrid.

Maze initial states are constructed by real MiniGrid on the CPU for each fresh seed and uploaded once per generation. Action decoding and small action/reward/spike transfers remain on the host. The worlds and sensory inputs stay on the GPU throughout the rollout. This is not a completely device-resident evolutionary loop.

## Verification

`tests/test_gym_maze_gpu.py` compares against installed MiniGrid:

1. 2,184 interaction cases covering all orientations/actions, empty space, walls, floors, goals, lava, open/closed/locked doors, correct/wrong keys, target/non-target balls, carried boxes, and empty/populated boxes. Checks exact observations, grid object identities, mutable object encodings, inventory, rewards and completion.
2. Sixteen seeded 400-step random trajectories, including native timeout semantics.
3. Thirty-two brain-driven 256-step trajectories, with bitwise-equal spikes, activity, actions and observations.

Separate benchmark runs compare the generation-143 evolved population against the original evaluator. Full 3,600-step episodes on 64 brains produce identical episode returns, step counts and aggregate motor spikes (919,001 environment steps). The large 2,048-brain 256-step comparison also matches those results exactly.

These are meaningful parity checks, not an exhaustive proof for every possible state. Unsupported environment IDs and dimensions are rejected.

## Measurements

Saved results: `runs/obstructedmaze-gpu/`.

- Population 64, four episodes each, full 3,600-step horizon: original **30.36 s**, GPU maze **5.91 s**; **5.14×** faster, identical results.
- Population 2,048, four episodes each, 256-step benchmark: original **67.63 s**, GPU maze **13.60 s**; **4.97×** faster, identical results.
- Population 2,048, full horizon: GPU maze **99.26 s** for **29,488,601** environment steps. Includes CPU reference resets and brain construction; rollout alone is **92.25 s**.

Other training processes are paused during controlled benchmarks. Historical training generations took about 17 minutes while other searches were running; that comparison is not an isolated matched benchmark. Actual generation time also includes breeding, reference validation and checkpoint saving.

The remaining major cost is brain stepping plus host action decoding: **87.50 s** of the full rollout. Maze transitions and result transfers now take **3.60 s**.

## Running

Use the ordinary evolution flags, replacing the module name with `gym_adapter.evolve_maze_gpu` and keeping `--backend cuda`. This backend is intentionally limited to `MiniGrid-ObstructedMaze-Full-v1`.

```sh
.venv-gym/bin/python -m gym_adapter.evolve_maze_gpu \
  --env MiniGrid-ObstructedMaze-Full-v1 --backend cuda \
  --output runs/my-maze --population 2048 --episodes 4 \
  --validation-episodes 12 --hidden 16 --connections 160 \
  --max-steps 3600 --generations 250 --workers 8 --seed 29 \
  --skip-final-evaluation
```

For an existing run, use its saved settings and `--resume` as usual. The trainer preserves its original neural source hash. `maze-backend.json` separately records the GPU environment source hashes and parent checkpoint provenance; the entry point rejects unreviewed environment backend changes on resume. `gpu-performance.jsonl` records each training evaluation's stage timings.

## Live run switch

The saved generation-143 population in `runs/gym-obstructedmaze-full-2048` resumed through the GPU entry point. Generation 143 completed and saved generation 144 successfully with the same neural source hash, config, settings and `fresh-training-only-v2` protocol. Its seed cursor advanced by exactly 8,192 episodes.

With all other searches resumed, the first complete generation took **135.85 seconds**, including training evaluation (**121.62 seconds**), breeding, CPU validation and checkpoint work. The last complete old-path generation took **1,049.46 seconds**. This observed live comparison is **7.73×** faster; the controlled matched benchmarks above establish about **5×**. The original checkpoint/controller/plan/status are backed up under `before-gpu-maze/`.

## Batch-size sweep

Two 512-step repetitions each at 8,192, 16,384 and 32,768 episodes, on frozen generation-143 brains, give **309,258**, **312,281** and **314,678** rollout steps/second. The larger sizes improve throughput only **0.98%** and **1.75%**; total benchmark throughput differs by roughly 1–2%. All common-seed episode results match exactly. Live batch size, population, and episode count remain unchanged. Detailed results are in `runs/obstructedmaze-gpu/batch-sweep/`. All training resumed after profiling.
