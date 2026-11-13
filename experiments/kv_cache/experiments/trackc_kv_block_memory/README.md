# Track C Experiment 1 — KV Block Memory Management

No LIBERO rollouts are run. The workload is nine saved Stage 3C `protect_10` R5 traces plus an isolated CUDA cache-update microbenchmark using the OpenVLA cache dimensions and one actual task4/state0 reuse mask.

## Container and execution

Use the established image `openvla-cached:latest`; no host Python dependencies or package changes are needed. The repository, LIBERO, robosuite, and `/tmp` bind mounts match the normal lab command. A detached Docker container keeps the process alive across SSH disconnects while the host and Docker daemon remain running. This is **not** configured for automatic recovery from a host reboot.

```bash
cd /home/oscar/projects/TrackingKVinOpenVLA/openvla
bash experiments/kv_cache/experiments/trackc_kv_block_memory/launch.sh
```

The launcher refuses to start a duplicate named container. To run phases manually in the same image:

```bash
docker run --rm --runtime nvidia --network host --group-add 2011 \
  -e NVIDIA_DRIVER_CAPABILITIES=all -e MUJOCO_GL=egl -e PYOPENGL_PLATFORM=egl \
  -e MPLCONFIGDIR=/tmp/matplotlib -e TF_CPP_MIN_LOG_LEVEL=3 \
  -v /home/oscar/projects/TrackingKVinOpenVLA/openvla:/opt/openvla \
  -v /home/oscar/projects/TrackingKVinOpenVLA/LIBERO:/opt/LIBERO \
  -v /home/oscar/projects/TrackingKVinOpenVLA/dusty-robosuite:/opt/robosuite \
  -v /tmp:/tmp -w /opt/openvla openvla-cached:latest \
  python3 experiments/kv_cache/experiments/trackc_kv_block_memory/run_queue.py
```

## Monitor, resume, stop

```bash
docker inspect -f '{{.State.Status}} exit={{.State.ExitCode}}' trackc-exp1
docker logs -f trackc-exp1
python3 experiments/kv_cache/experiments/trackc_kv_block_memory/status.py
tail -f experiments/kv_cache/outputs/trackc_kv_block_memory/logs/experiment.log
cat experiments/kv_cache/outputs/trackc_kv_block_memory/logs/progress.json
```

If the host account cannot read Docker-created files on a particular mount, use the status reader inside the running container: `docker exec trackc-exp1 python3 /opt/openvla/experiments/kv_cache/experiments/trackc_kv_block_memory/status.py /opt/openvla/experiments/kv_cache/outputs/trackc_kv_block_memory/logs/progress.json`. New atomic JSON files are written mode 0644 by the supervisor and benchmark.

If the container exits nonzero, inspect the phase and error in `progress.json`. Once that container is stopped, remove **only the stopped container** and resume with the same configuration (existing result files remain on the bind mount):

```bash
docker rm trackc-exp1
bash experiments/kv_cache/experiments/trackc_kv_block_memory/launch.sh
```

The GPU benchmark verifies the saved configuration hash and refuses incompatible resume data. It skips completed page-size/condition configurations and flushes raw CSV rows after each timed sample. For a controlled stop, allow the active configuration to finish; stopping the container can interrupt a sample, which is not counted as a completed configuration. Do not delete the output directory to resume.

## Phases and reproducibility

`run_queue.py` runs, in order: saved-trace analysis; exact correctness tests; 20 warmups + 50 timed repetitions per page-size/condition; report aggregation. Progress and status are atomic JSON under `outputs/trackc_kv_block_memory/logs/`. The detached container and the experiment process persist through SSH disconnects; tmux is unnecessary. Docker restart policy is intentionally not enabled, so host-reboot recovery is not claimed.

The correctness suite tests all-fresh, all-reused, one-fresh, contiguous, scattered, and partial final pages for capacities 1/4/8/16/32. The GPU benchmark reports synchronized wall time and CUDA-event time, peak allocated/reserved CUDA memory, a COW update without dense materialization, and COW with the dense K/V reconstruction required by the current DynamicCache/SDPA implementation. It does not measure full policy latency or hardware copy counters.
