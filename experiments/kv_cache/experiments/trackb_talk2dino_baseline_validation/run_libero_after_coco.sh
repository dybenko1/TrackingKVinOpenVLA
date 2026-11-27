#!/usr/bin/env bash
set -u
CONTAINER=trackb-v1-coco-object-20261126
WT=/opt/openvla/cache/worktrees/trackb-dense-grounding-v2
COCO_STATUS="$WT/experiments/kv_cache/outputs/trackb_talk2dino_baseline_validation/coco_object/RUN_STATUS.json"
LOG="$WT/experiments/kv_cache/outputs/trackb_talk2dino_baseline_validation/coco_object/logs/talk2dino-coco.log"
while true; do
  state=$(docker exec "$CONTAINER" python -c 'import json,sys;print(json.load(open(sys.argv[1])).get("status","missing"))' "$COCO_STATUS" 2>/dev/null || echo missing)
  echo "$(date -Is) COCO status=$state"
  case "$state" in
    complete)
      docker exec "$CONTAINER" bash -lc "export PYTHONPATH=/opt/openvla/cache/venvs/v1-coco-eval-opencv:$WT; cd $WT; python -u -m experiments.kv_cache.experiments.trackb_talk2dino_baseline_validation.run_libero" 2>&1 | tee -a "$LOG"
      rc=${PIPESTATUS[0]}
      if [ "$rc" -ne 0 ]; then exit "$rc"; fi
      docker exec "$CONTAINER" python "$WT/experiments/kv_cache/experiments/trackb_talk2dino_baseline_validation/make_report.py" 2>&1 | tee -a "$LOG"
      docker exec "$CONTAINER" python -c 'import json,time; p="/opt/openvla/cache/worktrees/trackb-dense-grounding-v2/experiments/kv_cache/outputs/trackb_talk2dino_baseline_validation/RUN_STATUS.json"; open(p,"w").write(json.dumps({"status":"complete","stage":"report","updated_unix":time.time()},indent=2)+"\n")'
      exit 0
      ;;
    failed)
      echo "COCO stage failed; LIBERO scoring was not started."
      exit 1
      ;;
    *) sleep 300 ;;
  esac
done
