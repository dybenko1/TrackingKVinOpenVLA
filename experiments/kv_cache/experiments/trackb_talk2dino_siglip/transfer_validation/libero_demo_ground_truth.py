"""Replay official LIBERO demonstrations and evaluate frozen V1/V2 masks.

This job never applies recorded actions. It restores recorded MuJoCo states,
renders RGB and instance labels from the same state, then runs only frozen
vision/text feature scoring. All data/model assets are read-only inputs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


REPO_ID = "yifengzhu-hf/LIBERO-datasets"
REPO_REVISION = "f13aa24a3da8c43c7225569f28c562979fa0e35a"
GRID = 16
PATCH_SIZE = 14
SEQUENCE_FRAMES = 32
DEMOS_PER_TASK = 3
V2_THRESHOLD = 0.20
V1_THRESHOLD = 0.55
V1_TEMPLATES = (
    "itap of a {}.", "a bad photo of a {}.", "a origami {}.",
    "a photo of the large {}.", "a {} in a video game.", "art of the {}.",
    "a photo of the small {}.",
)
V2_CHECKPOINT_REL = (
    "cache/worktrees/trackb-dense-grounding-v2/experiments/kv_cache/outputs/"
    "trackb_talk2dino_siglip/training_v2_dense_20261118/best_projection.pt"
)
V1_CHECKPOINT_REL = (
    "cache/worktrees/trackb-talk2dino-siglip/experiments/kv_cache/outputs/"
    "trackb_talk2dino_siglip/full_coco_training_20261117/checkpoints/epoch_050.pt"
)
V2_SHA256 = "8232ec8f657da47645d73741b16560a1140bfea89cce6826948b9457ce4854d4"
V1_SHA256 = "f1c57d5e3a52fde5ed6d17b5fad47aadaa6ce25dc58cab72a3aba9a180d015b1"
SIGLIP_ROOT = Path("/opt/openvla/cache/models/timm/ViT-SO400M-14-SigLIP-384")

# Predeclared task-level split, balanced within suite. This is frozen before
# model inference and is never changed based on predicted masks.
TASKS = [
    ("libero_spatial", "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5", "development"),
    ("libero_spatial", "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo.hdf5", "development"),
    ("libero_spatial", "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_demo.hdf5", "held_out"),
    ("libero_spatial", "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_demo.hdf5", "held_out"),
    ("libero_object", "pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5", "development"),
    ("libero_object", "pick_up_the_butter_and_place_it_in_the_basket_demo.hdf5", "development"),
    ("libero_object", "pick_up_the_salad_dressing_and_place_it_in_the_basket_demo.hdf5", "held_out"),
    ("libero_object", "pick_up_the_tomato_sauce_and_place_it_in_the_basket_demo.hdf5", "held_out"),
    ("libero_goal", "open_the_middle_drawer_of_the_cabinet_demo.hdf5", "development"),
    ("libero_goal", "put_the_bowl_on_the_plate_demo.hdf5", "development"),
    ("libero_goal", "open_the_top_drawer_and_put_the_bowl_inside_demo.hdf5", "held_out"),
    ("libero_goal", "put_the_wine_bottle_on_the_rack_demo.hdf5", "held_out"),
]
INITIAL_TASKS = {
    ("libero_spatial", TASKS[0][1]),
    ("libero_object", TASKS[4][1]),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temp, path)


def instruction_phrases(instruction: str) -> list[dict]:
    """Extract text-only object/fixture phrases with a frozen rule set.

    It reads only the task's language instruction. BDDL object identities and
    segmentation labels are never inputs to prompt generation.
    """
    text = re.sub(r"\s+", " ", instruction.lower().strip().strip(". "))
    found: list[dict] = []

    def add(phrase: str, role: str, pattern: str) -> None:
        phrase = re.sub(r"^[\s,.;:]+|[\s,.;:]+$", "", phrase)
        phrase = re.sub(r"^(?:the|a|an)\s+", "", phrase)
        phrase = re.sub(r"\s+", " ", phrase)
        if not phrase:
            return
        # For trained COCO concepts, use the category word used by V2 training.
        # This normalization is lexical and comes from the instruction only.
        synonyms = ((r"\b(?:akita )?(?:black )?bowl\b", "bowl"),
                    (r"\b(?:white |yellow |red )?mugs?\b", "cup"),
                    (r"\b(?:wine )?bottle\b", "bottle"),
                    (r"\b(?:dining )?table\b", "dining table"),
                    (r"\bchairs?\b", "chair"))
        for pat, canonical in synonyms:
            if re.search(pat, phrase):
                phrase = canonical
                break
        item = {"phrase": phrase, "role": role, "source": "task_instruction", "rule": pattern}
        if not any(x["phrase"] == phrase and x["role"] == role for x in found):
            found.append(item)

    # Explicit manipulation verbs and their grammatical object phrases.
    manip_patterns = (
        (r"\b(?:pick up|pick|grab) the (.+?)(?=\s+(?:between|from|next to|on top of|on|and place|and put|and move|to the|in the|inside the)\b|$)", "manipulated_object", "pick-object"),
        (r"\b(?:put|place|move) the (.+?)\s+(?:on|in|inside|into|onto)\s+(?:the\s+)?(.+?)(?=$|\s+and\s+)", "manipulated_object", "put-place-object-target"),
        (r"\bpush the (.+?)\s+(?:to|toward|towards)\s+(?:the\s+)?(.+)$", "manipulated_object", "push-object-target"),
    )
    for pat, role, rule in manip_patterns:
        match = re.search(pat, text)
        if match:
            add(match.group(1), role, rule)
            if match.lastindex and match.lastindex >= 2:
                add(match.group(2), "destination_or_fixture", rule)

    # Explicit goal/fixture nouns, plus targets of pronoun-based placement.
    for pat, role, rule in (
        (r"\b(?:open|close|turn on|turn off) the (.+?)(?=$|\s+and\s+)", "task_fixture", "fixture-command"),
        (r"\b(?:place|put|move) it\s+(?:on|in|inside|into|onto)\s+(?:the\s+)?(.+?)(?=$|\s+and\s+)", "destination_or_fixture", "pronoun-target"),
        (r"\b(?:place|put) the (.+?)\s+(?:inside|in|on|onto)\s+(?:the\s+)?(.+)$", "manipulated_object", "object-target"),
        (r"\bturn on the (.+?)\s+and\s+(?:put|place) the (.+?)\s+on it\b", "task_fixture", "fixture-plus-object"),
    ):
        match = re.search(pat, text)
        if not match:
            continue
        if rule == "fixture-plus-object":
            add(match.group(1), "task_fixture", rule)
            add(match.group(2), "manipulated_object", rule)
            add(match.group(1), "destination_or_fixture", rule)
        elif match.lastindex and match.lastindex >= 2:
            add(match.group(1), role, rule)
            add(match.group(2), "destination_or_fixture", rule)
        else:
            add(match.group(1), role, rule)

    # Preserve explicitly named contextual objects / fixtures as language
    # prompts too. This is a deliberately small grammar over the instruction,
    # not a lookup into BDDL labels or simulator masks.
    for pattern, role, rule in (
        (r"\bbetween\s+(?:the\s+)?(.+?)\s+and\s+(?:the\s+)?(.+?)(?=\s+and\s+place|$)", "mentioned_context_object", "between-context"),
        (r"\b(?:from|on|in|inside|near|next to|at|on top of)\s+(?:the\s+)?(wooden cabinet|table(?: center)?|stove|counter|drawer|rack)\b", "task_fixture", "spatial-fixture"),
        (r"\bbetween\s+(?:the\s+)?[^ ]+\s+and\s+(?:the\s+)?(ramekin|plate|basket)\b", "mentioned_context_object", "spatial-object-context"),
    ):
        match = re.search(pattern, text)
        if match:
            for group in match.groups():
                add(group, role, rule)

    if not found:
        # Fallback remains instruction-only and is explicitly marked.
        add(text, "instruction_fallback", "full-instruction-fallback")
    for prompt in ("robot arm", "robot gripper"):
        found.append({"phrase": prompt, "role": "robot_arm" if prompt.endswith("arm") else "robot_gripper",
                      "source": "fixed_robot_role_prompt", "rule": "always-include-robot-roles"})
    return found


def _region_base(symbol: str, entity_ids: list[str]) -> str | None:
    """Resolve a BDDL object/fixture region symbol to its owning instance."""
    if symbol in entity_ids:
        return symbol
    for entity in sorted(entity_ids, key=len, reverse=True):
        if symbol.startswith(entity + "_"):
            return entity
    return None


def parse_bddl_roles(path: Path) -> dict:
    """Extract simulator-only identities for ground-truth role assignment."""
    text = path.read_text(errors="replace")

    def section(name: str) -> str:
        match = re.search(r"\(:" + re.escape(name) + r"\s+(.*?)\n\s*\)", text, flags=re.S)
        return match.group(1) if match else ""

    def typed_names(value: str) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = defaultdict(list)
        for line in value.splitlines():
            line = line.split(";")[0].strip().strip("()")
            if not line or "-" not in line:
                continue
            left, right = line.split("-", 1)
            kind = right.strip().split()[0]
            groups[kind].extend(x for x in left.split() if x)
        return dict(groups)

    objects = typed_names(section("objects"))
    fixtures = typed_names(section("fixtures"))
    interest_body = section("obj_of_interest")
    interests = [x for x in re.findall(r"[A-Za-z][A-Za-z0-9_]*", interest_body)]
    goal_body = section("goal")
    goal_calls = []
    for match in re.finditer(r"\(([A-Za-z][A-Za-z0-9_]*)\s+([^()]+)\)", goal_body):
        pred = match.group(1).lower()
        args = [x for x in re.findall(r"[A-Za-z][A-Za-z0-9_]*", match.group(2)) if x.lower() not in {"and"}]
        if args:
            goal_calls.append({"predicate": pred, "args": args})
    object_ids = sorted({x for xs in objects.values() for x in xs})
    fixture_ids = sorted({x for xs in fixtures.values() for x in xs})
    role_names: dict[str, set[str]] = {"manipulated_objects": set(), "destination_objects": set(),
                                      "task_relevant_fixtures": set(), "task_relevant_objects": set()}
    for goal in goal_calls:
        args = goal["args"]
        pred = goal["predicate"]
        if pred in {"on", "inside", "in", "contains", "under", "nextto", "ontop"} and args:
            subject, *targets = args
            subject_entity = _region_base(subject, object_ids)
            if subject_entity:
                role_names["manipulated_objects"].add(subject_entity)
                role_names["task_relevant_objects"].add(subject_entity)
            for target in targets:
                target_object = _region_base(target, object_ids)
                target_fixture = _region_base(target, fixture_ids)
                if target_object:
                    role_names["destination_objects"].add(target_object)
                    role_names["task_relevant_objects"].add(target_object)
                elif target_fixture:
                    role_names["task_relevant_fixtures"].add(target_fixture)
        elif args:
            target = args[0]
            target_object = _region_base(target, object_ids)
            target_fixture = _region_base(target, fixture_ids)
            if target_object:
                role_names["manipulated_objects"].add(target_object)
                role_names["task_relevant_objects"].add(target_object)
            elif target_fixture:
                role_names["task_relevant_fixtures"].add(target_fixture)
            else:
                for fixture in fixture_ids:
                    if target == fixture or target.startswith(fixture + "_"):
                        role_names["task_relevant_fixtures"].add(fixture)
                        break
    for target in interests:
        if target in object_ids:
            role_names["task_relevant_objects"].add(target)
            if target not in role_names["destination_objects"]:
                role_names["manipulated_objects"].add(target)
        else:
            for fixture in fixture_ids:
                if target == fixture or target.startswith(fixture + "_"):
                    role_names["task_relevant_fixtures"].add(fixture)
                    break
    return {"objects_by_type": objects, "fixtures_by_type": fixtures,
            "object_ids": object_ids, "fixture_ids": fixture_ids,
            "obj_of_interest_symbols": interests, "goal_calls": goal_calls,
            "role_instance_names": {k: sorted(v) for k, v in role_names.items()}}


def add_instruction_named_fixtures(roles: dict, instruction: str) -> None:
    """Mark BDDL fixture instances explicitly named in task language as relevant.

    This augments ground-truth role metadata only. Prompt extraction remains a
    separate instruction-only operation and does not consult these identities.
    """
    text = re.sub(r"[^a-z0-9]+", " ", instruction.lower()).strip()
    relevant = set(roles["role_instance_names"].get("task_relevant_fixtures", []))
    for values in roles.get("fixtures_by_type", {}).values():
        for name in values:
            words = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
            if words and re.search(r"\b" + re.escape(words) + r"\b", text):
                relevant.add(name)
    roles["role_instance_names"]["task_relevant_fixtures"] = sorted(relevant)


def role_for_instance(name: str, roles: dict) -> str:
    if name.startswith(("PandaGripper", "MountedPandaGripper", "OnTheGroundPandaGripper")):
        return "robot_gripper"
    if name.startswith(("Panda", "MountedPanda", "OnTheGroundPanda")):
        return "robot_arm"
    if name.startswith("RethinkMount"):
        return "robot_mount"
    for role, values in roles["role_instance_names"].items():
        if name in values:
            if role == "task_relevant_objects" and name in roles["role_instance_names"]["manipulated_objects"]:
                return "manipulated_objects"
            return role
    if name in roles["object_ids"]:
        return "other_objects"
    if name in roles["fixture_ids"]:
        return "static_scene_background"
    return "unresolved_instance"


def target_prompt_kind(phrase: str, bddl_roles: dict) -> str | None:
    """Post-hoc object-vs-fixture association for a language target prompt.

    This is only used to score predictions against simulator labels after the
    prompt and patch scores are already fixed. It never changes model input.
    """
    words = set(re.sub(r"[^a-z0-9]+", " ", phrase.lower()).split())
    articles = {"a", "an", "the", "of", "and", "on", "in", "inside", "into"}
    words -= articles
    if not words:
        return None
    matches = {}
    for kind, groups in (("object", bddl_roles.get("objects_by_type", {})),
                         ("fixture", bddl_roles.get("fixtures_by_type", {}))):
        best = 0
        for entity_type, names in groups.items():
            candidates = [entity_type, *names]
            for candidate in candidates:
                tokens = set(re.sub(r"[^a-z0-9]+", " ", candidate.lower()).split())
                overlap = len(words & tokens)
                if overlap == len(words) and overlap > best:
                    best = overlap
        if best:
            matches[kind] = best
    if not matches:
        return None
    highest = max(matches.values())
    winners = [kind for kind, score in matches.items() if score == highest]
    return winners[0] if len(winners) == 1 else None


def expected_tasks() -> list[dict]:
    return [{"suite": suite, "file_name": filename, "split": split, "instruction": None,
             "task_id": f"{suite}/{filename.removesuffix('_demo.hdf5')}"}
            for suite, filename, split in TASKS]


def download_and_audit(task: dict, data_root: Path, source_revision: str, api, logger) -> dict:
    from huggingface_hub import hf_hub_download
    rel = f"{task['suite']}/{task['file_name']}"
    metadata = api.get_paths_info(REPO_ID, [rel], repo_type="dataset", revision=source_revision)
    if len(metadata) != 1 or not getattr(metadata[0], "lfs", None):
        raise RuntimeError(f"Official HDF5 LFS checksum metadata missing for {rel}")
    meta = metadata[0]
    expected_size = int(meta.size)
    expected_sha = str(meta.lfs.sha256)
    available = os.statvfs(data_root).f_bavail * os.statvfs(data_root).f_frsize
    if available < expected_size + (1 << 30):
        raise RuntimeError(f"Insufficient disk before {rel}: available={available}, file={expected_size}")
    path = data_root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        got_size = path.stat().st_size
        got_sha = sha256_file(path) if got_size == expected_size else None
        if got_size == expected_size and got_sha == expected_sha:
            logger(f"verified-cache hit {rel} bytes={got_size} sha256={got_sha}")
        else:
            # Preserve any corrupt/incomplete file for diagnosis; never truncate it.
            quarantine = path.with_name(path.name + f".unverified.{int(time.time())}")
            os.replace(path, quarantine)
            logger(f"quarantined nonmatching file {rel} at {quarantine}")
            path = None
    else:
        path = None
    if path is None:
        logger(f"download begin {rel} bytes={expected_size} revision={source_revision}")
        downloaded = Path(hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=rel,
                                          revision=source_revision, local_dir=str(data_root),
                                          force_download=False))
        if not downloaded.is_file():
            raise FileNotFoundError(downloaded)
        got_size = downloaded.stat().st_size
        got_sha = sha256_file(downloaded)
        if got_size != expected_size or got_sha != expected_sha:
            raise RuntimeError(f"Integrity failure {rel}: size={got_size}/{expected_size}, sha256={got_sha}/{expected_sha}")
        path = downloaded
        logger(f"download verified {rel} bytes={got_size} sha256={got_sha}")

    task["hdf5_path"] = str(path)
    task["hdf5_size_bytes"] = expected_size
    task["hdf5_sha256"] = expected_sha
    task["hf_blob_id"] = str(meta.blob_id)
    task["hdf5_repo_path"] = rel
    task["hdf5_verified"] = True

    import h5py
    with h5py.File(path, "r") as f:
        if "data" not in f:
            raise RuntimeError(f"Missing data group in {rel}")
        data = f["data"]
        attr_names = list(data.attrs.keys())
        demos = sorted(k for k in data.keys() if re.fullmatch(r"demo_\d+", k))
        if not demos:
            raise RuntimeError(f"No demo_N groups in {rel}")
        hdf5_instruction = None
        if "problem_info" in data.attrs:
            problem = data.attrs["problem_info"]
            if isinstance(problem, bytes): problem = problem.decode("utf-8")
            if isinstance(problem, str):
                try: problem = json.loads(problem)
                except json.JSONDecodeError: problem = {"raw": problem}
            if isinstance(problem, dict): hdf5_instruction = problem.get("language_instruction")
        episode_summary = []
        for demo_name in demos:
            demo = data[demo_name]
            if "states" not in demo or "actions" not in demo:
                raise RuntimeError(f"{rel}:{demo_name} lacks states or actions")
            states = demo["states"]
            actions = demo["actions"]
            if states.ndim != 2 or actions.ndim < 1 or not len(states) or not len(actions):
                raise RuntimeError(f"Malformed state/action arrays in {rel}:{demo_name}: {states.shape}/{actions.shape}")
            if not np.isfinite(states[:]).all() or not np.isfinite(actions[:]).all():
                raise FloatingPointError(f"Nonfinite state/action data in {rel}:{demo_name}")
            model_xml = demo.attrs.get("model_file", "")
            if isinstance(model_xml, bytes): model_xml = model_xml.decode("utf-8")
            if not model_xml:
                raise RuntimeError(f"Missing recorded model_file in {rel}:{demo_name}")
            episode_summary.append({"demo": demo_name, "state_shape": list(states.shape),
                                    "action_shape": list(actions.shape), "actions_used_for_replay": False,
                                    "model_xml_sha256": sha256_bytes(model_xml.encode("utf-8")),
                                    "recorded_samples_attr": int(demo.attrs.get("num_samples", len(states)))})
        task["hdf5_schema"] = {"root_groups": sorted(f.keys()), "data_attributes": attr_names,
                               "env": str(data.attrs.get("env", "")),
                               "repository_version": str(data.attrs.get("repository_version", "")),
                               "bddl_file_name": str(data.attrs.get("bddl_file_name", "")),
                               "env_info_present": "env_info" in data.attrs,
                               "problem_info_present": "problem_info" in data.attrs,
                               "language_instruction": hdf5_instruction,
                               "episode_count": len(demos), "episodes": episode_summary}
        task["instruction"] = hdf5_instruction
        task["episode_names"] = demos
        task["camera_keys_by_episode"] = {}
        for demo_name in demos[:1]:
            obs_group = f[f"data/{demo_name}/obs"] if f"data/{demo_name}/obs" in f else None
            task["camera_keys_by_episode"][demo_name] = sorted(obs_group.keys()) if obs_group is not None else []
    return task


def setup_segmentation_env(task: dict, libero_root: Path, size: int = 128):
    from libero.libero.envs import SegmentationRenderEnv
    bddl = (libero_root / "libero/libero/bddl_files" / task["suite"] /
            (task["file_name"].removesuffix("_demo.hdf5") + ".bddl"))
    if not bddl.is_file():
        raise FileNotFoundError(f"No exact BDDL match for official HDF5 task: {bddl}")
    return SegmentationRenderEnv(
        bddl_file_name=str(bddl), robots=["Panda"], controller="OSC_POSE",
        use_camera_obs=True, has_renderer=False, has_offscreen_renderer=True,
        render_camera="agentview", camera_names=["agentview"],
        camera_heights=size, camera_widths=size, camera_depths=False,
        camera_segmentations="instance", control_freq=20, horizon=1000,
        ignore_done=True, hard_reset=True, renderer="mujoco",
    ), bddl


def remap_demo_xml_assets(model_xml: str, libero_root: Path, robosuite_root: Path) -> str:
    """Resolve dataset-author absolute asset paths to installed official assets.

    The public demo XMLs reference the author's ``chiliocosm`` and
    ``robosuite-master`` checkouts. The installed robosuite rewriter assumes
    every path contains a ``robosuite`` segment, which is false for the
    LIBERO asset paths. Resolve each asset by its documented path suffix and
    require the exact target file to exist; no missing asset is substituted.
    """
    import xml.etree.ElementTree as ET
    tree = ET.fromstring(model_xml)
    asset = tree.find("asset")
    if asset is None:
        raise RuntimeError("Recorded MuJoCo XML has no asset block")
    libero_assets = (libero_root / "libero/libero/assets").resolve()
    robosuite_assets = (robosuite_root / "robosuite/models/assets").resolve()
    rewritten = []
    for element in asset.iter():
        old = element.get("file")
        if not old:
            continue
        if "/chiliocosm/assets/" in old:
            suffix = old.split("/chiliocosm/assets/", 1)[1]
            target = (libero_assets / suffix).resolve()
        elif "/robosuite-master/robosuite/models/assets/" in old:
            suffix = old.split("/robosuite-master/robosuite/models/assets/", 1)[1]
            target = (robosuite_assets / suffix).resolve()
        elif old.startswith("/opt/LIBERO/libero/libero/assets/") or old.startswith("/opt/robosuite/robosuite/models/assets/"):
            target = Path(old).resolve()
        else:
            raise RuntimeError(f"Unrecognized recorded asset root: {old}")
        if not target.is_file():
            raise FileNotFoundError(f"Recorded asset has no exact installed counterpart: {old} -> {target}")
        element.set("file", str(target))
        rewritten.append({"recorded": old, "resolved": str(target)})
    return ET.tostring(tree, encoding="utf8").decode("utf8")


def prepare_demo_env(env, model_xml: str, *, libero_root: Path, robosuite_root: Path) -> list[str]:
    env.reset()
    xml = remap_demo_xml_assets(model_xml, libero_root, robosuite_root)
    env.reset_from_xml_string(xml)
    env.env.sim.reset()
    return list(env.env.model.instances_to_ids.keys())


def direct_mapping_check(env, obs, instance_names: list[str], size: int) -> dict:
    """Cross-check RGB convention and exact rendered geom->instance labels."""
    from robosuite import macros
    from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING
    conv = IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]
    model = env.env.model
    name_to_id = {name: i for i, name in enumerate(instance_names)}
    geom_map = {int(geom_id): name_to_id[name]
                for geom_id, name in model.geom_ids_to_instances.items() if name in name_to_id}
    raw_rgb = env.env.sim.render(camera_name="agentview", width=size, height=size, depth=False)
    if isinstance(raw_rgb, tuple): raw_rgb = raw_rgb[0]
    direct_rgb = raw_rgb[::conv, :, :]
    direct_seg = env.env.sim.render(camera_name="agentview", width=size, height=size,
                                    depth=False, segmentation=True)
    if isinstance(direct_seg, tuple): direct_seg = direct_seg[0]
    if direct_seg.ndim != 3 or direct_seg.shape[-1] < 2:
        raise RuntimeError(f"Unexpected raw segmentation render shape {direct_seg.shape}")
    raw_geom = direct_seg[::conv, :, 1]
    expected = np.fromiter((geom_map.get(int(g), -1) for g in raw_geom.ravel()),
                           dtype=np.int32, count=raw_geom.size).reshape(size, size) + 1
    obs_rgb = np.asarray(obs["agentview_image"])
    obs_seg = np.asarray(obs["agentview_segmentation_instance"]).squeeze(-1).astype(np.int32)
    if direct_rgb.shape != obs_rgb.shape or not np.array_equal(direct_rgb, obs_rgb):
        raise RuntimeError(f"RGB orientation/render parity failed: raw={direct_rgb.shape} obs={obs_rgb.shape}, equal={np.array_equal(direct_rgb,obs_rgb)}")
    if expected.shape != obs_seg.shape or not np.array_equal(expected, obs_seg):
        bad = int(np.count_nonzero(expected != obs_seg)) if expected.shape == obs_seg.shape else -1
        raise RuntimeError(f"Rendered-label mapping mismatch: expected shape={expected.shape}, observed={obs_seg.shape}, differing pixels={bad}")
    unique, counts = np.unique(obs_seg, return_counts=True)
    by_label = {int(label): {"instance_name": "background" if label == 0 else instance_names[int(label)-1],
                             "pixel_count": int(count)} for label, count in zip(unique, counts)}
    return {"image_convention": str(macros.IMAGE_CONVENTION), "vertical_slice_step": conv,
            "raw_geom_channel": 1, "rendered_label_convention": "instance_order_index + 1; 0 background",
            "instance_name_order": instance_names,
            "direct_raw_geom_to_name_mapping_equals_observation": True,
            "rgb_direct_render_equals_observation": True,
            "unique_rendered_labels": by_label,
            "special_wrapper_behavior": "SegmentationRenderEnv.get_segmentation_instances collapses labels above the Panda0 robot ID into a robot label and omits Panda0/RethinkMount0/PandaGripper0 from the ordinary mapping; this audit instead validates the lower-level instance sensor before that helper's collapse so arm, mount and gripper IDs remain distinguishable."}


def render_one_state(task: dict, demo_name: str, state_index: int, *, libero_root: Path,
                     robosuite_root: Path = Path("/opt/robosuite"),
                     image_size: int = 128, validate_mapping: bool = False):
    import h5py
    bddl = (libero_root / "libero/libero/bddl_files" / task["suite"] /
            (task["file_name"].removesuffix("_demo.hdf5") + ".bddl"))
    with h5py.File(task["hdf5_path"], "r") as f:
        demo = f[f"data/{demo_name}"]
        xml = demo.attrs["model_file"]
        if isinstance(xml, bytes): xml = xml.decode("utf-8")
        states = demo["states"]
        state = np.asarray(states[state_index], dtype=np.float64)
        state_digest = sha256_bytes(np.ascontiguousarray(states[max(0,state_index-1):min(len(states),state_index+2)]).tobytes())
    env, bddl = setup_segmentation_env(task, libero_root, image_size)
    try:
        names = prepare_demo_env(env, xml, libero_root=libero_root,
                                 robosuite_root=robosuite_root)
        sim_state_size = int(env.env.sim.get_state().flatten().shape[0])
        if len(state) != sim_state_size:
            raise RuntimeError(f"Recorded state length {len(state)} != installed simulator state length {sim_state_size}")
        env.env.sim.set_state_from_flattened(state)
        env.env.sim.forward()
        env.check_success()
        env.env._post_process()
        env.env._update_observables(force=True)
        obs = env.env._get_observations()
        rgb = np.asarray(obs["agentview_image"]).copy()
        labels = np.asarray(obs["agentview_segmentation_instance"]).squeeze(-1).astype(np.int32)
        if rgb.shape != (image_size, image_size, 3) or labels.shape != (image_size, image_size):
            raise RuntimeError(f"Unexpected camera shape RGB={rgb.shape}, labels={labels.shape}")
        if validate_mapping:
            mapping = direct_mapping_check(env, obs, names, image_size)
        else:
            mapping = None
        visible = {}
        for label, count in zip(*np.unique(labels, return_counts=True)):
            label = int(label)
            visible[str(label)] = {"name": "background" if label == 0 else names[label-1], "pixels": int(count)}
        return {"rgb": rgb, "labels": labels, "instance_names": names,
                "state_length": len(state), "sim_state_length": sim_state_size,
                "state_segment_sha256": state_digest, "visible_instances": visible,
                "mapping_validation": mapping, "bddl_path": str(bddl),
                "rgb_from_same_state": True, "actions_applied": False}
    finally:
        env.close()


def write_alignment_figure(result: dict, task: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from experiments.kv_cache.experiments.trackb_talk2dino_siglip.training_v2_dense.masks import (
        apply_mask_geometry, fractional_grid,
    )
    rgb = result["rgb"]
    labels = result["labels"]
    names = result["instance_names"]
    # Smoke uses square 128 input and the audited OpenVLA 224 resize/crop. Use
    # a provisional exact 224 resize here; full scoring records the processor's
    # actual geometry and regenerates its gallery after checking parity.
    expanded = Image.fromarray(rgb).resize((224, 224), Image.Resampling.BICUBIC)
    mask = np.zeros_like(labels, dtype=np.uint8)
    for label in np.unique(labels):
        if label == 0: continue
        name = names[int(label)-1]
        if name.startswith(("Panda", "MountedPanda", "OnTheGroundPanda", "RethinkMount")):
            mask[labels == label] = 1
    mask224 = np.asarray(Image.fromarray(mask).resize((224, 224), Image.Resampling.NEAREST))
    occupancy = fractional_grid(mask224)
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    axes[0].imshow(rgb); axes[0].set_title(f"Replay RGB ({rgb.shape[1]}×{rgb.shape[0]})")
    axes[1].imshow(labels, interpolation="nearest"); axes[1].set_title("Rendered instance IDs")
    masked_robot = np.ma.masked_where(mask224 == 0, mask224)
    robot_cmap = plt.get_cmap("cool").copy(); robot_cmap.set_bad(alpha=0)
    axes[2].imshow(expanded); axes[2].imshow(masked_robot, alpha=.72, cmap=robot_cmap, interpolation="nearest")
    axes[2].set_title("Robot-mask overlay at policy size")
    axes[3].imshow(occupancy.reshape(16,16), vmin=0, vmax=1, cmap="viridis", interpolation="nearest")
    axes[3].set_title("Robot occupancy on 16×16 grid")
    for ax in axes: ax.axis("off")
    fig.suptitle(f"Same-state simulator replay · {task['task_id']} · {task['split']}")
    fig.tight_layout(); output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=130); plt.close(fig)


def bddl_instance_roles(instance_names: list[str], bddl_roles: dict) -> dict[str, str]:
    return {name: role_for_instance(name, bddl_roles) for name in instance_names}


def grid_masks(labels: np.ndarray, names: list[str], bddl_roles: dict, geometry: dict) -> tuple[dict, dict, np.ndarray]:
    from experiments.kv_cache.experiments.trackb_talk2dino_siglip.training_v2_dense.masks import (
        apply_mask_geometry, fractional_grid,
    )
    roles = bddl_instance_roles(names, bddl_roles)
    inst_masks, role_masks = {}, defaultdict(lambda: np.zeros((GRID*GRID,), dtype=np.float32))
    unresolved = np.zeros(labels.shape, dtype=np.uint8)
    for label, name in enumerate(names, start=1):
        pixel = (labels == label).astype(np.uint8)
        if not pixel.any(): continue
        role = roles[name]
        transformed = apply_mask_geometry(pixel, geometry)
        occupancy = fractional_grid(transformed)
        inst_masks[name] = occupancy
        if role == "unresolved_instance": unresolved |= pixel
        else: role_masks[role] += occupancy
        # Also preserve an overall task-relevant union for R5/cache-proxy metrics.
    for key in role_masks:
        role_masks[key] = np.clip(role_masks[key], 0.0, 1.0)
    relevant = np.zeros((GRID*GRID,), dtype=np.float32)
    for key in ("manipulated_objects", "destination_objects", "task_relevant_objects",
                "task_relevant_fixtures", "robot_arm", "robot_gripper", "robot_mount"):
        if key in role_masks: relevant = np.maximum(relevant, role_masks[key])
    valid224 = apply_mask_geometry((unresolved == 0).astype(np.uint8), geometry)
    valid16 = fractional_grid(valid224)
    return {k: np.asarray(v, dtype=np.float32) for k,v in role_masks.items()}, inst_masks, np.stack((relevant, valid16), axis=0)


def prompt_score_inputs(task: dict, device: str):
    import torch
    from experiments.kv_cache.experiments.trackb_talk2dino_siglip.text_encoder import (
        encode_text_embeddings, load_siglip_text_tower,
    )
    text_model, tokenizer, metadata = load_siglip_text_tower(SIGLIP_ROOT, device=device)
    prompt_specs = instruction_phrases(task["instruction"] or "")
    v2_strings = [f"a photo of a {p['phrase']}" for p in prompt_specs]
    v1_strings = [template.format(p["phrase"]) for p in prompt_specs for template in V1_TEMPLATES]
    with torch.inference_mode():
        v2_ids = tokenizer(v2_strings).to(device)
        v1_ids = tokenizer(v1_strings).to(device)
        v2_emb = encode_text_embeddings(text_model, v2_ids).float()
        v1_all = encode_text_embeddings(text_model, v1_ids).float()
    v1_emb = v1_all.reshape(len(prompt_specs), len(V1_TEMPLATES), -1).mean(dim=1)
    return prompt_specs, v2_emb, v1_emb, metadata


def load_models(repo_root: Path, device: str, logger):
    import torch
    import torch.nn.functional as F
    from experiments.kv_cache.experiments.trackb_talk2dino_siglip.training_v2_dense.extract import (
        load_native_dino, process_image_batch,
    )
    from experiments.kv_cache.experiments.trackb_talk2dino_siglip.projection import RegionTextProjection
    v2_path = repo_root / V2_CHECKPOINT_REL
    v1_path = repo_root / V1_CHECKPOINT_REL
    for path, expected in ((v2_path, V2_SHA256), (v1_path, V1_SHA256)):
        got = sha256_file(path)
        if got != expected: raise RuntimeError(f"Checkpoint integrity mismatch {path}: {got} != {expected}")
    model, processor, vision, dino, geometry, _ = load_native_dino()
    dino.eval()
    # Explicitly validate that the V2 path is the policy's penultimate DINO
    # forward and V1 remains its established final-normalized feature control.
    v2_raw = torch.load(v2_path, map_location="cpu", weights_only=True)
    v2 = RegionTextProjection().to(device)
    v2.load_state_dict(v2_raw["model"], strict=True); v2.eval()
    # V1's trusted epoch checkpoint also stores NumPy RNG metadata. Keep
    # weights-only loading enabled and allow only the inert NumPy/PyTorch
    # metadata types used by this repository's own checkpoint writer.
    np_core = getattr(np, "_core", np.core)
    safe_types = [np_core.multiarray._reconstruct, np.ndarray, np.dtype]
    if hasattr(torch, "torch_version"):
        safe_types.append(torch.torch_version.TorchVersion)
    if hasattr(np, "dtypes") and hasattr(np.dtypes, "UInt32DType"):
        safe_types.append(np.dtypes.UInt32DType)
    with torch.serialization.safe_globals(safe_types):
        v1_raw = torch.load(v1_path, map_location="cpu", weights_only=True)
    v1_state = v1_raw.get("model", v1_raw.get("state_dict", v1_raw))
    v1 = RegionTextProjection().to(device)
    v1.load_state_dict(v1_state, strict=True); v1.eval()
    for network in (v2, v1):
        for p in network.parameters(): p.requires_grad_(False)
    logger(f"V2 SHA256={V2_SHA256}; V1 epoch50 SHA256={V1_SHA256}; device={device}")
    return {"openvla": model, "processor": processor, "dino": dino, "geometry": geometry,
            "v2": v2, "v1": v1, "torch": torch, "F": F, "process_image_batch": process_image_batch,
            "device": device, "checkpoint_paths": {"v2": str(v2_path), "v1": str(v1_path)}}


def score_one_frame(image: np.ndarray, models: dict, prompt_specs: list, text_v2, text_v1) -> tuple[np.ndarray, np.ndarray]:
    from PIL import Image
    torch, F, device = models["torch"], models["F"], models["device"]
    pil = Image.fromarray(image.astype(np.uint8), mode="RGB")
    pixels = models["processor"].image_processor(images=[pil], return_tensors="pt")["pixel_values"]
    if pixels.ndim != 4 or pixels.shape[-2:] != (224, 224):
        raise RuntimeError(f"OpenVLA processor returned unexpected image tensor {tuple(pixels.shape)}")
    x = pixels[:, :3].to(device=device, dtype=next(models["dino"].parameters()).dtype)
    with torch.inference_mode():
        penultimate = models["dino"](x)
        if isinstance(penultimate, (tuple, list)): penultimate = penultimate[0]
        if tuple(penultimate.shape) != (1, 256, 1024):
            raise RuntimeError(f"Policy DINO feature shape is {tuple(penultimate.shape)}")
        final = models["dino"].get_intermediate_layers(
            x, n={len(models["dino"].blocks)-1}, norm=True,
        )[0]
        if tuple(final.shape) != (1, 256, 1024):
            raise RuntimeError(f"V1 final DINO feature shape is {tuple(final.shape)}")
        p2 = F.normalize(penultimate[0].float(), dim=-1)
        p1 = F.normalize(final[0].float(), dim=-1)
        t2 = F.normalize(models["v2"].text_projection(text_v2.float()), dim=-1)
        t1 = F.normalize(models["v1"].text_projection(text_v1.float()), dim=-1)
        cosine2 = t2 @ p2.T
        cosine1 = t1 @ p1.T
        v2_prob = torch.sigmoid(10.0 * (cosine2 - 0.20))
        v1_prob = torch.sigmoid(cosine1)
    a = v2_prob.detach().cpu().numpy().astype(np.float16)
    b = v1_prob.detach().cpu().numpy().astype(np.float16)
    if a.shape != (len(prompt_specs), 256) or b.shape != (len(prompt_specs), 256):
        raise RuntimeError(f"Malformed score maps V2={a.shape} V1={b.shape}")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise FloatingPointError("Nonfinite frozen V1/V2 patch scores")
    return a, b


def select_segments(demo_names: list[str], lengths: dict[str, int]) -> list[dict]:
    eligible = [(n, lengths[n]) for n in demo_names if lengths[n] >= SEQUENCE_FRAMES]
    selected = []
    for index, (name, count) in enumerate(eligible[:DEMOS_PER_TASK]):
        max_start = count - SEQUENCE_FRAMES
        if index == 0: start = 0
        elif index == DEMOS_PER_TASK - 1: start = max_start
        else: start = max_start // 2
        selected.append({"demo": name, "state_count": count, "start": int(start),
                         "end_exclusive": int(start + SEQUENCE_FRAMES),
                         "sample_count": SEQUENCE_FRAMES, "consecutive": True})
    return selected


def save_smoke_image(result: dict, task: dict, out: Path) -> None:
    write_alignment_figure(result, task, out)


def metric_counts(pred: np.ndarray, target: np.ndarray, valid: np.ndarray) -> tuple[float, float, float, float]:
    p = np.asarray(pred, dtype=np.float32).reshape(-1)
    y = np.asarray(target, dtype=np.float32).reshape(-1)
    v = np.asarray(valid, dtype=np.float32).reshape(-1)
    tp = float((p * y * v).sum()); fp = float((p * (1-y) * v).sum())
    fn = float(((1-p) * y * v).sum()); tn = float(((1-p) * (1-y) * v).sum())
    return tp, fp, fn, tn


def get_union_mask(scores: np.ndarray, threshold: float, prompt_indices: list[int] | None = None) -> np.ndarray:
    if prompt_indices is None: prompt_indices = list(range(scores.shape[0]))
    if not prompt_indices: return np.zeros((256,), dtype=bool)
    return np.any(scores[np.asarray(prompt_indices)] >= threshold, axis=0)


def save_sequence_cache(path: Path, arrays: dict, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".npz.tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays, metadata=np.asarray(json.dumps(meta, sort_keys=True)))
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def load_sequence_cache(path: Path, expected_manifest_hash: str, checkpoint_hashes: dict):
    if not path.exists(): return None
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["metadata"].item()))
        if meta.get("task_manifest_sha256") != expected_manifest_hash or meta.get("checkpoint_sha256") != checkpoint_hashes:
            raise RuntimeError(f"Refusing to mix cached sequence from another manifest/checkpoint: {path}")
        return {k: np.array(z[k]) for k in z.files if k != "metadata"}, meta


def process_sequence(task: dict, segment: dict, *, models: dict, prompt_specs: list,
                     text_v2, text_v1, bddl_roles: dict, geometry: dict,
                     libero_root: Path, robosuite_root: Path, image_size: int, run_cache: Path,
                     manifest_hash: str, checkpoint_hashes: dict, logger) -> tuple[dict, dict]:
    task_slug = re.sub(r"[^A-Za-z0-9_-]+", "_", task["task_id"])
    cache_path = run_cache / "scoring" / task_slug / f"{segment['demo']}_s{segment['start']:05d}.npz"
    cached = load_sequence_cache(cache_path, manifest_hash, checkpoint_hashes)
    if cached is not None:
        logger(f"resume verified sequence {task['task_id']} {segment['demo']} start={segment['start']}")
        return cached[0], cached[1]

    import h5py
    env, bddl_path = setup_segmentation_env(task, libero_root, image_size)
    try:
        with h5py.File(task["hdf5_path"], "r") as f:
            demo = f[f"data/{segment['demo']}"]
            xml = demo.attrs["model_file"]
            if isinstance(xml, bytes): xml = xml.decode("utf-8")
            state_ds = demo["states"]
            if len(state_ds) < segment["end_exclusive"]: raise RuntimeError("Selected demo segment no longer matches manifest")
            states = np.asarray(state_ds[segment["start"]:segment["end_exclusive"]], dtype=np.float64)
        instance_names = prepare_demo_env(env, xml, libero_root=libero_root,
                                          robosuite_root=robosuite_root)
        instance_roles = bddl_instance_roles(instance_names, bddl_roles)
        if len(states) != SEQUENCE_FRAMES: raise RuntimeError(f"Expected {SEQUENCE_FRAMES} consecutive states, got {len(states)}")
        state_length = int(env.env.sim.get_state().flatten().shape[0])
        if states.shape[1] != state_length: raise RuntimeError(f"State width {states.shape[1]} != sim state width {state_length}")
        score_v2, score_v1, role_gt, instance_gt, valid_rows = [], [], [], [], []
        frame_rgb_for_gallery = None
        step_diagnostics = []
        for offset, state in enumerate(states):
            absolute_step = segment["start"] + offset
            env.env.sim.set_state_from_flattened(state)
            env.env.sim.forward(); env.check_success(); env.env._post_process()
            env.env._update_observables(force=True)
            obs = env.env._get_observations()
            rgb = np.asarray(obs["agentview_image"]).copy()
            labels = np.asarray(obs["agentview_segmentation_instance"]).squeeze(-1).astype(np.int32)
            if rgb.shape != (image_size, image_size, 3) or labels.shape != (image_size, image_size):
                raise RuntimeError(f"Unexpected rendered pair: rgb={rgb.shape}, instance={labels.shape}")
            unique = np.unique(labels)
            if (unique < 0).any() or (unique > len(instance_names)).any():
                raise RuntimeError(f"Rendered segmentation IDs outside index+1 map: {unique.tolist()}")
            role_masks, inst_masks, joint = grid_masks(labels, instance_names, bddl_roles, geometry)
            scores2, scores1 = score_one_frame(rgb, models, prompt_specs, text_v2, text_v1)
            score_v2.append(scores2); score_v1.append(scores1)
            role_gt.append(role_masks)
            instance_gt.append(inst_masks)
            valid_rows.append(joint[1])
            if frame_rgb_for_gallery is None: frame_rgb_for_gallery = rgb
            step_diagnostics.append({"frame_index": int(offset), "state_index": int(absolute_step),
                                     "state_sha256": sha256_bytes(np.ascontiguousarray(state).tobytes()),
                                     "rendered_instance_ids": [int(x) for x in unique],
                                     "pixel_counts": {str(int(x)): int((labels == x).sum()) for x in unique},
                                     "instance_names": instance_names})
            if offset % 8 == 7:
                logger(f"score {task['task_id']} {segment['demo']} frame={offset+1}/{SEQUENCE_FRAMES}")

        role_keys = sorted({r for row in role_gt for r in row})
        # Use a deterministic role order and preserve category/instance IDs.
        role_arr = np.zeros((SEQUENCE_FRAMES, len(role_keys), 256), dtype=np.float16)
        for ti, row in enumerate(role_gt):
            for ri, role in enumerate(role_keys): role_arr[ti, ri] = row.get(role, np.zeros(256, np.float32))
        instance_keys = sorted({name for row in instance_gt for name in row})
        instance_arr = np.zeros((SEQUENCE_FRAMES, len(instance_keys), 256), dtype=np.float16)
        for ti, row in enumerate(instance_gt):
            for ii, name in enumerate(instance_keys): instance_arr[ti, ii] = row.get(name, np.zeros(256, np.float32))
        arrays = {"v2_probabilities": np.asarray(score_v2, dtype=np.float16),
                  "v1_probabilities": np.asarray(score_v1, dtype=np.float16),
                  "role_gt_occupancy": role_arr,
                  "instance_gt_occupancy": instance_arr,
                  "valid_fraction": np.asarray(valid_rows, dtype=np.float16),
                  "state_indices": np.arange(segment["start"], segment["end_exclusive"], dtype=np.int32)}
        cache_meta = {"task_manifest_sha256": manifest_hash, "checkpoint_sha256": checkpoint_hashes,
                      "task_id": task["task_id"], "split": task["split"], "suite": task["suite"],
                      "demo": segment["demo"], "start": segment["start"], "sample_count": SEQUENCE_FRAMES,
                      "consecutive": True, "prompt_specs": prompt_specs, "role_keys": role_keys,
                      "instance_keys": instance_keys, "instance_roles": instance_roles,
                      "bddl_roles": bddl_roles, "bddl_path": str(bddl_path),
                      "camera": {"name": "agentview", "height": image_size, "width": image_size},
                      "geometry": geometry, "step_diagnostics": step_diagnostics,
                      "actions_applied": False}
        save_sequence_cache(cache_path, arrays, cache_meta)
        payload, _ = load_sequence_cache(cache_path, manifest_hash, checkpoint_hashes)
        gallery = {"rgb": frame_rgb_for_gallery, "labels": None, "task": task,
                   "prompt_specs": prompt_specs, "scores_v2": arrays["v2_probabilities"][0],
                   "scores_v1": arrays["v1_probabilities"][0], "role_gt": arrays["role_gt_occupancy"][0],
                   "role_keys": role_keys, "cache_meta": cache_meta}
        return payload, gallery
    finally:
        env.close()


def collect_metrics(cache_rows: list[tuple[dict,dict]], model_key: str, threshold: float) -> dict:
    stat = defaultdict(lambda: np.zeros(4, dtype=np.float64))
    pair_ious = defaultdict(list); visible_frames = Counter(); empty_visible = Counter()
    fg_count, coverage, rel_total = [], [], []
    task_rows = defaultdict(lambda: defaultdict(lambda: np.zeros(4, dtype=np.float64)))
    instance_small = defaultdict(lambda: np.zeros(4, dtype=np.float64))
    role_counts = Counter(); r5_rows = []
    for arrays, meta in cache_rows:
        probs = arrays["v2_probabilities"] if model_key == "v2" else arrays["v1_probabilities"]
        role_keys = meta["role_keys"]; role_gt = arrays["role_gt_occupancy"].astype(np.float32)
        valid = arrays["valid_fraction"].astype(np.float32)
        role_pred_idxs = defaultdict(list)
        for i, prompt in enumerate(meta["prompt_specs"]):
            role_pred_idxs[prompt["role"]].append(i)
        frame_union = []
        for ti in range(probs.shape[0]):
            v = valid[ti]
            role_map = {r: j for j,r in enumerate(role_keys)}
            # Associate text-only prompt roles with simulator-only GT roles.
            # Destination phrases can name either an object receptacle or a
            # fixture; classify that only after scoring using BDDL type names.
            aliases = {"manipulated_object": "manipulated_objects",
                       "robot_gripper": "robot_gripper", "robot_arm": "robot_arm"}
            predicted_roles = {}
            for prompt_role, gt_role in aliases.items():
                idx = role_pred_idxs.get(prompt_role, [])
                predicted_roles[gt_role] = get_union_mask(probs[ti], threshold, idx)
            target_idxs = []
            destination_object_idxs = []
            fixture_idxs = list(role_pred_idxs.get("task_fixture", []))
            bddl_roles = meta.get("bddl_roles", {})
            for i, prompt in enumerate(meta["prompt_specs"]):
                role = prompt["role"]
                if role == "task_fixture":
                    target_idxs.append(i)
                elif role == "destination_or_fixture":
                    target_idxs.append(i)
                    kind = target_prompt_kind(prompt["phrase"], bddl_roles)
                    if kind == "object": destination_object_idxs.append(i)
                    elif kind == "fixture": fixture_idxs.append(i)
            predicted_roles["destination_or_fixture"] = get_union_mask(probs[ti], threshold, target_idxs)
            predicted_roles["destination_objects"] = get_union_mask(probs[ti], threshold, destination_object_idxs)
            predicted_roles["task_relevant_fixtures"] = get_union_mask(probs[ti], threshold, fixture_idxs)
            for gt_role in ("manipulated_objects", "destination_objects", "task_relevant_fixtures",
                            "robot_arm", "robot_gripper", "robot_mount", "task_relevant_objects"):
                if gt_role not in role_map: continue
                p = predicted_roles.get(gt_role)
                if gt_role == "task_relevant_objects":
                    p = np.logical_or(predicted_roles.get("manipulated_objects", False),
                                      predicted_roles.get("destination_objects", False))
                if p is None: p = np.zeros(256, dtype=bool)
                gt = role_gt[ti, role_map[gt_role]]
                vals = metric_counts(p, gt, v)
                stat[gt_role] += vals; task_rows[meta["task_id"]][gt_role] += vals
                if gt.sum() > 0:
                    visible_frames[gt_role] += 1
                    iou = vals[0] / max(1e-12, vals[0]+vals[1]+vals[2]); pair_ious[gt_role].append(iou)
                    if not p.any(): empty_visible[gt_role] += 1
            # A language target may be an object (plate/basket) or a fixture
            # (rack/drawer). Score that prompt role against their GT union.
            dest_idx = role_map.get("destination_objects")
            fixture_idx = role_map.get("task_relevant_fixtures")
            if dest_idx is not None or fixture_idx is not None:
                dest_gt = np.zeros(256, np.float32)
                if dest_idx is not None: dest_gt = np.maximum(dest_gt, role_gt[ti, dest_idx])
                if fixture_idx is not None: dest_gt = np.maximum(dest_gt, role_gt[ti, fixture_idx])
                p = predicted_roles.get("destination_or_fixture", np.zeros(256, bool))
                vals = metric_counts(p, dest_gt, v)
                stat["destination_or_fixture"] += vals
                task_rows[meta["task_id"]]["destination_or_fixture"] += vals
                if dest_gt.sum() > 0:
                    visible_frames["destination_or_fixture"] += 1
                    pair_ious["destination_or_fixture"].append(vals[0] / max(1e-12, vals[0]+vals[1]+vals[2]))
                    if not p.any(): empty_visible["destination_or_fixture"] += 1
            all_idxs = list(range(probs.shape[1]))
            pred_union = get_union_mask(probs[ti], threshold, all_idxs)
            rel_gt = np.zeros(256, dtype=np.float32)
            for role in ("manipulated_objects", "destination_objects", "task_relevant_fixtures",
                         "robot_arm", "robot_gripper", "robot_mount"):
                if role in role_map: rel_gt = np.maximum(rel_gt, role_gt[ti, role_map[role]])
            vals = metric_counts(pred_union, rel_gt, v)
            stat["task_relevant_foreground"] += vals
            task_rows[meta["task_id"]]["task_relevant_foreground"] += vals
            if rel_gt.sum() > 0:
                visible_frames["task_relevant_foreground"] += 1
                pair_ious["task_relevant_foreground"].append(
                    vals[0] / max(1e-12, vals[0] + vals[1] + vals[2])
                )
                if not pred_union.any():
                    empty_visible["task_relevant_foreground"] += 1
            frame_union.append(pred_union)
            fg_count.append(int(pred_union.sum()))
            coverage.append(float(((~pred_union) * v).sum() / max(1e-12, v.sum())))
            rel_total.append(float(rel_gt.sum()))
            # Small-object diagnostic: binary union prediction against each
            # visible relevant instance with <4 fractional patch equivalents.
            # This evaluates foreground coverage, not instance identity.
            instance_roles = meta.get("instance_roles", {})
            for instance_name, occ in zip(meta.get("instance_keys", []), arrays["instance_gt_occupancy"][ti]):
                inst_role = instance_roles.get(instance_name, "unresolved_instance")
                if inst_role not in {"manipulated_objects", "destination_objects", "task_relevant_fixtures",
                                     "robot_arm", "robot_gripper", "robot_mount"}:
                    continue
                occupancy = np.asarray(occ, dtype=np.float32)
                area = float((occupancy * v).sum())
                if 0 < area < 4.0:
                    vals = metric_counts(pred_union, occupancy, v)
                    instance_small["small_relevant_instance"] += vals
                    role_counts["small_relevant_instance"] += 1
        # Source/current ground-truth and predicted mask turnover at R5 ages.
        # One immutable source at the first state of each 5-frame window.
        for anchor in range(0, probs.shape[0], 5):
            for age in range(1, min(5, probs.shape[0]-anchor)):
                cur = anchor + age
                rmap = {r:j for j,r in enumerate(role_keys)}
                if "task_relevant_foreground" in rmap:
                    source_gt = role_gt[anchor, rmap["task_relevant_foreground"]] > 0
                    current_gt = role_gt[cur, rmap["task_relevant_foreground"]] > 0
                else:
                    source_gt = np.zeros(256,bool);current_gt=np.zeros(256,bool)
                    for role in ("manipulated_objects","destination_objects","task_relevant_fixtures","robot_arm","robot_gripper","robot_mount"):
                        if role in rmap:
                            source_gt |= role_gt[anchor,rmap[role]]>0;current_gt |= role_gt[cur,rmap[role]]>0
                # Scores are evaluated at current frame; the mask is therefore
                # a current-frame policy candidate compared with anchor GT.
                current_pred = get_union_mask(probs[cur], threshold)
                stale = source_gt & ~current_gt
                newly_relevant = current_gt & ~source_gt
                yield_row = {"task_id":meta["task_id"],"demo":meta["demo"],"anchor_state":int(arrays["state_indices"][anchor]),
                             "current_state":int(arrays["state_indices"][cur]),"source_age":age,
                             "source_current_gt_iou":float((source_gt & current_gt).sum()/max(1,(source_gt | current_gt).sum())),
                             "source_fg_current_bg_tokens":int(stale.sum()),
                             "current_fg_source_bg_tokens":int(newly_relevant.sum()),
                             "current_gt_fg_not_pred_fg":int((current_gt & ~current_pred).sum()),
                             "candidate_reused_tokens":int((~current_pred).sum()),
                             "reused_tokens_that_are_current_gt_fg":int((~current_pred & current_gt).sum()),
                             "stale_source_fg_among_reused":int((~current_pred & stale).sum()),
                             "interpretation":"offline mask/GT proxy only; no K/V cache equivalence or safety claim"}
                r5_rows.append(yield_row)
    result = {"model": model_key, "threshold": float(threshold), "role_metrics": {},
              "per_task_metrics": {}, "foreground": {"mean_predicted_fg_tokens": float(np.mean(fg_count)) if fg_count else None,
                "mean_reusable_tokens": float(np.mean([256-x for x in fg_count])) if fg_count else None,
                "mean_candidate_background_coverage": float(np.mean(coverage)) if coverage else None,
                "mean_task_relevant_gt_patch_equivalents": float(np.mean(rel_total)) if rel_total else None}}
    for role, vals in stat.items():
        if role.startswith("_"): continue
        tp,fp,fn,tn=map(float,vals)
        result["role_metrics"][role]={"pooled_iou":tp/max(1e-12,tp+fp+fn),"precision":tp/max(1e-12,tp+fp),
             "recall":tp/max(1e-12,tp+fn),"false_background_rate":fn/max(1e-12,tp+fn),
             "positive_frame_mean_iou":float(np.mean(pair_ious[role])) if pair_ious[role] else None,
             "empty_mask_frequency_when_visible":empty_visible[role]/max(1,visible_frames[role]),
             "visible_role_frames":int(visible_frames[role]),"tp_fp_fn_tn":vals.tolist()}
    if role_counts["small_relevant_instance"]:
        tp,fp,fn,tn=map(float,instance_small["small_relevant_instance"])
        result["small_object_diagnostic"]={"definition":"visible relevant instances with <4 fractional 16x16 patch-equivalents","instance_count":role_counts["small_relevant_instance"],
            "union_foreground_iou":tp/max(1e-12,tp+fp+fn),"precision":tp/max(1e-12,tp+fp),
            "recall":tp/max(1e-12,tp+fn),"interpretation":"overall predicted foreground union coverage; not per-instance category matching"}
    else:
        result["small_object_diagnostic"]={"definition":"visible relevant instances with <4 fractional 16x16 patch-equivalents","instance_count":0,"union_foreground_iou":None,"precision":None,"recall":None}
    for task_id, roles in task_rows.items():
        result["per_task_metrics"][task_id]={}
        for role, vals in roles.items():
            if role.startswith("_"): continue
            tp,fp,fn,tn=map(float,vals)
            result["per_task_metrics"][task_id][role]={"iou":tp/max(1e-12,tp+fp+fn),"precision":tp/max(1e-12,tp+fp),
                "recall":tp/max(1e-12,tp+fn),"false_background_rate":fn/max(1e-12,tp+fn),"tp_fp_fn_tn":vals.tolist()}
    result["r5_temporal_proxy"]=r5_rows
    return result


def aggregate_arrays(sequence_cache_paths: list[Path], split: str, manifest_hash: str) -> list[tuple[dict,dict]]:
    rows=[]
    for path in sequence_cache_paths:
        with np.load(path,allow_pickle=False) as z:
            meta=json.loads(str(z["metadata"].item()))
            if meta.get("task_manifest_sha256") != manifest_hash: raise RuntimeError(f"Wrong manifest in {path}")
            if meta["split"] != split: continue
            arrays={k:np.array(z[k]) for k in z.files if k!="metadata"}
        rows.append((arrays,meta))
    return rows


def calibrate_dev(dev_rows: list[tuple[dict,dict]]) -> dict:
    best=None
    for threshold in np.round(np.linspace(0.0,1.0,101),2):
        tp=fn=fg=valid_total=0.0
        for arrays,meta in dev_rows:
            scores=arrays["v2_probabilities"].astype(np.float32)
            gt=arrays["role_gt_occupancy"].astype(np.float32)
            role_keys=meta["role_keys"]
            role_map={r:i for i,r in enumerate(role_keys)}
            valid=arrays["valid_fraction"].astype(np.float32)
            for ti in range(scores.shape[0]):
                pred=get_union_mask(scores[ti],float(threshold))
                target=np.zeros(256,np.float32)
                for r in ("manipulated_objects","destination_objects","task_relevant_fixtures","robot_arm","robot_gripper","robot_mount"):
                    if r in role_map: target=np.maximum(target,gt[ti,role_map[r]])
            a,b,c,_=metric_counts(pred,target,valid[ti]);tp+=a;fn+=c
            fg+=float((pred.astype(np.float32)*valid[ti]).sum());valid_total+=valid[ti].sum()
        recall=tp/max(1e-12,tp+fn);reuse=1.0-fg/max(1.0,valid_total)
        if recall >= .98:
            candidate={"threshold":float(threshold),"development_task_relevant_recall":float(recall),
                       "development_candidate_background_coverage":float(reuse),"development_predicted_foreground_fraction":float(fg/max(1,valid_total)),
                       "selection_rule":"largest threshold on 0.01 grid achieving >=0.98 weighted patch-occupancy recall on development tasks; task-level split frozen before inference"}
            if best is None or candidate["threshold"]>best["threshold"]:best=candidate
    if best is None:
        raise RuntimeError("98% development recall target was not attained even at zero threshold")
    return best


def temporal_csv_rows(results: list[dict]) -> list[dict]:
    rows=[]
    for result in results:
        rows.extend(result.get("r5_temporal_proxy",[]))
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows: path.write_text(""); return
    keys=[]
    for row in rows:
        for k in row:
            if k not in keys: keys.append(k)
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def generate_report(out: Path, manifest: dict, dev_metrics: dict, heldout_metrics: dict,
                    threshold_selection: dict, source_revisions: dict, logger) -> None:
    task_rows=[]
    role_rows=[]
    for split, metrics in (("development",dev_metrics),("held_out",heldout_metrics)):
        for model, result in metrics.items():
            fg=result["role_metrics"].get("task_relevant_foreground",{})
            task_rows.append(f"| {split} | {model} | {result['threshold']:.2f} | {fg.get('recall',0):.4f} | {fg.get('false_background_rate',0):.4f} | {result['foreground']['mean_reusable_tokens']:.1f} | {result['foreground']['mean_candidate_background_coverage']:.3f} |")
            for role, m in result.get("role_metrics", {}).items():
                role_rows.append(f"| {split} | {model} | {role} | {m.get('visible_role_frames',0)} | {m.get('recall',0):.4f} | {m.get('false_background_rate',0):.4f} | {m.get('pooled_iou',0):.4f} | {m.get('empty_mask_frequency_when_visible',0):.4f} |")
            small=result.get("small_object_diagnostic",{})
            role_rows.append(f"| {split} | {model} | small relevant instances (<4 patch-eq; union-mask diagnostic) | {small.get('instance_count',0)} | {small.get('recall') if small.get('recall') is not None else '—'} | — | {small.get('union_foreground_iou') if small.get('union_foreground_iou') is not None else '—'} | — |")
    report = [
        "# LIBERO Ground-Truth Evaluation — Frozen Talk2DINO-SigLIP V2",
        "",
        "## Outcome",
        "",
        "This is an offline replay of official demonstration simulator states. No recorded actions were applied, no OpenVLA policy rollouts were run, no model was trained, and no KV-cache code was changed.",
        "",
        "The development/held-out split is task-level, balanced within each suite and frozen in `libero_task_manifest.json` before prediction. The V2 threshold 0.20 is the frozen COCO development setting; the separate 98%-recall operating point is calibrated only on development tasks and then evaluated on held-out tasks.",
        "",
        "## Main results",
        "",
        "| Split | Model | Threshold | Task-FG recall | False-background rate | Mean candidate reused tokens / 256 | Mean background coverage |",
        "|---|---|---:|---:|---:|---:|---:|",
        *task_rows,
        "",
        "## Per-role foreground metrics",
        "",
        "Counts are fractional 16×16 patch occupancy, evaluated only over pixels/paches mapped to known instances. Empty-mask frequency is among frames where that role is visible. `destination_or_fixture` is an additional combined target-role diagnostic; BDDL metadata is used only to associate that text role with object-vs-fixture ground truth after scoring.",
        "",
        "| Split | Model | Role | Visible frames / small instance-frames | Recall | False-background rate | Pooled IoU | Empty-mask frequency |",
        "|---|---|---|---:|---:|---:|---:|---:|",
        *role_rows,
        "",
        f"Development-only V2 conservative point: threshold **{threshold_selection['threshold']:.2f}**, task-relevant foreground recall **{threshold_selection['development_task_relevant_recall']:.4f}**, candidate background coverage **{threshold_selection['development_candidate_background_coverage']:.4f}**. See `libero_ground_truth_metrics.json` for held-out metrics at this frozen threshold.",
        "",
        "## Data and replay validation",
        "",
        f"- Official source: `{REPO_ID}`, revision `{manifest['dataset_revision']}`.",
        f"- Selected tasks: {len(manifest['tasks'])} across `{', '.join(manifest['suites'])}`; task-level split: {manifest['development_task_count']} development and {manifest['held_out_task_count']} held-out.",
        f"- Consecutive replay states: {manifest['total_sequence_frames']}; selected demonstration sequences: {manifest['total_sequences']}.",
        f"- Actual frame count by split: development {sum(t.get('split') == 'development' for t in manifest['tasks'])} tasks / {manifest['development_sequence_count']} sequences / {manifest['development_frame_count']} frames; held-out {sum(t.get('split') == 'held_out' for t in manifest['tasks'])} tasks / {manifest['held_out_sequence_count']} sequences / {manifest['held_out_frame_count']} frames.",
        "",
        "| Suite | Task | Split | HDF5 bytes | HDF5 SHA-256 | Sequences | Frames |",
        "|---|---|---|---:|---|---:|---:|",
        *[f"| {t['suite']} | {t['task_id'].split('/',1)[1]} | {t['split']} | {t.get('hdf5_size_bytes','—')} | `{t.get('hdf5_sha256','not downloaded')}` | {len(t.get('selected_sequences',[]))} | {sum(s.get('sample_count',0) for s in t.get('selected_sequences',[]))} |" for t in manifest["tasks"]],
        f"- LIBERO source revision: `{source_revisions.get('libero')}`; robosuite revision: `{source_revisions.get('robosuite')}`.",
        "- HDF5 `states` and `actions` were audited, but actions were never applied. Each episode's stored `model_file` was loaded, and the recorded flattened state was installed directly into MuJoCo.",
        "- The RGB observation and instance labels were rendered from the same restored simulator state. Direct RGB sensor parity and raw geom-ID → `instances_to_ids` → rendered-label `index + 1` equality were checked in the two-task smoke test.",
        "- The raw camera output follows the installed OpenGL/OpenCV image convention. The binary instance masks use nearest-neighbor image preprocessing and fractional area pooling to the 16×16 OpenVLA patch grid.",
        "- Separate `Panda*`, `PandaGripper*`, and `RethinkMount*` labels were preserved from the low-level instance sensor. The wrapper's helper that collapses the robot IDs was not used to construct the per-role labels.",
        "",
        "## Role definitions and prompts",
        "",
        "Ground-truth manipulated/destination/fixture identities come from BDDL object/fixture declarations, `obj_of_interest`, and goal predicates. Those simulator identities are scoring labels only. Model prompts are generated only from each HDF5 language instruction using the documented deterministic noun-phrase rules, plus the fixed `robot arm` and `robot gripper` prompts.",
        "",
        "`Panda0`/mounted-Panda instances are arm, `PandaGripper*` instances are gripper, and `RethinkMount*` instances are robot mount. Non-target object instances are other objects; non-target fixtures and world/background are structural background. Unmapped instance labels are marked unresolved and excluded from the valid scoring area rather than assumed background.",
        "",
        "## Model comparisons and limitations",
        "",
        f"V2 epoch-40 checkpoint SHA-256 `{V2_SHA256}` uses penultimate OpenVLA DINO patches and the COCO-trained `a photo of a …` prompt form with threshold 0.20. V1 epoch-50 SHA-256 `{V1_SHA256}` is a control using its published seven-template average and its established final-normalized DINO features with fixed probability threshold 0.55.",
        "",
        "V1 and V2 therefore differ in visual feature layer and training objective as well as the projection checkpoint; the comparison is of the frozen systems, not an isolated training-loss ablation. Category prompts cannot distinguish two same-class instances reliably; per-instance target scores are consequently a stricter grounding diagnostic.",
        "",
        "R5 age-1–4 tables compare simulator foreground unions and predicted current-frame masks only. They are offline mask-safety proxies, not a cache experiment and not proof that reusing K/V is safe. Motionless visible background is not treated as safe by definition.",
        "",
        "The task set is a 12-task selected subset, not the full LIBERO suite. A 98% development recall operating point may leave little reusable coverage; even a favorable mask metric would require a later, separately authorized cache-safety experiment.",
        "",
        "## Artifacts",
        "",
        "- `libero_task_manifest.json`: pinned source revision, task split, file hashes, HDF5 schema, episode ranges, and frame/state hashes.",
        "- `libero_ground_truth_metrics.json`: per-role, per-task, per-model and R5-age summaries.",
        "- `per_task_metrics.csv`, `per_frame_diagnostics.csv`, `temporal_r5_age_metrics.csv`.",
        "- `visualizations/`: deterministic first-frame examples for each selected task.",
        "- `smoke_validation.json` and `logs/`: exact mapping checks and execution record.",
        "",
    ]
    (out / "LIBERO_GROUND_TRUTH_REPORT.md").write_text("\n".join(report))


def run(args) -> None:
    out = args.output_dir
    data_root = args.dataset_root
    cache_root = args.cache_root
    logs = out / "logs"
    out.mkdir(parents=True, exist_ok=True); logs.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    log_path = logs / "worker.log"
    def log(message: str) -> None:
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {message}"
        print(line, flush=True)
        with log_path.open("a") as f: f.write(line + "\n"); f.flush(); os.fsync(f.fileno())

    status_path = out / "RUN_STATUS.json"
    state = {"status":"running","stage":"preflight","started_at_utc":time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
             "host":os.uname().nodename,"pid":os.getpid(),"last_error":None,"completed_tasks":[],"completed_sequences":[]}
    atomic_json(status_path,state)
    try:
        import h5py
        from huggingface_hub import HfApi
        libero_root = args.libero_root
        robosuite_root = args.robosuite_root
        libero_rev = os.popen(f"git -C {libero_root} rev-parse HEAD").read().strip()
        robosuite_rev = os.popen(f"git -C {robosuite_root} rev-parse HEAD").read().strip()
        df = os.statvfs(data_root)
        free_bytes = df.f_bavail * df.f_frsize
        api = HfApi()
        source_info = api.repo_info(REPO_ID, repo_type="dataset", revision=REPO_REVISION, files_metadata=True)
        if source_info.sha != REPO_REVISION:
            raise RuntimeError(f"Pinned LIBERO dataset revision mismatch: {source_info.sha}")
        tasks = expected_tasks()
        for t in tasks:
            bddl = (libero_root / "libero/libero/bddl_files" / t["suite"] /
                    (t["file_name"].removesuffix("_demo.hdf5") + ".bddl"))
            if not bddl.is_file(): raise FileNotFoundError(f"Selected official demo lacks exact installed BDDL task: {bddl}")
            t["bddl_path"] = str(bddl); t["bddl_sha256"] = sha256_file(bddl)
            t["role_ground_truth"] = parse_bddl_roles(bddl)
            t["language_instruction_from_bddl"] = " ".join(re.findall(r"\(:language\s+([^\n]+)", bddl.read_text(), re.I)).strip().rstrip(")")
            add_instruction_named_fixtures(t["role_ground_truth"], t["language_instruction_from_bddl"])
            t["prompt_specs_preview"] = instruction_phrases(t["instruction"] or t["language_instruction_from_bddl"])
        # Verify HDF5 file-size budget from the pinned repo before downloads.
        metadata_sizes = []
        for t in tasks:
            rel = f"{t['suite']}/{t['file_name']}"
            item = api.get_paths_info(REPO_ID,[rel],repo_type="dataset",revision=REPO_REVISION)[0]
            metadata_sizes.append(int(item.size))
        required = int(sum(metadata_sizes) * 1.2 + (2 << 30))
        if free_bytes < required: raise RuntimeError(f"Need ~{required} bytes for selected 12 files + headroom; free={free_bytes}")
        log(f"preflight pinned={REPO_REVISION} selected_files={len(tasks)} estimated_bytes={sum(metadata_sizes)} free_bytes={free_bytes} libero={libero_rev} robosuite={robosuite_rev}")
        state.update({"stage":"download_initial_two","source_revision":REPO_REVISION,"source_repository":REPO_ID,
                      "libero_revision":libero_rev,"robosuite_revision":robosuite_rev,"estimated_selected_bytes":sum(metadata_sizes),
                      "free_disk_bytes_preflight":free_bytes}); atomic_json(status_path,state)

        # Stage 1: acquire only one Spatial and one Object task; validate schema
        # and exact simulator render/mapping before downloading more.
        task_by_key={(t["suite"],t["file_name"]):t for t in tasks}
        initial=[]
        for suite, filename in sorted(INITIAL_TASKS):
            t=task_by_key[(suite,filename)]
            download_and_audit(t,data_root,REPO_REVISION,api,log); initial.append(t)
        atomic_json(out/"initial_two_task_hdf5_audit.json",initial)
        smoke = []
        for task in initial:
            first_demo=task["episode_names"][0]
            rendered=render_one_state(task,first_demo,0,libero_root=libero_root,
                robosuite_root=robosuite_root,image_size=224,validate_mapping=True)
            mapping=rendered["mapping_validation"]
            if not mapping["direct_raw_geom_to_name_mapping_equals_observation"] or not mapping["rgb_direct_render_equals_observation"]:
                raise RuntimeError("Initial simulator alignment/mapping smoke test failed")
            if rendered["state_length"] != rendered["sim_state_length"]:
                raise RuntimeError("Recorded flattened simulator state does not match installed simulator model")
            if not any(v["name"].startswith(("Panda","MountedPanda","OnTheGroundPanda","RethinkMount")) for v in rendered["visible_instances"].values()):
                raise RuntimeError("No robot component is visible in the initial segmentation smoke frame")
            diag=out/"validation"/f"smoke_{task['suite']}_{Path(task['file_name']).stem}.png"
            write_alignment_figure(rendered,task,diag)
            smoke.append({"task_id":task["task_id"],"demo":first_demo,"state_index":0,
                          "state_length":rendered["state_length"],"mapping_validation":mapping,
                          "visible_instances":rendered["visible_instances"],"figure":str(diag),
                          "recorded_actions_applied":False})
            log(f"smoke passed {task['task_id']} instances={len(mapping['unique_rendered_labels'])-1} state_width={rendered['state_length']}")
        atomic_json(out/"smoke_validation.json",{"status":"passed","tasks":smoke,
                   "checks":["state vector length/model compatibility","RGB same-state orientation parity","raw geom-ID to ordered instance name plus one equality","visible robot segmentation","16x16 row-major patch grid"]})

        # Only after the two-task smoke passes, fetch the remaining selected
        # official files. No suite-wide snapshot or unrelated dataset download.
        state.update({"stage":"download_remaining_selected_tasks","initial_smoke":"passed"}); atomic_json(status_path,state)
        for task in tasks:
            if (task["suite"],task["file_name"]) in INITIAL_TASKS: continue
            download_and_audit(task,data_root,REPO_REVISION,api,log)
            state["completed_tasks"].append(task["task_id"]); atomic_json(status_path,state)
        for task in initial:
            state["completed_tasks"].append(task["task_id"])
        state.update({"stage":"freeze_task_split_and_manifest"}); atomic_json(status_path,state)

        # Determine 3 distinct, >=32-state demonstrations per selected task.
        sequences=[]
        for task in tasks:
            with h5py.File(task["hdf5_path"],"r") as f:
                demo_names=task["episode_names"]
                lengths={name:int(f[f"data/{name}/states"].shape[0]) for name in demo_names}
                selected=select_segments(demo_names,lengths)
                task["available_episode_count"]=len(demo_names)
                task["episode_lengths"]=lengths
                task["selected_sequences"]=selected
                for segment in selected:
                    segment["states_slice_sha256"]=sha256_bytes(np.ascontiguousarray(f[f"data/{segment['demo']}/states"][segment["start"]:segment["end_exclusive"]]).tobytes())
                    segment["model_xml_sha256"]=task["hdf5_schema"]["episodes"][demo_names.index(segment["demo"])]["model_xml_sha256"]
                    segment["actions_applied"]=False
                    sequences.append({"task":task,"segment":segment})
        if len({x["task_id"] for x in tasks}) < 12:
            raise RuntimeError("The predeclared task set is incomplete")
        if any(len(t["selected_sequences"]) < 1 for t in tasks):
            raise RuntimeError("At least one selected task has no >=32-state demonstrator sequence")
        if any(any(x["sample_count"] < 32 for x in t["selected_sequences"]) for t in tasks):
            raise RuntimeError("Selected sequence shorter than 32 consecutive states")
        manifest={"schema_version":1,"status":"frozen_before_model_evaluation","dataset_repository":REPO_ID,
            "dataset_revision":REPO_REVISION,"libero_revision":libero_rev,"robosuite_revision":robosuite_rev,
            "libero_source_tree_sha256":sha256_file(libero_root/"libero/libero/envs/env_wrapper.py"),
            "robosuite_source_tree_sha256":sha256_file(robosuite_root/"robosuite/environments/robot_env.py"),
            "camera_name":"agentview","camera_native_resolution":[224,224],"policy_grid":[16,16],
            "state_selection":{"demos_per_task":DEMOS_PER_TASK,"consecutive_states_per_demo":SEQUENCE_FRAMES,
                "selected_demo_rule":"lexically first 3 demos with >=32 states; starts at beginning, midpoint, end respectively"},
            "split_rule":"predeclared task-level 6 development / 6 held-out, balanced within each of Spatial, Object, Goal; no trajectory from a task crosses split",
            "tasks":tasks,"task_count":len(tasks),"suites":sorted({t["suite"] for t in tasks}),
            "development_task_count":sum(t["split"]=="development" for t in tasks),
            "held_out_task_count":sum(t["split"]=="held_out" for t in tasks),
            "development_sequence_count":sum(t["split"]=="development" for t in tasks for _ in t["selected_sequences"]),
            "held_out_sequence_count":sum(t["split"]=="held_out" for t in tasks for _ in t["selected_sequences"]),
            "development_frame_count":sum(s["sample_count"] for t in tasks if t["split"]=="development" for s in t["selected_sequences"]),
            "held_out_frame_count":sum(s["sample_count"] for t in tasks if t["split"]=="held_out" for s in t["selected_sequences"]),
            "total_sequences":len(sequences),"total_sequence_frames":sum(x["segment"]["sample_count"] for x in sequences),
            "prompts_generated_from_task_instruction_only":True,"oracle_identity_used_for_prompt_generation":False,
            "recorded_actions_applied":False}
        manifest_hash=sha256_bytes(json.dumps(manifest,sort_keys=True,separators=(",",":")).encode())
        manifest["manifest_sha256"] = manifest_hash
        atomic_json(out/"libero_task_manifest.json",manifest)
        log(f"frozen manifest={manifest_hash} tasks={len(tasks)} sequences={len(sequences)} frames={manifest['total_sequence_frames']}")

        # Audit prompt generation before any model scoring, then load frozen
        # checkpoints and actual OpenVLA/DINO/SigLIP inference components.
        state.update({"stage":"load_frozen_models","task_manifest_sha256":manifest_hash}); atomic_json(status_path,state)
        models=load_models(args.repo_root,"cuda:0" if __import__("torch").cuda.is_available() else "cpu",log)
        from experiments.kv_cache.experiments.trackb_talk2dino_siglip.text_encoder import (
            encode_text_embeddings, load_siglip_text_tower,
        )
        text_model, tokenizer, text_meta=load_siglip_text_tower(SIGLIP_ROOT,device=models["device"])
        geometry=None
        from experiments.kv_cache.experiments.trackb_talk2dino_siglip.training_v2_dense.masks import geometry_from_transform
        geometry=geometry_from_transform(models["processor"].image_processor)
        if tuple(geometry["crop_size"]) != (224,224): raise RuntimeError(f"Policy DINO crop differs from 224x224: {geometry}")
        atomic_json(out/"model_and_preprocessing.json",{"v1_checkpoint":models["checkpoint_paths"]["v1"],"v1_sha256":V1_SHA256,
            "v1_prompt_templates":list(V1_TEMPLATES),"v1_fixed_threshold":V1_THRESHOLD,
            "v2_checkpoint":models["checkpoint_paths"]["v2"],"v2_sha256":V2_SHA256,
            "v2_prompt_template":"a photo of a {instruction-derived phrase}","v2_fixed_probability_threshold":V2_THRESHOLD,
            "v2_visual_features":"OpenVLA fine-tuned DINO penultimate layer as emitted by dino_featurizer forward",
            "v1_visual_features":"final-normalized OpenVLA DINO DINOv2 patches from get_intermediate_layers(norm=True)",
            "policy_image_processor_geometry":geometry,"text_checkpoint":text_meta,
            "all_backbones_and_projections_frozen":True})

        # Score development sequences only first. Caches are immutable and keyed
        # to the frozen manifest plus checkpoint hashes.
        checkpoint_hashes={"v1":V1_SHA256,"v2":V2_SHA256}
        state.update({"stage":"score_development_tasks","completed_sequences":[]}); atomic_json(status_path,state)
        all_cache_paths=[]; galleries=[]
        for i, pair in enumerate(sequences):
            task,segment=pair["task"],pair["segment"]
            if task["split"] != "development": continue
            prompt_specs=instruction_phrases(task["instruction"] or task["language_instruction_from_bddl"])
            tok2=tokenizer([f"a photo of a {p['phrase']}" for p in prompt_specs]).to(models["device"])
            tok1=tokenizer([template.format(p["phrase"]) for p in prompt_specs for template in V1_TEMPLATES]).to(models["device"])
            with models["torch"].inference_mode():
                txt2=encode_text_embeddings(text_model,tok2).float()
                txt1=encode_text_embeddings(text_model,tok1).float().reshape(len(prompt_specs),len(V1_TEMPLATES),-1).mean(1)
            arrays,gallery=process_sequence(task,segment,models=models,prompt_specs=prompt_specs,text_v2=txt2,text_v1=txt1,
                bddl_roles=task["role_ground_truth"],geometry=geometry,libero_root=libero_root,robosuite_root=robosuite_root,image_size=224,
                run_cache=cache_root,manifest_hash=manifest_hash,checkpoint_hashes=checkpoint_hashes,logger=log)
            cache_path=cache_root/"scoring"/re.sub(r"[^A-Za-z0-9_-]+","_",task["task_id"])/f"{segment['demo']}_s{segment['start']:05d}.npz"
            all_cache_paths.append(cache_path)
            if gallery is not None: galleries.append(gallery)
            task["scoring_done_sequences"] = task.get("scoring_done_sequences",0)+1
            state["completed_sequences"].append({"task_id":task["task_id"],"demo":segment["demo"],"split":"development"})
            atomic_json(status_path,state)
        dev_rows=aggregate_arrays(all_cache_paths,"development",manifest_hash)
        # Existing verified caches may contain already-completed development
        # segments from a resumed run; include all deterministic dev caches.
        dev_paths=list((cache_root/"scoring").glob("*/*.npz"))
        dev_rows=aggregate_arrays(dev_paths,"development",manifest_hash)
        if not dev_rows: raise RuntimeError("No development sequence scores are available")
        calibration=calibrate_dev(dev_rows)
        atomic_json(out/"development_calibration.json",{"status":"frozen_before_heldout_evaluation",
            "manifest_sha256":manifest_hash,"v2_threshold_0_20_metrics":collect_metrics(dev_rows,"v2",V2_THRESHOLD),
            "v1_threshold_0_55_metrics":collect_metrics(dev_rows,"v1",V1_THRESHOLD),"v2_98_percent_recall_point":calibration,
            "calibration_scope":"development tasks only"})
        state.update({"stage":"development_calibrated","development_calibration":calibration,
                      "heldout_scoring_started":False}); atomic_json(status_path,state)
        log(f"development threshold frozen={calibration['threshold']:.2f} recall={calibration['development_task_relevant_recall']:.4f} candidate_bg={calibration['development_candidate_background_coverage']:.4f}")

        # Held-out evaluation starts only after the calibration artifact has
        # been atomically written. No re-selection uses these results.
        state.update({"stage":"score_heldout_tasks","heldout_scoring_started":True}); atomic_json(status_path,state)
        for pair in sequences:
            task,segment=pair["task"],pair["segment"]
            if task["split"] != "held_out": continue
            prompt_specs=instruction_phrases(task["instruction"] or task["language_instruction_from_bddl"])
            tok2=tokenizer([f"a photo of a {p['phrase']}" for p in prompt_specs]).to(models["device"])
            tok1=tokenizer([template.format(p["phrase"]) for p in prompt_specs for template in V1_TEMPLATES]).to(models["device"])
            with models["torch"].inference_mode():
                txt2=encode_text_embeddings(text_model,tok2).float()
                txt1=encode_text_embeddings(text_model,tok1).float().reshape(len(prompt_specs),len(V1_TEMPLATES),-1).mean(1)
            arrays,gallery=process_sequence(task,segment,models=models,prompt_specs=prompt_specs,text_v2=txt2,text_v1=txt1,
                bddl_roles=task["role_ground_truth"],geometry=geometry,libero_root=libero_root,robosuite_root=robosuite_root,image_size=224,
                run_cache=cache_root,manifest_hash=manifest_hash,checkpoint_hashes=checkpoint_hashes,logger=log)
            cache_path=cache_root/"scoring"/re.sub(r"[^A-Za-z0-9_-]+","_",task["task_id"])/f"{segment['demo']}_s{segment['start']:05d}.npz"
            all_cache_paths.append(cache_path)
            if gallery is not None: galleries.append(gallery)
            state["completed_sequences"].append({"task_id":task["task_id"],"demo":segment["demo"],"split":"held_out"})
            atomic_json(status_path,state)
        held_rows=aggregate_arrays(list((cache_root/"scoring").glob("*/*.npz")),"held_out",manifest_hash)
        if not held_rows: raise RuntimeError("No held-out sequence scores are available")
        model_names=("v1","v2")
        dev_metrics={"V1_epoch50_seven_templates":collect_metrics(dev_rows,"v1",V1_THRESHOLD),
                     "V2_epoch40_COCO_prompt_0_20":collect_metrics(dev_rows,"v2",V2_THRESHOLD),
                     "V2_epoch40_dev_98pct_recall":collect_metrics(dev_rows,"v2",calibration["threshold"])}
        held_metrics={"V1_epoch50_seven_templates":collect_metrics(held_rows,"v1",V1_THRESHOLD),
                      "V2_epoch40_COCO_prompt_0_20":collect_metrics(held_rows,"v2",V2_THRESHOLD),
                      "V2_epoch40_dev_98pct_recall":collect_metrics(held_rows,"v2",calibration["threshold"])}
        atomic_json(out/"libero_ground_truth_metrics.json",{"status":"complete","manifest_sha256":manifest_hash,
            "development_task_metrics":dev_metrics,"held_out_task_metrics":held_metrics,
            "threshold_calibration":calibration,"threshold_calibration_used_only_development":True,
            "model_weights_modified":False,"policy_rollouts":False,"recorded_actions_applied":False,
            "kv_cache_changed":False,"r5_is_offline_proxy_only":True})
        write_csv(out/"temporal_r5_age_metrics.csv",temporal_csv_rows([dev_metrics["V2_epoch40_COCO_prompt_0_20"],held_metrics["V2_epoch40_COCO_prompt_0_20"]]))
        task_csv=[]
        for split, metrics in (("development",dev_metrics),("held_out",held_metrics)):
            for model_name, result in metrics.items():
                for task_id, roles in result["per_task_metrics"].items():
                    for role, val in roles.items(): task_csv.append({"split":split,"model":model_name,"task_id":task_id,"role":role,**val})
        write_csv(out/"per_task_metrics.csv",task_csv)
        frame_csv=[]
        for rows in (dev_rows,held_rows):
            for arrays,meta in rows:
                p2=arrays["v2_probabilities"].astype(np.float32); p1=arrays["v1_probabilities"].astype(np.float32)
                for ti in range(p2.shape[0]):
                    pred2=get_union_mask(p2[ti],V2_THRESHOLD);pred1=get_union_mask(p1[ti],V1_THRESHOLD)
                    roles={r:i for i,r in enumerate(meta["role_keys"])};gt=np.zeros(256,np.float32)
                    for role in ("manipulated_objects","destination_objects","task_relevant_fixtures","robot_arm","robot_gripper","robot_mount"):
                        if role in roles:gt=np.maximum(gt,arrays["role_gt_occupancy"][ti,roles[role]])
                    frame_csv.append({"split":meta["split"],"task_id":meta["task_id"],"demo":meta["demo"],
                        "state_index":int(arrays["state_indices"][ti]),"v2_fg_tokens_at_0_20":int(pred2.sum()),
                        "v2_candidate_bg_tokens_at_0_20":int((~pred2).sum()),"v1_fg_tokens_at_0_55":int(pred1.sum()),
                        "gt_task_relevant_patch_equivalents":float(gt.sum()),"v2_missed_gt_occupancy":float((gt*(~pred2)).sum()),
                        "v1_missed_gt_occupancy":float((gt*(~pred1)).sum()),"source_age_score":"current-frame rendering"})
        write_csv(out/"per_frame_diagnostics.csv",frame_csv)
        # Deterministic fixed sample selection: first selected frame of each task,
        # not best/worst model outcome. No use for threshold/checkpoint selection.
        visdir=out/"visualizations"; visdir.mkdir(exist_ok=True)
        render_deterministic_gallery(galleries,visdir,geometry)
        generate_report(out,manifest,dev_metrics,held_metrics,calibration,
                        {"libero":libero_rev,"robosuite":robosuite_rev},log)
        state.update({"status":"complete","stage":None,"completed_at_utc":time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                      "development_sequences":len(dev_rows),"heldout_sequences":len(held_rows),
                      "development_frames":sum(x[0]["v2_probabilities"].shape[0] for x in dev_rows),
                      "heldout_frames":sum(x[0]["v2_probabilities"].shape[0] for x in held_rows),
                      "last_error":None,"training_or_policy_rollouts":False}); atomic_json(status_path,state)
        log("E2 LIBERO ground-truth evaluation completed")
    except Exception as exc:
        state.update({"status":"failed","stage":state.get("stage"),"failed_at_utc":time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                      "last_error":str(exc),"traceback":traceback.format_exc(),
                      "recovery":"rerun the same command; verified HDF5 files and sequence caches are skipped only when hashes match"})
        atomic_json(status_path,state); log(f"FATAL {exc}\n{traceback.format_exc()}")
        raise


def render_deterministic_gallery(galleries: list[dict], output: Path, geometry: dict) -> None:
    """Render fixed first-frame examples; scores do not select images."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from experiments.kv_cache.experiments.trackb_talk2dino_siglip.training_v2_dense.masks import apply_mask_geometry
    # The compact gallery is generated by re-rendering the first demo frame of
    # each task from its state, so no frame is selected by performance.
    for item in galleries:
        if item is None: continue
        task=item["task"]; rgb=item["rgb"]
        # role grid labels are available at 16x16, upsample nearest for display.
        role_keys=item["role_keys"]; gt=item["role_gt"]
        relevant=np.zeros(256,np.float32)
        for role in ("manipulated_objects","destination_objects","task_relevant_fixtures","robot_arm","robot_gripper","robot_mount"):
            if role in role_keys: relevant=np.maximum(relevant,gt[role_keys.index(role)])
        v2=get_union_mask(item["scores_v2"],V2_THRESHOLD)
        v1=get_union_mask(item["scores_v1"],V1_THRESHOLD)
        img224=np.asarray(Image.fromarray(rgb).resize((224,224),Image.Resampling.BICUBIC))
        panel=lambda mask: np.asarray(mask,dtype=np.uint8).reshape(16,16).repeat(14,0).repeat(14,1)
        def transparent(mask):
            value = panel(mask).astype(np.float32)
            return np.ma.masked_where(value <= 0, value)
        magma = plt.get_cmap("magma").copy(); magma.set_bad(alpha=0)
        cool = plt.get_cmap("cool").copy(); cool.set_bad(alpha=0)
        fig,ax=plt.subplots(1,5,figsize=(17,3.6))
        ax[0].imshow(rgb);ax[0].set_title("Same-state RGB")
        ax[1].imshow(panel(relevant),cmap="gray",vmin=0,vmax=1);ax[1].set_title("Oracle task-FG at 16×16")
        ax[2].imshow(img224);ax[2].imshow(transparent(v2),alpha=.70,cmap=magma,vmin=0,vmax=1,interpolation="nearest");ax[2].set_title("V2 raw, COCO τ=.20")
        ax[3].imshow(img224);ax[3].imshow(transparent(v1),alpha=.70,cmap=magma,vmin=0,vmax=1,interpolation="nearest");ax[3].set_title("V1 7-template τ=.55")
        ax[4].imshow(img224);ax[4].imshow(transparent(v2),alpha=.72,cmap=cool,vmin=0,vmax=1,interpolation="nearest")
        ax[4].set_xticks(np.arange(0,225,14));ax[4].set_yticks(np.arange(0,225,14));ax[4].grid(color="white",alpha=.35,linewidth=.4)
        ax[4].set_title(f"V2 foreground cells {int(v2.sum())}/256")
        for a in ax[:4]:a.axis("off")
        ax[4].set_xlim(0,224);ax[4].set_ylim(224,0)
        fig.suptitle(f"{task['task_id']} · {task['split']} · deterministic first frame",fontsize=10)
        fig.tight_layout()
        filename=re.sub(r"[^A-Za-z0-9_-]+","_",task["task_id"])+".png"
        fig.savefig(output/filename,dpi=110);plt.close(fig)


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--repo-root",type=Path,required=True)
    p.add_argument("--libero-root",type=Path,default=Path("/opt/LIBERO"))
    p.add_argument("--robosuite-root",type=Path,default=Path("/opt/robosuite"))
    p.add_argument("--dataset-root",type=Path,required=True)
    p.add_argument("--cache-root",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    run(p.parse_args())


if __name__ == "__main__": main()
