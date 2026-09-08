import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


TASK_SUITE_NAME = "libero_spatial"
TASK_ID = 0
INITIAL_STATE_ID = 0

RESOLUTION = 256

KV_ANALYSIS_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = KV_ANALYSIS_DIR / "inputs"
DEFAULT_STEP_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
DEFAULT_MODEL_ID = "openvla/openvla-7b-finetuned-libero-spatial"


def create_libero_env(task, resolution=256):
    """
    Minimal version of OpenVLA's get_libero_env().

    Avoids importing experiments.robot.libero.libero_utils,
    because that pulls in the full OpenVLA / Prismatic stack.
    """
    task_description = task.language

    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"),
        task.problem_folder,
        task.bddl_file,
    )

    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=resolution,
        camera_widths=resolution,
    )

    # OpenVLA's official LIBERO evaluation uses seed 0 here.
    env.seed(0)

    return env, task_description


def extract_libero_image(obs):
    """
    Extract the agent-view RGB image.

    OpenVLA rotates the LIBERO image by 180 degrees
    before passing it to the model.
    """
    img = obs["agentview_image"]

    # Same orientation correction used by OpenVLA.
    img = img[::-1, ::-1]

    return img


def build_task_and_env():
    """Create the deterministic LIBERO spatial/task-0/initial-state-0 setup."""
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[TASK_SUITE_NAME]()
    task = task_suite.get_task(TASK_ID)
    initial_states = task_suite.get_task_init_states(TASK_ID)
    env, task_description = create_libero_env(task, resolution=RESOLUTION)
    return task_suite, task, initial_states, env, task_description


def save_image(image: np.ndarray, path: Path) -> None:
    Image.fromarray(image).save(path)


def prepare_openvla_policy_image(image: np.ndarray, center_crop: bool) -> np.ndarray:
    """Apply the evaluator's optional OpenVLA center crop before persisting a frame.

    Persisting this final processor input means an offline cache experiment can
    consume the PNG directly without silently using a different image than the
    rollout policy used for its action.
    """
    if not center_crop:
        return image

    import tensorflow as tf

    tensor = tf.convert_to_tensor(image)
    original_dtype = tensor.dtype
    tensor = tf.image.convert_image_dtype(tensor, tf.float32)
    # Same 0.9-area centered crop used by openvla_utils.crop_and_resize().
    crop_extent = tf.sqrt(tf.constant(0.9, dtype=tf.float32))
    boxes = tf.reshape(
        tf.stack(
            [
                (1 - crop_extent) / 2,
                (1 - crop_extent) / 2,
                (1 + crop_extent) / 2,
                (1 + crop_extent) / 2,
            ]
        ),
        (1, 4),
    )
    tensor = tf.image.crop_and_resize(
        tf.expand_dims(tensor, axis=0), boxes, box_indices=[0], crop_size=(224, 224)
    )[0]
    tensor = tf.clip_by_value(tensor, 0, 1)
    return tf.image.convert_image_dtype(tensor, original_dtype, saturate=True).numpy()


def run_fixed_pair(args) -> None:
    """Original lightweight behavior: one fixed transition, no model load."""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    previous_path = args.output_dir / args.previous_name
    current_path = args.output_dir / args.current_name
    task_suite, _task, initial_states, env, task_description = build_task_and_env()
    print(f"Task suite: {TASK_SUITE_NAME}; tasks: {task_suite.n_tasks}")
    print(f"Task description: {task_description}")

    try:
        env.reset()
        obs = env.set_init_state(initial_states[INITIAL_STATE_ID])
        save_image(extract_libero_image(obs), previous_path)
        obs, _reward, _done, _info = env.step(args.step_action)
        save_image(extract_libero_image(obs), current_path)
        print(f"Saved previous observation: {previous_path.resolve()}")
        print(f"Saved current observation: {current_path.resolve()}")
        print("Intervening LIBERO action:", args.step_action)
    finally:
        env.close()


def load_rollout_policy(model_id: str):
    """Load the HF-exported OpenVLA policy without optional DROID dependencies."""
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    unnorm_key = TASK_SUITE_NAME
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"
    if unnorm_key not in model.norm_stats:
        raise KeyError(f"Action un-normalization key {unnorm_key!r} is absent from model.norm_stats.")
    return model, processor, unnorm_key, device


def predict_openvla_action(model, processor, image: np.ndarray, task_description: str, unnorm_key: str, device) -> np.ndarray:
    """The official OpenVLA prompt/processor/greedy inference path, dependency-minimized."""
    import torch

    prompt = f"In: What action should the robot take to {task_description.lower()}?\nOut:"
    inputs = processor(prompt, Image.fromarray(image).convert("RGB")).to(device, dtype=torch.bfloat16)
    return model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)


def normalize_and_invert_openvla_gripper(action: np.ndarray) -> np.ndarray:
    """Exact LIBERO/OpenVLA environment conversion from the official evaluator."""
    environment_action = np.asarray(action).copy()
    environment_action[..., -1] = 2 * environment_action[..., -1] - 1
    environment_action[..., -1] = np.sign(environment_action[..., -1])
    environment_action[..., -1] *= -1.0
    return environment_action


def run_rollout(args) -> None:
    """Save one continuous, official-style OpenVLA LIBERO rollout.

    Frame ``step_000.png`` is state s_0. Its ``action`` metadata entry is
    action_000, the exact environment action applied from s_0 to s_1, which is
    stored as ``step_001.png``. The final saved frame has ``action: null``.
    """
    # Local imports avoid the optional DROID/TensorFlow Graphics dependencies in
    # the broad robot-evaluation utilities.
    import random
    import tensorflow as tf
    import torch

    if args.num_steps < 2:
        raise ValueError("--num-steps must be at least 2 to create an adjacent frame pair.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    task_suite, _task, initial_states, env, task_description = build_task_and_env()
    model, processor, unnorm_key, device = load_rollout_policy(args.model_id)
    metadata = {
        "task_suite": TASK_SUITE_NAME,
        "task_id": TASK_ID,
        "task_description": task_description,
        "initial_state_id": INITIAL_STATE_ID,
        "model_id": args.model_id,
        "seed": args.seed,
        "settling_steps": args.num_steps_wait,
        "center_crop": args.center_crop,
        "frame_action_indexing": (
            "frames[i] is observation s_i. frames[i].action is action_i applied from s_i to s_(i+1). "
            "The final frame has action null."
        ),
        "frames": [],
        "done": False,
        "success": False,
    }

    try:
        env.reset()
        obs = env.set_init_state(initial_states[INITIAL_STATE_ID])

        # Match the evaluator: wait for dropped objects to settle before control frame 0.
        for _ in range(args.num_steps_wait):
            obs, _reward, done, _info = env.step(DEFAULT_STEP_ACTION)
            if done:
                metadata["done"] = True
                metadata["success"] = True
                break

        # Save exactly N observations (or fewer on termination) and N-1 connecting actions.
        for step in range(args.num_steps):
            # Official evaluation: rotate 180°, JPEG round-trip, Lanczos resize,
            # then optional 0.9-area center crop. Persist the final model image.
            # Same get_libero_image(): rotate 180°, JPEG round-trip, Lanczos resize to 224px.
            image = tf.image.encode_jpeg(extract_libero_image(obs))
            image = tf.io.decode_image(image, expand_animations=False, dtype=tf.uint8)
            image = tf.image.resize(image, (224, 224), method="lanczos3", antialias=True)
            image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8).numpy()
            policy_image = prepare_openvla_policy_image(image, center_crop=args.center_crop)
            filename = f"step_{step:03d}.png"
            save_image(policy_image, args.output_dir / filename)
            metadata["frames"].append({"step": step, "image": filename, "action": None})

            if step + 1 == args.num_steps or metadata["done"]:
                break

            policy_action = predict_openvla_action(
                model, processor, policy_image, task_description, unnorm_key, device
            )
            env_action = normalize_and_invert_openvla_gripper(policy_action)
            metadata["frames"][-1]["action"] = env_action.tolist()
            metadata["frames"][-1]["policy_action_before_gripper_conversion"] = np.asarray(policy_action).tolist()

            obs, _reward, done, _info = env.step(env_action.tolist())
            metadata["done"] = bool(done)
            metadata["success"] = bool(done)  # This is the evaluator's success criterion.
            if done:
                # The loop's next iteration writes the terminal successor state,
                # preserving the action_i -> frame_(i+1) relationship.
                if step + 1 < args.num_steps:
                    image = tf.image.encode_jpeg(extract_libero_image(obs))
                    image = tf.io.decode_image(image, expand_animations=False, dtype=tf.uint8)
                    image = tf.image.resize(image, (224, 224), method="lanczos3", antialias=True)
                    image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8).numpy()
                    policy_image = prepare_openvla_policy_image(image, center_crop=args.center_crop)
                    filename = f"step_{step + 1:03d}.png"
                    save_image(policy_image, args.output_dir / filename)
                    metadata["frames"].append({"step": step + 1, "image": filename, "action": None})
                break
    finally:
        env.close()

    metadata["number_of_saved_observations"] = len(metadata["frames"])
    metadata["number_of_generated_actions"] = sum(frame["action"] is not None for frame in metadata["frames"])
    metadata_path = args.output_dir / "trajectory_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {metadata['number_of_saved_observations']} observations")
    print(f"Generated {metadata['number_of_generated_actions']} actions")
    print(f"Final success: {metadata['success']}")
    print(f"Output directory: {args.output_dir.resolve()}")
    print(f"Metadata: {metadata_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate fixed-pair or continuous OpenVLA LIBERO trajectory frames for KV-cache experiments."
    )
    parser.add_argument("--mode", choices=("fixed-pair", "rollout"), default="rollout")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "trajectory_frames")
    parser.add_argument("--num-steps", type=int, default=100, help="Maximum number of consecutive saved rollout frames.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--num-steps-wait", type=int, default=10, help="Official-style unrecorded settling steps.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--center-crop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--previous-name", default="control_step_t.png")
    parser.add_argument("--current-name", default="control_step_t_plus_1.png")
    parser.add_argument(
        "--step-action",
        type=float,
        nargs=7,
        default=DEFAULT_STEP_ACTION,
        metavar=("DX", "DY", "DZ", "DRX", "DRY", "DRZ", "GRIPPER"),
        help="LIBERO action applied between the two saved observations.",
    )
    args = parser.parse_args()
    if args.mode == "fixed-pair":
        run_fixed_pair(args)
    else:
        run_rollout(args)


if __name__ == "__main__":
    main()
