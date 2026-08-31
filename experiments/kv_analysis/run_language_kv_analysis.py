"""
run_language_kv_analysis.py: experiment driver. It should load 
OpenVLA, prepare image + instruction inputs, 
register the hooks, run inference over multiple control steps, isolate 
the language-token positions, 
and save/compare the captured K/V values.
"""
from PIL import Image
import os
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor 

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from register_kv_hooks import KVHookManager

import numpy as np

import imageio



# running the openvla finetuned for LIBERO, not the base model
MODEL_ID = "openvla/openvla-7b-finetuned-libero-spatial"

OUTPUT_DIR = "experiments/kv_analysis/outputs"

TASK_SUITE = "libero_spatial"


def load_model():
    """
    Loads Hugging Face version of OpenVLA
    """
    print(f"Loading {MODEL_ID}...")

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        trust_remote_code=True, # allows HF to load OpenVLA's custom implementation
    )

    vla = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, 
        trust_remote_code=True, 
    )

    # Moving the model to GPU, the processor doesn't need to be moved, but the tensor it produces its moved to GPU
    if torch.cuda.is_available():
        vla = vla.to("cuda")

    # Put model in evaluation/predict mode. 
    vla.eval()

    return vla, processor

def load_libero_env():
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(0)

    bddl_path = os.path.join(
        get_libero_path("bddl_files"),
        task.problem_folder,
        task.bddl_file,
    )

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=256,
        camera_widths=256,
    )

    # Get predefined initial states for task 0
    initial_states = suite.get_task_init_states(0)

    env.reset()

    # Use initial state 0 for this experiment
    obs = env.set_init_state(initial_states[0])

    return env, task, obs

# Helper functions to convert action from OpenVLa format to LIBERO format to run the simulation
def normalize_gripper_action(action, binarize=True):
    action = action.copy()
    # Convert gripper from [0, 1] to [-1, 1].
    action[..., -1] = 2 * action[..., -1] - 1
    if binarize:
        action[..., -1] = np.sign(action[..., -1])
    return action
def invert_gripper_action(action):
    action = action.copy()
    # Match LIBERO's gripper convention.
    action[..., -1] *= -1
    return action


def main():
    vla, processor = load_model()

    # Register K/V hooks
    kv_hooks = KVHookManager()
    kv_hooks.register(vla)

    # Load real LIBERO environment
    env, task, obs = load_libero_env()

    # Use LIBERO task language
    instruction = task.language

    # Default LIBERO action normalization key.
    # Fall back to the "_no_noops" variant if needed.
    # Ensure the checkpoint has valid action normalization statistics.
    unnorm_key = TASK_SUITE
    if (
        unnorm_key not in vla.norm_stats
        and f"{unnorm_key}_no_noops" in vla.norm_stats
    ):
        unnorm_key = f"{unnorm_key}_no_noops"
    assert unnorm_key in vla.norm_stats, (
        f"Action un-norm key {unnorm_key} not found in model norm_stats!"
    )

    print("Using unnorm key:", unnorm_key)


    print("\nInstruction:")
    print(instruction)

    # Store language K/V for every OpenVLA control step
    trajectory_kv = []

    video_path = "experiments/kv_analysis/libero_trajectory.mp4"
    video_writer = imageio.get_writer(
        video_path,
        fps=20,
    )

    max_steps = 220
    num_steps_wait = 10

    # Used while allowing the environment to settle
    dummy_action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]

    done = False

    # to store video of simulation
    video_frames = []

    for step in range(max_steps + num_steps_wait):

         # Save current simulator frame
        video_writer.append_data(
            np.rot90(obs["agentview_image"], 2)
        )

        # Let the environment settle before running the policy
        if step < num_steps_wait:
            obs, reward, done, info = env.step(dummy_action)
            continue

        # Save current simulator frame for video
        video_frames.append(obs["agentview_image"].copy())

        # Current LIBERO observation
        image_np = np.rot90(obs["agentview_image"], 2)
        image = Image.fromarray(image_np)

        # Same language instruction at every control step
        prompt = (
            f"In: What action should the robot take to {instruction}?\n"
            "Out:"
        )

        # Convert current image + language into OpenVLA inputs
        inputs = processor(
            prompt,
            image,
        )

        num_prompt_tokens = inputs["input_ids"].shape[1]

        # to then identify the token IDs later for our analysis
        prompt_token_ids = inputs["input_ids"][0].clone().cpu()

        if torch.cuda.is_available():
            inputs = inputs.to(
                "cuda",
                dtype=torch.bfloat16,
            )

        # Clear K/V captured from the previous control step
        kv_hooks.clear()

        # Predict one action for the current observation
        with torch.inference_mode():
            action = vla.predict_action(
                **inputs,
                unnorm_key=unnorm_key,
                do_sample=False,
            )

        # Extract language K/V from all 32 layers
        language_kv = {}

        for layer_idx in range(32):
            language_kv[layer_idx] = kv_hooks.get_language_prefill(
                layer_idx,
                num_prompt_tokens=num_prompt_tokens,
            )

        # Save this control step's K/V
        trajectory_kv.append(language_kv)

        print(f"\nControl step {len(trajectory_kv) - 1}")
        print("Predicted action:")
        print(action)

        # Convert OpenVLA gripper convention to LIBERO convention
        action = normalize_gripper_action(
            action,
            binarize=True,
        )
        action = invert_gripper_action(action)

        # Execute action and obtain the next observation
        obs, reward, done, info = env.step(action.tolist())

        if done:
            print(
                f"\nTask completed after "
                f"{len(trajectory_kv)} OpenVLA control steps."
            )
            break

    if not done:
        print(
            f"\nTask did not complete within "
            f"{max_steps} OpenVLA control steps."
        )

    # Inspect one example:
    # layer 0 K/V from control step 0
    if trajectory_kv:
        language_0 = trajectory_kv[0][0]

        if language_0 is not None:
            print("\nStep 0, Layer 0 language prefill:")
            print("K:", language_0["k"].shape)
            print("V:", language_0["v"].shape)
            print("Positions:", language_0["positions"])


    video_writer.close()
    print(f"\nSaved video to: {video_path}")

    # Saving KV to disk. Create output directory if it does not exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    kv_path = os.path.join(
        OUTPUT_DIR,
        "trajectory_task0_episode0.pt",
    )
    torch.save(
        {
            "model_id": MODEL_ID,
            "task_suite": TASK_SUITE,
            "task_id": 0,
            "episode_id": 0,
            "instruction": instruction,
            "num_steps": len(trajectory_kv),
            "num_prompt_tokens": num_prompt_tokens,
            "trajectory_kv": trajectory_kv,
            "prompt_token_ids": prompt_token_ids,
        },
        kv_path,
    )
    print(f"\nSaved K/V trajectory to: {kv_path}")

    env.close()
    kv_hooks.remove()
        

if __name__ == "__main__":
    main()