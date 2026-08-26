import os
from pathlib import Path

from PIL import Image

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


TASK_SUITE_NAME = "libero_spatial"
TASK_ID = 0
INITIAL_STATE_ID = 0

RESOLUTION = 256

OUTPUT_DIR = Path("experiments/kv_analysis/outputs")
OUTPUT_PATH = OUTPUT_DIR / "libero_observation.png"


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


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load benchmark suite.
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[TASK_SUITE_NAME]()

    print(f"Task suite: {TASK_SUITE_NAME}")
    print(f"Number of tasks: {task_suite.n_tasks}")

    # 2. Select task.
    task = task_suite.get_task(TASK_ID)

    # 3. Get LIBERO's predefined initial states.
    initial_states = task_suite.get_task_init_states(TASK_ID)

    # 4. Create environment.
    env, task_description = create_libero_env(
        task,
        resolution=RESOLUTION,
    )

    print(f"Task description: {task_description}")

    try:
        # 5. Reset simulator.
        env.reset()

        # 6. Load one predefined initial state.
        obs = env.set_init_state(
            initial_states[INITIAL_STATE_ID]
        )

        print("\nObservation keys:")
        for key in obs.keys():
            print(f"  {key}")

        # 7. Extract RGB observation.
        img = extract_libero_image(obs)

        print("\nImage:")
        print(f"  shape: {img.shape}")
        print(f"  dtype: {img.dtype}")

        # 8. Save it.
        Image.fromarray(img).save(OUTPUT_PATH)

        print("\nSaved image to:")
        print(OUTPUT_PATH.resolve())

    finally:
        env.close()


if __name__ == "__main__":
    main()