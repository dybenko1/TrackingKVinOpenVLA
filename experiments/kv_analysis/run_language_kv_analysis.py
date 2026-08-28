"""
run_language_kv_analysis.py: experiment driver. It should load 
OpenVLA, prepare image + instruction inputs, 
register the hooks, run inference over multiple control steps, isolate 
the language-token positions, 
and save/compare the captured K/V values.
"""
from PIL import Image
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor 
# Autoprocessor loads OpenVLA's input processor, the one transforms image and language into tensors
# The processor handles input just like humands understand, text and images and transforms it to numbers for the model
# AutoModelForVision2Seq loads the VLA model

from register_kv_hooks import KVHookManager


MODEL_ID = "openvla/openvla-7b"

# Change this to a real image on your machine
# TODO
IMAGE_PATH = "experiments/kv_analysis/inputs/imagenLIBERO.png"

INSTRUCTION = "pick up the red object"




def load_model():
    """
    Loads Hugging Face version of OpenVLA
    """
    print(f"Loading {MODEL_ID}...")

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        
        trust_remote_code=True, # Since this model (OpenVLA) is not the standard Hugging Face 
        # architecture, it needs to load custom code so this let it load
        # example of what is loaded: OpenVLAForActionPrediction, PrismaticProcessor, OpenVLAConfig
    )

    # Loading the actual neural network
    vla = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, # tells HF to use memory-efficient loading procedure
        trust_remote_code=True, # allows HF to load OpenVLA's custom implementation
    )

    # Moving the model to GPU, the processor doesn't need to be moved, but the tensor it produces its moved to GPU
    if torch.cuda.is_available():
        vla = vla.to("cuda")

    # Put model in evaluation/predict mode. E.g. module.training = False (so things like dropout are not implemented)
    vla.eval()


    return vla, processor


def main():
    vla, processor = load_model()

    # 1. Register the hooks
    kv_hooks = KVHookManager()
    kv_hooks.register(vla)

    # 2. Load one RGB image
    image = Image.open(IMAGE_PATH).convert("RGB")

    # 3. Build the prompt format expected by OpenVLA
    prompt = (
        f"In: What action should the robot take to {INSTRUCTION}?\n"
        "Out:"
    )

    # 4. Processor converts image + text into tensors
    inputs = processor(
        prompt,
        image,
    )

    # Inspect the original language input before visual tokens
    # are inserted into the LLM sequence.
    input_ids = inputs["input_ids"]

    print("\n=== Original text input ===")
    print("input_ids shape:", input_ids.shape)

    tokenizer = processor.tokenizer

    token_ids = input_ids[0].tolist() # Grabbing tokens ids from the prompt [0] only, because image is [1]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    for idx, (token_id, token) in enumerate(zip(token_ids, tokens)):
        print(
            f"text position {idx:2d}: "
            f"id={token_id:6d}  token={repr(token)}"
        )


    # 5. Move processor outputs to the model device
    if torch.cuda.is_available():
        inputs = inputs.to(
            "cuda",
            dtype=torch.bfloat16,
        )

    # Important: clear old captures before this inference.
    kv_hooks.clear()

    # 6. Run one real OpenVLA inference
    with torch.inference_mode():
        action = vla.predict_action(
            **inputs,
            unnorm_key="bridge_orig",
            do_sample=False,
        )

    print("\nPredicted action:")
    print(action)

    # 7. The hooks have now fired.
    kv_hooks.summary()

    # Example: inspect layer 0
    prefill_0 = kv_hooks.get_prefill(0)

    if prefill_0 is not None:
        print("\nLayer 0 prefill:")
        print("K:", prefill_0["k"].shape)
        print("V:", prefill_0["v"].shape)

        # Number of positions entering the LLM during prefill.
        prefill_seq_len = prefill_0["k"].shape[1]

        # Number of original textual tokens.
        text_seq_len = input_ids.shape[1]

        # OpenVLA inserts the visual patch embeddings after BOS.
        num_visual_tokens = prefill_seq_len - text_seq_len

        print("\n=== Multimodal sequence composition ===")
        print("Prefill sequence length:", prefill_seq_len)
        print("Text token count:", text_seq_len)
        print("Visual token count:", num_visual_tokens)

        # OpenVLA sequence:
        #
        # [BOS] [VISUAL TOKENS] [REMAINING TEXT TOKENS]
        #
        bos_position = 0

        visual_start = 1
        visual_end = visual_start + num_visual_tokens

        language_start = visual_end

        print("\nPosition ranges inside LLM prefill:")
        print(f"BOS:                 {bos_position}")
        print(
            f"Visual tokens:       "
            f"{visual_start} ... {visual_end - 1}"
        )
        print(
            f"Language tokens:     "
            f"{language_start} ... {prefill_seq_len - 1}"
        )

        print("\n=== Text token -> multimodal position mapping ===")

        for text_idx, (token_id, token) in enumerate(
            zip(token_ids, tokens)
        ):
            if text_idx == 0:
                multimodal_pos = 0
            else:
                multimodal_pos = (
                    num_visual_tokens + text_idx
                )

            print(
                f"text_idx={text_idx:2d}  "
                f"multimodal_pos={multimodal_pos:3d}  "
                f"id={token_id:6d}  "
                f"token={repr(token)}"
            )

    

    # 8. Clean up hooks when finished
    kv_hooks.remove()


if __name__ == "__main__":
    main()