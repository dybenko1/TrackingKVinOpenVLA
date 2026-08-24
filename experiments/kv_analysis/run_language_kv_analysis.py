"""
run_language_kv_analysis.py: experiment driver. It should load OpenVLA, prepare image + instruction inputs, 
register the hooks, run inference over multiple control steps, isolate the language-token positions, 
and save/compare the captured K/V values.
"""
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor


MODEL_ID = "openvla/openvla-7b"


def load_model():
    print(f"Loading {MODEL_ID}...")

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )

    vla = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    if torch.cuda.is_available():
        vla = vla.to("cuda")

    vla.eval()

    return vla, processor


def inspect_kv_projection_modules(vla):
    print("\n=== K/V projection modules ===\n")

    found = 0

    for name, module in vla.named_modules():
        if name.endswith("k_proj") or name.endswith("v_proj"):
            print(f"{name}: {type(module)}")
            found += 1

    print(f"\nFound {found} K/V projection modules.")


def main():
    vla, processor = load_model()

    print("\nModel loaded successfully.")
    print("Model type:", type(vla))
    print("Processor type:", type(processor))

    inspect_kv_projection_modules(vla)


if __name__ == "__main__":
    main()