"""
run_language_kv_analysis.py: experiment driver. It should load 
OpenVLA, prepare image + instruction inputs, 
register the hooks, run inference over multiple control steps, isolate 
the language-token positions, 
and save/compare the captured K/V values.
"""
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor 
# Autoprocessor loads OpenVLA's input processor, the one transforms image and language into tensors
# The processor handles input just like humands understand, text and images and transforms it to numbers for the model
# AutoModelForVision2Seq loads the VLA model



MODEL_ID = "openvla/openvla-7b"


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