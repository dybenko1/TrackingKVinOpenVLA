FROM openvla-cached:latest

# Separate derived image. The validated OpenVLA image itself is not modified.
# Keep its pinned torch/CUDA/timm/transformers stack and add only OpenCLIP's
# Python package for the paired SigLIP text tower.
USER root
RUN python3 -m pip install --no-cache-dir --no-deps --index-url https://pypi.org/simple open_clip_torch==2.32.0 ftfy==6.3.1
USER 2011:2011
