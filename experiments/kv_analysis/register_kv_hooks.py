"""
register_kv_hooks.py: instrumentation only. Its job is to find the LLM backbone layers, 
attach forward hooks to k_proj and v_proj, and collect the tensors produced at each layer. 
It should not run the whole experiment or make plots.
"""