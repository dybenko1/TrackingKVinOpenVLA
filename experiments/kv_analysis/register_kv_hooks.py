"""
register_kv_hooks.py: instrumentation only. It does not run OpenVLA
itself. It only attaches “observers” to the K and V projection modules 
so that, whenever those modules execute during inference, their outputs
are copied into a dictionary.
"""
import torch


class KVHookManager:
    # Its job is register hooks, collect KV outputs, inspect them and later
    #  clean those hooks up
    def __init__(self):
        # Where kv (tensors) for every layer will be stored, e.g.
        # self.activations["layer_0"]["k"]
        # self.activations["layer_0"]["v"]
        self.activations = {}

        # Keep hook handles so we can remove them later.
        # everytime we register (add) a hook an object named handler is created
        # it only contains like the ID of the hook, useful to then remove the hook

        """
            register hook
                ↓
            PyTorch gives handle
                ↓
            save handle
                ↓
            later:
            handle.remove()
        """
        self.handles = []

    def _make_hook(self, layer_idx, kind):
        """
        This is the prep to then do the hooking, for that see register()
        It creates a hook function for one specific layer and projection.
        Kind is either:
                    "k"
                    "v"
        
        This is a closure: function that returns anoter function. It is useful
        because the inner function remembers the input of _make_hook even if 
        it finishes. Useful because 1. Pytorch expects every forward hook
        to have the same signature: hook(module, inputs, output) hence
        this format does nto allow us to identify to which layer is this
        hook attached, and also its kind of data (k or v) it will track

        
        """

        def hook(module, inputs, output):
            layer_key = f"layer_{layer_idx}"

            if layer_key not in self.activations:
                self.activations[layer_key] = {
                    "k": [],
                    "v": [],
                }

            # Detach so the stored tensor is not connected to PyTorch autogra used 
            # for gradient calculation. This tells that we want this tensor as data
            # only, not to track its history for gradients
            
            # Move to CPU so we don't fill GPU memory while
            # collecting tensors from all 32 layers.
            self.activations[layer_key][kind].append(
                output.detach().cpu()
                # detach() removes the tensor from the autograd graph;
            )

        return hook

    def register(self, vla):
        """
        Attach hooks to k_proj and v_proj in every
        Transformer layer of OpenVLA.
        """

        self.remove()
        # We call the above method because in case we call register twice we don't
        # want two sets of hooks attached. So this is initial cleaning

        layers = vla.language_model.model.layers

        print(
            f"Registering K/V hooks on {len(layers)} "
            "Transformer layers..."
        )

        for layer_idx, layer in enumerate(layers):

            k_handle = (
                layer.self_attn.k_proj.register_forward_hook(
                    self._make_hook(layer_idx, "k") # this is the hook
                )
            )
            # What is layer.self_attn.k_proj? 
            # It is the path through the model hierarchy.
            """
            layer       one Transformer layer
                ↓
            self_attn   that layer’s self-attention module
                ↓
            k_proj      the linear projection that computes the Key representation
            """

            # what is register_forward_hook?
            # It is a method provided by PyTorch’s nn.Module. "It is the hooking"
            # When module finishes itsw forward computatio, Pytorch calls the hook

            v_handle = (
                layer.self_attn.v_proj.register_forward_hook(
                    self._make_hook(layer_idx, "v")
                )
            )

            self.handles.extend(
                [k_handle, v_handle]
            )

        print(
            f"Registered {len(self.handles)} hooks."
        )

    def clear(self):
        """
        Delete activations collected during the previous
        inference/control step.
        """
        self.activations.clear()

    def remove(self):
        """
        Remove hooks from the model.

        Why remove?
        Because once registered, hooks stay attached to the module. 
        If you accidentally register them again, you can get duplicate
        executions and duplicate storage.
        """
        for handle in self.handles:
            handle.remove()

        self.handles.clear()

    def get_layer(self, layer_idx):
        """
        Return K and V captured (means saving a copy/reference of an intermediate) for one Transformer layer.
        """
        return self.activations.get(
            f"layer_{layer_idx}"
        )

    def summary(self):
        """
        Print the tensors captured by the hooks.
        """

        print("\n=== Captured K/V tensors ===")

        for layer_name in sorted(
            self.activations.keys(),
            key=lambda x: int(x.split("_")[1]),
        ):
            values = self.activations[layer_name]

            print(f"\n{layer_name}")

            for kind in ["k", "v"]:
                tensors = values.get(kind, [])

                print(f"  {kind.upper()} calls: {len(tensors)}")

                for call_idx, tensor in enumerate(tensors):
                    print(
                        f"    call {call_idx}: "
                        f"shape={tuple(tensor.shape)} "
                        f"dtype={tensor.dtype}"
                    )

    def get_prefill(self, layer_idx):
        """
        Return the K and V tensors from the first forward call,
        which corresponds to the prefill call in the current experiment.
        """
        layer_data = self.get_layer(layer_idx)

        if layer_data is None:
            return None

        if len(layer_data["k"]) == 0 or len(layer_data["v"]) == 0:
            return None

        return {
            "k": layer_data["k"][0],
            "v": layer_data["v"][0],
        }