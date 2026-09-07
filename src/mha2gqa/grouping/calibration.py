import torch
from .kmean import PyTorchKMeans

class Calibration:
    def __init__(self, model_config, user_config):
        self.num_kv_groups = user_config.target_kv_groups
        self.num_heads = getattr(model_config, "num_attention_heads", None) or getattr(model_config, "num_heads", None)
        self.head_dim = getattr(model_config, "head_dim", None) or (model_config.hidden_size // self.num_heads if self.num_heads else None)

    def calibrate(self, model, calibration_texts):
        num_layers = len(model.model.layers)
        captured = {i: [] for i in range(num_layers)}
        handles = []

        # Register forward hook on v_proj for EVERY layer
        def make_hook(layer_idx):
            def hook(module, inputs, output):
                captured[layer_idx].append(output.detach())
            return hook

        for i in range(num_layers):
            h = model.model.layers[i].self_attn.v_proj.register_forward_hook(make_hook(i))
            handles.append(h)
        try:
            with torch.no_grad():
                for text in calibration_texts:
                    batch_inputs = {}
                    for k, v in text.items():
                        tensor_v = v.to(model.device) if isinstance(v, torch.Tensor) else torch.tensor(v, device=model.device)
                        if tensor_v.ndim == 1:
                            tensor_v = tensor_v.unsqueeze(0)
                        batch_inputs[k] = tensor_v
                    model(**batch_inputs)
        finally:
            for h in handles:
                h.remove()

        per_layer_groupings = {}
        for i in range(num_layers):
            layer_v_outs = captured[i]
            layer_head_vectors = []
            for v_out in layer_v_outs:
                v_out_reshaped = v_out.view(-1, self.num_heads, self.head_dim)
                layer_head_vectors.append(v_out_reshaped.mean(dim=0))
            
            head_matrix = torch.stack(layer_head_vectors).mean(dim=0)  # [num_heads, head_dim]
            grouping = PyTorchKMeans(n_clusters=self.num_kv_groups, max_iter=300, tol=1e-4).fit_predict(head_matrix)
            
            clustered_heads = [[] for _ in range(self.num_kv_groups)]
            for head_id, cluster_id in enumerate(grouping):
                clustered_heads[cluster_id].append(head_id)
            
            per_layer_groupings[i] = [torch.tensor(h_ids) for h_ids in clustered_heads]

        return per_layer_groupings

    def calibrate_for_permutation(self, model, calibration_texts):
        return self.calibrate(model, calibration_texts)

