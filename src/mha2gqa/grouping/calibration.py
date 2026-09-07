import torch
from .kmean import PyTorchKMeans

class Calibration:
    def __init__(self, model_config, user_config):
        self.captured = {}
        self.model = user_config.model_id
        self.LAYER_IDX = 0
        self.num_kv_groups = user_config.target_kv_groups
        self.num_heads = getattr(model_config, "num_attention_heads", None) or getattr(model_config, "num_heads", None)
        self.head_dim = getattr(model_config, "head_dim", None) or (model_config.hidden_size // self.num_heads if self.num_heads else None)

    def _build_head_permutation(self, clustered_heads):
        return torch.cat([torch.tensor(group) for group in clustered_heads])

    def capture_v_proj_output(self, module, inputs, output):
        # output shape: [batch, seq_len, hidden_size]
        self.captured["v_proj_out"] = output.detach()
    
    def calibrate(self, model, calibration_texts):
        handle = model.model.layers[self.LAYER_IDX].self_attn.v_proj.register_forward_hook(self.capture_v_proj_output)

        all_head_vectors = []  # will collect [num_heads, head_dim] per example, then average

        with torch.no_grad():
            for text in calibration_texts:
                batch_inputs = {}
                for k, v in text.items():
                    if isinstance(v, torch.Tensor):
                        tensor_v = v.to(model.device)
                    else:
                        tensor_v = torch.tensor(v, device=model.device)
                    if tensor_v.ndim == 1:
                        tensor_v = tensor_v.unsqueeze(0)
                    batch_inputs[k] = tensor_v

                model(**batch_inputs)
                v_out = self.captured["v_proj_out"]  # [batch, seq_len, hidden_size]
                v_out = v_out.view(-1, self.num_heads, self.head_dim)  # [total_tokens, num_heads, head_dim]
                all_head_vectors.append(v_out.mean(dim=0))  # mean over tokens -> [num_heads, head_dim]

        handle.remove()

        head_matrix = torch.stack(all_head_vectors).mean(dim=0)  # [num_heads, head_dim]

        grouping = PyTorchKMeans(n_clusters=self.num_kv_groups, max_iter=300, tol=1e-4).fit_predict(head_matrix)
        clustered_heads = [[] for _ in range(self.num_kv_groups)]
        for head_id, cluster_id in enumerate(grouping):
            clustered_heads[cluster_id].append(head_id)

        clustered_heads_tensors = [torch.tensor(heads) for heads in clustered_heads]
        return clustered_heads_tensors

    def calibrate_for_permutation(self, model, calibration_texts):
        return self.calibrate(model, calibration_texts)
