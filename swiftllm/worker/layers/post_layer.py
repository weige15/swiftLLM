import torch

from swiftllm.model_config import LlamaModelConfig
from swiftllm.worker.weight import LlamaWeight
from swiftllm.worker.kernels.rmsnorm import rmsnorm_inplace
from swiftllm.worker.infer_state import LlamaInferState
from swiftllm.worker.kernels.linear import linear

class LlamaPostLayer:
    def __init__(
        self,
        model_config: LlamaModelConfig,
        weights: LlamaWeight,
    ):
        self.model_config = model_config
        self.weights = weights
    
    def forward(
        self,
        input_embds: torch.Tensor,	# [num_total_tokens, hidden_size]
        infer_state: LlamaInferState,
        return_token_scores: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Slice to get the last token embedding for each request
        last_token_indices = torch.cat(
            (
                infer_state.prefill_seq_start_locs + infer_state.prefill_seq_lens - 1,
                torch.arange(infer_state.num_prefill_tokens, infer_state.num_tokens, device=input_embds.device, dtype=torch.int32)
            ), dim=0
        )
        last_input = torch.empty((infer_state.batch_size, self.model_config.hidden_size), device=input_embds.device, dtype=input_embds.dtype)
        last_input[:, :] = input_embds[last_token_indices, :]
        # Apply RMS-norm
        rmsnorm_inplace(
            last_input,
            self.weights.final_norm,
            self.model_config.rms_norm_eps
        )
        logits = linear(last_input, self.weights.lm_head)    # [batch_size, vocab_size]
        output_tokens = torch.argmax(logits, dim=1)
        if not return_token_scores:
            return output_tokens

        # Scores of the chosen tokens, computed in float32 from the FP16 logits:
        # - log_probs: log-probability of the chosen token
        # - top2_gaps: logit of the chosen token minus logit of the runner-up
        #   (0 means the choice was an exact tie, broken by argmax)
        logits = logits.float()
        log_probs = torch.log_softmax(logits, dim=1).gather(1, output_tokens[:, None]).squeeze(1)
        top2 = torch.topk(logits, 2, dim=1).values
        top2_gaps = top2[:, 0] - top2[:, 1]
        return output_tokens, log_probs, top2_gaps
    