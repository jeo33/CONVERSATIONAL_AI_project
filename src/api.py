import os
import types
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import inspect


def _patch_torch_autocast_compat():
    try:
        params = inspect.signature(torch.is_autocast_enabled).parameters
    except (TypeError, ValueError):
        params = {}
    if "device_type" not in params:
        _orig = torch.is_autocast_enabled
        def _compat(device_type=None):
            return _orig()
        torch.is_autocast_enabled = _compat

_patch_torch_autocast_compat()


CACHE_SIZE  = 512
MAX_SEQ_LEN = 4096

KV_CONFIGS = {
    "full":      {"h2o": False, "random": False, "budget_ratio": 1.0, "recent_ratio": 1.0},
    "random_b1": {"h2o": False, "random": True,  "budget_ratio": 0.1, "recent_ratio": 0.1},
    "random_b2": {"h2o": False, "random": True,  "budget_ratio": 0.2, "recent_ratio": 0.1},
    "random_b4": {"h2o": False, "random": True,  "budget_ratio": 0.4, "recent_ratio": 0.1},
    "random_b6": {"h2o": False, "random": True,  "budget_ratio": 0.6, "recent_ratio": 0.1},
    "random_b8": {"h2o": False, "random": True,  "budget_ratio": 0.8, "recent_ratio": 0.1},
    "h2o_b1_r1": {"h2o": True,  "random": False, "budget_ratio": 0.1, "recent_ratio": 0.1},
    "h2o_b2_r1": {"h2o": True,  "random": False, "budget_ratio": 0.2, "recent_ratio": 0.1},
    "h2o_b2_r2": {"h2o": True,  "random": False, "budget_ratio": 0.2, "recent_ratio": 0.2},
    "h2o_b3_r1": {"h2o": True,  "random": False, "budget_ratio": 0.3, "recent_ratio": 0.1},
    "h2o_b3_r2": {"h2o": True,  "random": False, "budget_ratio": 0.3, "recent_ratio": 0.2},
    "h2o_b3_r3": {"h2o": True,  "random": False, "budget_ratio": 0.3, "recent_ratio": 0.3},
    "h2o_b4_r1": {"h2o": True,  "random": False, "budget_ratio": 0.4, "recent_ratio": 0.1},
    "h2o_b4_r2": {"h2o": True,  "random": False, "budget_ratio": 0.4, "recent_ratio": 0.2},
    "h2o_b4_r3": {"h2o": True,  "random": False, "budget_ratio": 0.4, "recent_ratio": 0.3},
    "h2o_b4_r4": {"h2o": True,  "random": False, "budget_ratio": 0.4, "recent_ratio": 0.4},
    "h2o_b5_r1": {"h2o": True,  "random": False, "budget_ratio": 0.5, "recent_ratio": 0.1},
    "h2o_b5_r2": {"h2o": True,  "random": False, "budget_ratio": 0.5, "recent_ratio": 0.2},
    "h2o_b5_r3": {"h2o": True,  "random": False, "budget_ratio": 0.5, "recent_ratio": 0.3},
    "h2o_b5_r4": {"h2o": True,  "random": False, "budget_ratio": 0.5, "recent_ratio": 0.4},
    "h2o_b5_r5": {"h2o": True,  "random": False, "budget_ratio": 0.5, "recent_ratio": 0.5},
    "h2o_b6_r1": {"h2o": True,  "random": False, "budget_ratio": 0.6, "recent_ratio": 0.1},
    "h2o_b6_r2": {"h2o": True,  "random": False, "budget_ratio": 0.6, "recent_ratio": 0.2},
    "h2o_b6_r3": {"h2o": True,  "random": False, "budget_ratio": 0.6, "recent_ratio": 0.3},
    "h2o_b6_r4": {"h2o": True,  "random": False, "budget_ratio": 0.6, "recent_ratio": 0.4},
    "h2o_b6_r5": {"h2o": True,  "random": False, "budget_ratio": 0.6, "recent_ratio": 0.5},
    "h2o_b7_r1": {"h2o": True,  "random": False, "budget_ratio": 0.7, "recent_ratio": 0.1},
    "h2o_b7_r2": {"h2o": True,  "random": False, "budget_ratio": 0.7, "recent_ratio": 0.2},
    "h2o_b7_r3": {"h2o": True,  "random": False, "budget_ratio": 0.7, "recent_ratio": 0.3},
    "h2o_b7_r4": {"h2o": True,  "random": False, "budget_ratio": 0.7, "recent_ratio": 0.4},
    "h2o_b7_r5": {"h2o": True,  "random": False, "budget_ratio": 0.7, "recent_ratio": 0.5},
    "h2o_b8_r1": {"h2o": True,  "random": False, "budget_ratio": 0.8, "recent_ratio": 0.1},
    "h2o_b8_r2": {"h2o": True,  "random": False, "budget_ratio": 0.8, "recent_ratio": 0.2},
    "h2o_b8_r3": {"h2o": True,  "random": False, "budget_ratio": 0.8, "recent_ratio": 0.3},
    "h2o_b8_r4": {"h2o": True,  "random": False, "budget_ratio": 0.8, "recent_ratio": 0.4},
    "h2o_b8_r5": {"h2o": True,  "random": False, "budget_ratio": 0.8, "recent_ratio": 0.5},
    "local_b1":  {"h2o": True,  "random": False, "budget_ratio": 0.1, "recent_ratio": 0.1},
    "local_b2":  {"h2o": True,  "random": False, "budget_ratio": 0.2, "recent_ratio": 0.2},
    "local_b3":  {"h2o": True,  "random": False, "budget_ratio": 0.3, "recent_ratio": 0.3},
    "local_b4":  {"h2o": True,  "random": False, "budget_ratio": 0.4, "recent_ratio": 0.4},
    "local_b5":  {"h2o": True,  "random": False, "budget_ratio": 0.5, "recent_ratio": 0.5},
    "local_b6":  {"h2o": True,  "random": False, "budget_ratio": 0.6, "recent_ratio": 0.6},
    "local_b7":  {"h2o": True,  "random": False, "budget_ratio": 0.7, "recent_ratio": 0.7},
    "local_b8":  {"h2o": True,  "random": False, "budget_ratio": 0.8, "recent_ratio": 0.8},
}


try:
    from rouge_score import rouge_scorer as _rs
    _scorer = _rs.RougeScorer(['rougeL'], use_stemmer=False)
    def compute_rouge_l(pred, ref):
        if not pred.strip() or not ref.strip():
            return 0.0
        return _scorer.score(ref, pred)['rougeL'].fmeasure
except ImportError:
    def compute_rouge_l(pred, ref):
        p, r = pred.lower().split(), ref.lower().split()
        m, n = len(p), len(r)
        if not m or not n:
            return 0.0
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                dp[i][j] = dp[i-1][j-1] + 1 if p[i-1] == r[j-1] else max(dp[i-1][j], dp[i][j-1])
        lcs = dp[m][n]
        pr = lcs / m; rc = lcs / n
        return 2 * pr * rc / (pr + rc) if pr + rc else 0.0


def apply_h2o(model, budget_ratio=0.2, recent_ratio=0.1,
              cache_size=CACHE_SIZE, strategy="per_head"):
    num_layers    = len(model.model.layers)
    acc_scores    = [None] * num_layers
    step_counter  = [0]
    TOTAL_BUDGET  = max(16, int(cache_size * budget_ratio))
    RECENT_BUDGET = max(8,  int(cache_size * recent_ratio))
    HH_BUDGET     = TOTAL_BUDGET - RECENT_BUDGET
    NUM_Q_HEADS, NUM_KV_HEADS = 32, 8
    GQA_GROUP = NUM_Q_HEADS // NUM_KV_HEADS

    def reset_scores():
        for li in range(num_layers):
            acc_scores[li] = None
        step_counter[0] = 0

    def make_patched_forward(orig, li):
        def patched_forward(self, hidden_states, **kwargs):
            kwargs["output_attentions"] = True
            out = orig(hidden_states, **kwargs)
            attn_w = out[1] if isinstance(out, tuple) else getattr(out, "attentions", None)
            if attn_w is not None:
                raw_score = attn_w.float().mean(dim=2)[0]
                score = raw_score.view(NUM_KV_HEADS, GQA_GROUP, -1).mean(dim=1).detach()
                if acc_scores[li] is None:
                    acc_scores[li] = score
                else:
                    cur_len, new_len = acc_scores[li].shape[1], score.shape[1]
                    if new_len > cur_len:
                        pad = torch.zeros(NUM_KV_HEADS, new_len - cur_len, device=score.device)
                        acc_scores[li] = torch.cat([acc_scores[li], pad], dim=1)
                    acc_scores[li][:, :new_len] += score
            return (out[0], None) + out[2:] if isinstance(out, tuple) else out
        return patched_forward

    orig_forwards = {}
    for layer_idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        orig_forwards[layer_idx] = attn.forward
        attn.forward = types.MethodType(make_patched_forward(attn.forward, layer_idx), attn)

    def eviction_hook(module, input, output):
        past_kv = getattr(output, "past_key_values", None)
        if past_kv is None or not hasattr(past_kv, 'layers'):
            return output
        for li in range(len(past_kv.layers)):
            if acc_scores[li] is None:
                continue
            k = past_kv.layers[li].keys
            v = past_kv.layers[li].values
            bsz, n_heads, seq_len, head_dim = k.shape
            if seq_len <= TOTAL_BUDGET:
                continue
            device = k.device
            recent_start = seq_len - RECENT_BUDGET
            old_scores = acc_scores[li][:, :recent_start]
            if strategy == "layer_shared":
                hh_idx = old_scores.mean(dim=0).topk(HH_BUDGET).indices.unsqueeze(0).expand(n_heads, -1)
            else:
                hh_idx = old_scores.topk(HH_BUDGET, dim=1).indices
            recent_idx = torch.arange(recent_start, seq_len, device=device).unsqueeze(0).expand(n_heads, -1)
            keep_idx, _ = torch.cat([hh_idx, recent_idx], dim=1).sort(dim=1)
            g = keep_idx.unsqueeze(0).unsqueeze(-1).expand(bsz, -1, -1, head_dim)
            past_kv.layers[li].keys   = torch.gather(k, 2, g)
            past_kv.layers[li].values = torch.gather(v, 2, g)
            acc_scores[li] = torch.gather(acc_scores[li], 1, keep_idx)
        return output

    hook = model.model.register_forward_hook(eviction_hook)
    return hook, reset_scores, orig_forwards


def apply_random(model, budget_ratio=0.2, cache_size=CACHE_SIZE):
    num_layers   = len(model.model.layers)
    TOTAL_BUDGET = max(16, int(cache_size * budget_ratio))

    def reset_scores():
        pass

    def eviction_hook(module, input, output):
        past_kv = getattr(output, "past_key_values", None)
        if past_kv is None or not hasattr(past_kv, 'layers'):
            return output
        for li in range(len(past_kv.layers)):
            k = past_kv.layers[li].keys
            v = past_kv.layers[li].values
            bsz, n_heads, seq_len, head_dim = k.shape
            if seq_len <= TOTAL_BUDGET:
                continue
            device = k.device
            perm = torch.randperm(seq_len, device=device)[:TOTAL_BUDGET]
            keep_idx, _ = perm.unsqueeze(0).expand(n_heads, -1).sort(dim=1)
            g = keep_idx.unsqueeze(0).unsqueeze(-1).expand(bsz, -1, -1, head_dim)
            past_kv.layers[li].keys   = torch.gather(k, 2, g)
            past_kv.layers[li].values = torch.gather(v, 2, g)
        return output

    hook = model.model.register_forward_hook(eviction_hook)
    return hook, reset_scores


def restore_attn_forwards(model, orig_forwards):
    for layer_idx, layer in enumerate(model.model.layers):
        if layer_idx in orig_forwards:
            layer.self_attn.forward = orig_forwards[layer_idx]


class H2OSession:
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
        hf_token: str | None = None,
        device: str = "cuda",
        strategy: str = "per_head",
    ):
        self.model_name    = model_name
        self.hf_token      = hf_token or os.environ.get("HF_TOKEN") or None
        self.device        = device
        self.strategy      = strategy
        self.kv_mode       = "full"
        self.kv_cfg        = KV_CONFIGS["full"]
        self._hook         = None
        self._reset_scores = lambda: None
        self._orig_forwards = None

        print(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, token=self.hf_token
        )

        print(f"Loading model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            token=self.hf_token,
        ).to(device)
        self.model.eval()
        print(f"  Device: {next(self.model.parameters()).device}")
        print(f"  Layers: {len(self.model.model.layers)}")
        print("Ready.\n")

    def set_mode(self, mode: str, strategy: str | None = None):
        if mode not in KV_CONFIGS:
            raise ValueError(f"Unknown mode '{mode}'. Available: {sorted(KV_CONFIGS.keys())}")
        strat = strategy or self.strategy

        if self._hook is not None:
            self._hook.remove()
            self._hook = None
        if self._orig_forwards is not None:
            restore_attn_forwards(self.model, self._orig_forwards)
            self._orig_forwards = None

        self.kv_mode = mode
        self.kv_cfg  = KV_CONFIGS[mode]
        cfg = self.kv_cfg

        if cfg["h2o"]:
            self._hook, self._reset_scores, self._orig_forwards = apply_h2o(
                self.model,
                budget_ratio=cfg["budget_ratio"],
                recent_ratio=cfg["recent_ratio"],
                strategy=strat,
            )
        elif cfg.get("random", False):
            self._hook, self._reset_scores = apply_random(
                self.model, budget_ratio=cfg["budget_ratio"]
            )
        else:
            self._reset_scores = lambda: None

        print(f"Mode set: {mode}")

    def generate(
        self,
        text: str,
        reference: str = "",
        prompt_template: str = "Summarize this article in 2-3 sentences:\n\nArticle: {text}\n\nSummary:",
        max_new_tokens: int = 100,
        seed: int | None = None,
    ) -> dict:
        self._reset_scores()
        if seed is not None:
            torch.manual_seed(seed)

        prompt = prompt_template.format(text=text)
        inputs = self.tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=MAX_SEQ_LEN
        ).to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        prediction = self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

        rouge = compute_rouge_l(prediction, reference) if reference else 0.0
        return {
            "mode":       self.kv_mode,
            "prediction": prediction,
            "rouge_l":    rouge,
        }

    def compare_modes(
        self,
        text: str,
        reference: str = "",
        modes: list[str] | None = None,
        strategy: str | None = None,
        max_new_tokens: int = 100,
        seed: int = 0,
    ) -> list[dict]:
        if modes is None:
            modes = ["full", "random_b4", "h2o_b4_r2"]

        results = []
        for mode in modes:
            self.set_mode(mode, strategy=strategy)
            result = self.generate(
                text, reference=reference,
                max_new_tokens=max_new_tokens, seed=seed,
            )
            results.append(result)
            tag = f"[{mode:20s}]"
            rouge_str = f"ROUGE-L: {result['rouge_l']:.4f}" if reference else ""
            print(f"{tag} {rouge_str}")
            print(f"  {result['prediction'][:120]}")
            print()

        return results

    def cleanup(self):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None
        if self._orig_forwards is not None:
            restore_attn_forwards(self.model, self._orig_forwards)
            self._orig_forwards = None
        print("Session cleaned up.")
