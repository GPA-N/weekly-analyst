"""SWEET / loop-direction style weight edit on the language model's down_proj columns. Exec on the VM after wsp_lib.py and wsp_gold.py.

direction d_l: variance-scaled mean over pairs of (mean residual of the failing completion - mean residual of the correct one) at layer l.
attribution of MLP column j at layer l: (d_l . W[:, j]) * mean_t a_j(t) over the failing tokens. Edit: W[:, j] -= alpha * d_l (d_l . W[:, j]) for the top-k columns with positive attribution.
"""
import torch

LM = m.model.language_model
NL = len(LM.layers)
_orig = {}


def _ids(prompt, completion):
    ptxt = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
    pids = tok(ptxt, return_tensors="pt").input_ids
    cids = tok(completion.replace("<|im_end|>", "") + "<|im_end|>", return_tensors="pt", add_special_tokens=False).input_ids
    return torch.cat([pids, cids], 1).cuda(), pids.shape[1]


@torch.no_grad()
def trace(prompt, completion):
    ids, p = _ids(prompt, completion)
    acts = {}
    hooks = [LM.layers[l].mlp.down_proj.register_forward_pre_hook(
        lambda mod, inp, l=l: acts.__setitem__(l, inp[0][0, p:].float().mean(0))) for l in range(NL)]
    try:
        out = m(input_ids=ids, output_hidden_states=True)
    finally:
        for h in hooks:
            h.remove()
    hs = torch.stack([out.hidden_states[l + 1][0, p:].float().mean(0) for l in range(NL)])
    return hs, torch.stack([acts[l] for l in range(NL)])


def directions(pairs, lam=0.1):
    """pairs: list of (prompt, good_text, bad_text). Returns unit directions (NL x 1024) and mean failing activations (NL x 3584)."""
    diffs, acts = [], []
    for prompt, good, bad in pairs:
        hg, _ = trace(prompt, good)
        hb, ab = trace(prompt, bad)
        diffs.append(hb - hg)
        acts.append(ab)
    D = torch.stack(diffs)
    mu, sd = D.mean(0), D.std(0) + 1e-6
    v = mu / (sd + lam * sd.median(dim=-1, keepdim=True).values)
    return v / v.norm(dim=-1, keepdim=True), torch.stack(acts).mean(0)


def restore():
    for l, w in _orig.items():
        LM.layers[l].mlp.down_proj.weight.data.copy_(w)


@torch.no_grad()
def apply_edit(d, abar, layers, k, alpha=1.0):
    """Returns the number of changed weights."""
    changed = 0
    for l in layers:
        W = LM.layers[l].mlp.down_proj.weight
        if l not in _orig:
            _orig[l] = W.detach().clone()
        Wf = W.float()
        proj = d[l].to(Wf.device) @ Wf
        score = proj * abar[l].to(Wf.device)
        top = torch.topk(score, k).indices
        top = top[score[top] > 0]
        W.data[:, top] = (Wf[:, top] - alpha * torch.outer(d[l].to(Wf.device), proj[top])).to(W.dtype)
        changed += int(top.numel()) * W.shape[0]
    return changed


def random_directions(seed=0):
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(NL, 1024, generator=g)
    return r / r.norm(dim=-1, keepdim=True)
