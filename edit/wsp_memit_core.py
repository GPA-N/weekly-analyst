"""MEMIT-style, training-free edit of the draft format.

Where the model, given the correct prefix of a gold draft, ranks a structural token (a key name, a bracket, a quote) below another token,
we record the MLP input at that position (key k) and the residual shift that would raise the gold token by a margin (value, a single
gradient-direction estimate; no weight is ever trained). Then for each chosen layer the down projection gets the closed-form
least-squares update  dW = R^T K (K^T K + lam * n * C + eps I)^-1,  where C is the mean covariance of all MLP inputs on the calibration
prompts, so that other positions are disturbed as little as possible.

Exec'd (not imported) after wsp_main.py has defined m, tok, LM, NL, _ids, _orig, restore, DEV, generate, score_draft, passed.
"""
import re

import torch

LAYERSETS = {
    "l15_19": [15, 19],
    "late5": [18, 19, 20, 21, 22],
    "mid_late": [12, 14, 16, 18, 20, 22],
    "ww_hi": [16, 17, 18, 20, 21, 22],  # WeightWatcher alpha above 9 on down_proj
    "ww_lo": [9, 10, 11, 12, 13],  # WeightWatcher alpha 4.0 to 4.7
}
MARGIN = 3.0  # extra logits asked for beyond a tie
MAX_REL_SHIFT = 1.0  # cap on |delta| relative to the layer's residual norm


def _struct_chars(text):
    """True for characters outside JSON string values (keys, quotes, brackets, commas, whitespace)."""
    sm = [True] * len(text)
    for mo in re.finditer(r'"(?:[^"\\]|\\.)*"', text):
        a, b = mo.span()
        if not text[b:].lstrip().startswith(":"):
            for c in range(a + 1, b - 1):
                sm[c] = False
    return sm


def collect(tasks, layers):
    """Returns {layer: (K [n,3584], R [n,1024], cov_sum [3584,3584], count)} over the structural error positions of the gold drafts."""
    return collect_items([{"prompt": t["draft_prompt"], "comp": t["gold_draft"]} for t in tasks], layers)


def collect_items(items, layers):
    """Same as collect, for items {"prompt", "comp", optional "target": LongTensor of the wanted token at every completion position}.
    Without "target" the wanted token is the completion's own token (gold draft). With a target (on-policy: the student's own sampled
    text, target = a teacher's top-1 token) positions are corrected only where the student's top-1 differs from the target."""
    for p_ in m.parameters():
        p_.requires_grad_(False)
    keys = {l: [] for l in layers}
    vals = {l: [] for l in layers}
    cov = {l: torch.zeros(3584, 3584, device="cuda") for l in layers}
    cnt = {l: 0 for l in layers}
    n_err = n_skip = 0
    for t in items:
        prompt, comp = t["prompt"], t["comp"]
        ids, p = _ids(prompt, comp)
        text = comp.replace("<|im_end|>", "") + "<|im_end|>"
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        sm = _struct_chars(text)
        smask = torch.tensor([any(sm[c] for c in range(a, b)) if b > a else True for a, b in enc["offset_mapping"]], device="cuda")
        n = len(smask)
        emb = m.get_input_embeddings()(ids).detach().requires_grad_(True)
        acts = {}

        def mk(l):
            def hook(mod, inp):
                x = inp[0][0].detach().float()
                cov[l] += x.T @ x
                cnt[l] += x.shape[0]
                acts[l] = x[p - 1:p - 1 + n]
            return hook

        hooks = [LM.layers[l].mlp.down_proj.register_forward_pre_hook(mk(l)) for l in layers]
        oom = False
        try:
            out = m(inputs_embeds=emb, output_hidden_states=True)
        except torch.OutOfMemoryError:
            oom = True  # a very long prompt in the slow fallback attention path: skip this item, keep going
        finally:
            for h in hooks:
                h.remove()
        if oom:
            n_skip += 1
            torch.cuda.empty_cache()
            continue
        lg = out.logits[0, p - 1:p - 1 + n].float()
        gold = ids[0, p:p + n] if t.get("target") is None else t["target"].to(lg.device)
        top = lg.argmax(-1)
        wrong = (top != gold) & smask
        if int(wrong.sum()) == 0:
            del out, lg, emb
            torch.cuda.empty_cache()
            continue
        margin = lg.gather(1, gold[:, None])[:, 0] - lg.gather(1, top[:, None])[:, 0]
        hs = out.hidden_states
        for l in layers:
            hs[l + 1].retain_grad()
        try:
            margin[wrong].sum().backward()
        except torch.OutOfMemoryError:
            n_skip += 1
            del out, lg, emb, hs, margin
            torch.cuda.empty_cache()
            continue
        need = (MARGIN - margin[wrong]).clamp(min=0)
        for l in layers:
            g = hs[l + 1].grad[0, p - 1:p - 1 + n][wrong].float()
            hn = hs[l + 1][0, p - 1:p - 1 + n][wrong].float().norm(dim=-1, keepdim=True)
            d = need[:, None] * g / (g.pow(2).sum(-1, keepdim=True) + 1e-6)
            d = d * (hn * MAX_REL_SHIFT / (d.norm(dim=-1, keepdim=True) + 1e-6)).clamp(max=1.0)
            keys[l].append(acts[l][wrong])
            vals[l].append(d)
        n_err += int(wrong.sum())
        del out, lg, emb, hs
        torch.cuda.empty_cache()
    print("structural error positions", n_err, "over", len(items), "prompts;", n_skip, "skipped for memory", flush=True)
    empty = lambda d: torch.zeros(0, d, device="cuda")  # a group can have no wrong position at all (e.g. plans the model already gets right)
    return {l: (torch.cat(keys[l]) if keys[l] else empty(3584), torch.cat(vals[l]) if vals[l] else empty(1024), cov[l], cnt[l]) for l in layers}


@torch.no_grad()
def build_edit(coll, layers, lam, scale):
    dws = {}
    for l in layers:
        K, R, cov, cnt = coll[l]
        n = K.shape[0]
        A = K.T @ K
        B = lam * n * (cov / cnt)
        M = A + B
        M = M + 1e-3 * torch.diagonal(M).mean() * torch.eye(M.shape[0], device=M.device)
        dws[l] = torch.linalg.solve(M, K.T @ (R * (scale / len(layers)))).T
    return dws


@torch.no_grad()
def apply_dw(dws):
    rel = []
    for l, dW in dws.items():
        W = LM.layers[l].mlp.down_proj.weight
        if l not in _orig:
            _orig[l] = W.detach().clone()
        rel.append((dW.norm() / W.float().norm()).item())
        W.data.copy_((W.float() + dW).to(W.dtype))
    return round(sum(rel) / len(rel), 5)


def dev_drafts(label, tasks=None, max_new=700):
    tasks = DEV if tasks is None else tasks
    out = generate([t["draft_prompt"] for t in tasks], max_new=max_new, batch=8)
    sc = [score_draft(o, t["draft_prompt"]) for o, t in zip(out, tasks)]
    fails = [o for o, s in zip(out, sc) if not passed(s)]
    return {"label": label, "pass": sum(map(passed, sc)), "n": len(sc), "parts": {k: sum(s[k] for s in sc) for k in sc[0]},
            "fail_sample": (fails[0][:300] if fails else "")}


@torch.no_grad()
def struct_error_rate(tasks):
    """Teacher-forced tuning loss proxy: share of structural completion positions of the gold drafts where the model's top-1 token is not
    the gold token. Needs no generation. Compare it on CAL (the data the edit was fitted on) and DEV/HOLD: a big drop on CAL with
    little drop elsewhere is the overfit signal."""
    wrong = total = 0
    for t in tasks:
        ids, p = _ids(t["draft_prompt"], t["gold_draft"])
        text = t["gold_draft"].replace("<|im_end|>", "") + "<|im_end|>"
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        sm = _struct_chars(text)
        smask = torch.tensor([any(sm[c] for c in range(a, b)) if b > a else True for a, b in enc["offset_mapping"]], device="cuda")
        n = len(smask)
        top = m(input_ids=ids).logits[0, p - 1:p - 1 + n].argmax(-1)
        wrong += int(((top != ids[0, p:p + n]) & smask).sum())
        total += int(smask.sum())
    return round(wrong / max(total, 1), 4)
