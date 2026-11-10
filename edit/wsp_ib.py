"""InsightBench narrative check: insights -> summary, ROUGE-L against the reference summary. Needs globals m, tok, generate (wsp_lib)."""
import json, glob
from huggingface_hub import snapshot_download
from rouge_score import rouge_scorer

def build_ib_tasks(limit=100):
    root = snapshot_download("ServiceNow/insight_bench", repo_type="dataset", allow_patterns=["json/*.json"])
    tasks = []
    for f in sorted(glob.glob(root + "/json/flag-*.json"))[:limit]:
        j = json.load(open(f, encoding="utf-8"))
        ins, ref = j.get("insights") or [], j.get("summary")
        if not ins or not ref:
            continue
        goal = j["metadata"]["goal"]
        prompt = ("You are a business data analyst. Goal: " + goal + "\nFindings:\n" + "\n".join("- " + str(i) for i in ins)
                  + "\nWrite a short summary of these findings for a stakeholder, at most 120 words. Reply with the summary only.")
        tasks.append({"prompt": prompt, "ref": str(ref)})
    return tasks

def run_ib(tasks, label, max_new=220, batch=8):
    outs = generate([t["prompt"] for t in tasks], max_new, batch)
    sc = rouge_scorer.RougeScorer(["rougeL"])
    f = [sc.score(t["ref"], o)["rougeL"].fmeasure for t, o in zip(tasks, outs)]
    return {"label": label, "n": len(tasks), "rougeL": round(sum(f) / len(f), 4), "mean_words": round(sum(len(o.split()) for o in outs) / len(outs), 1)}, outs
