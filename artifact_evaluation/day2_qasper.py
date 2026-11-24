#!/usr/bin/env python3
"""External Day-2 QASPER v0.3 preparation, run, and reporting kit.

Requires only Python standard library. Never writes into a project source tree.
Model runs are performed by the existing app.benchmark against a separate APP_DATA_DIR.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import random
import re
import statistics
import subprocess
import sys
import tarfile
import urllib.request

URL = "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz"
# Source: official HF allenai/qasper dataset_infos.json, version 0.3.0.
ARCHIVE_SHA256 = "a28fdf966db827bcee3d873107d6b6669864fb7ca8fbf73a192f5e39191bdb5a"
EXPECTED_DEV_BASENAME = "qasper-dev-v0.3.json"
SCRIPT_DIR = Path(__file__).resolve().parent


def dump_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(data: bytes):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_download(destination: Path, url: str = URL):
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.name + ".download")
    print(f"Downloading official QASPER v0.3: {url}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Day2-QASPER-Project-Evaluation/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=80) as response, tmp.open("wb") as target:
            while chunk := response.read(1024 * 1024):
                target.write(chunk)
        tmp.replace(destination)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return destination


def load_dev_json(args, output_dir: Path):
    if args.dev_json is not None:
        p = args.dev_json.resolve()
        if not p.is_file():
            raise FileNotFoundError(p)
        raw = p.read_bytes()
        print(f"Using supplied Development JSON: {p}")
        return raw, {"source_kind": "local_dev_json", "path": str(p), "dev_sha256": sha256(raw), "archive_sha256": None}
    archive = args.archive.resolve() if args.archive else (output_dir / "qasper-train-dev-v0.3.tgz")
    if not archive.is_file():
        safe_download(archive)
    got = sha256_file(archive)
    if got != ARCHIVE_SHA256:
        raise ValueError(f"Official QASPER archive SHA-256 mismatch: got {got}; expected {ARCHIVE_SHA256}. Delete and re-download.")
    print("Official archive SHA-256: VERIFIED")
    with tarfile.open(archive, mode="r:gz") as tar:
        dev_files = [m for m in tar.getmembers() if m.isfile() and Path(m.name).name == EXPECTED_DEV_BASENAME]
        if len(dev_files) != 1:
            raise ValueError(f"Expected one {EXPECTED_DEV_BASENAME} in official archive; got {len(dev_files)}")
        file = tar.extractfile(dev_files[0])
        if file is None:
            raise ValueError("Unable to read official development member")
        raw = file.read()
    return raw, {"source_kind": "verified_official_archive", "url": URL,
                 "archive_sha256": got, "dev_sha256": sha256(raw), "archive_path": str(archive)}


def norm(text):
    return " ".join(str(text).split())


def is_gold_answer(annotation):
    a = annotation.get("answer", annotation)
    return bool(a.get("unanswerable") or a.get("extractive_spans") or a.get("free_form_answer") or a.get("yes_no") is not None)


def paragraph_inventory(record):
    texts = []
    full_text = record.get("full_text", [])
    if isinstance(full_text, dict):
        full_text = [{"section_name": n, "paragraphs": ps} for n, ps in zip(full_text.get("section_name", []), full_text.get("paragraphs", []))]
    for si, sec in enumerate(full_text):
        ps = sec.get("paragraphs", [])
        if isinstance(ps, str):
            ps = [ps]
        for pi, para in enumerate(ps):
            s = norm(para)
            if s:
                texts.append((s, f"s{si}-p{pi}"))
    return texts


def map_gold_evidence(qa, paragraphs):
    # Mirrors the production QASPER paragraph mapping (exact or substring match).
    lookup = defaultdict(list)
    for txt, pid in paragraphs:
        lookup[txt].append(pid)
    refs = []
    for ann in qa.get("answers", []):
        ans = ann.get("answer", ann)
        for evidence in ans.get("evidence", []) or []:
            s = norm(evidence)
            matches = list(lookup.get(s, ()))
            if not matches and s:
                matches = [pid for t, pid in paragraphs if s in t or t in s]
            refs.extend(matches)
    return list(dict.fromkeys(refs))


def seeded_rank(seed: int, key: str):
    return hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()


def select_papers(dev: dict, count: int, per_paper: int, seed: int):
    candidates = []
    for pid, doc in dev.items():
        if not isinstance(doc, dict) or not isinstance(doc.get("qas"), list):
            continue
        paras = paragraph_inventory(doc)
        total_chars = sum(len(txt) for txt, _ in paras)
        if len(paras) < 8 or total_chars < 2500:
            continue
        qs = []
        for qa in doc["qas"]:
            if len(str(qa.get("question", "")).strip()) < 12:
                continue
            if not qa.get("answers") or not any(is_gold_answer(a) for a in qa["answers"]):
                continue
            refs = map_gold_evidence(qa, paras)
            qs.append({"qa": qa, "refs": refs, "question_id": str(qa.get("question_id", ""))})
        if len(qs) < per_paper:
            continue
        mapped_count = sum(bool(x["refs"]) for x in qs)
        # Modest mapping requirement makes evidence F1 usable, without selecting based on the model's answers.
        if mapped_count < min(2, per_paper):
            continue
        candidates.append((str(pid), doc, qs, total_chars, mapped_count))
    if len(candidates) < count:
        raise ValueError(f"Only {len(candidates)} eligible QASPER papers; requested {count}. Try --papers 2 or --questions-per-paper 4.")
    # Stable pseudo-random selection among eligible official dev papers. Avoid alphabetical or performance-based cherry-picking.
    candidates.sort(key=lambda x: seeded_rank(seed, x[0]))
    output = {}
    manifest = []
    for pid, doc, qs, chars, mapped_count in candidates[:count]:
        # Prefer evidence-mappable questions for interpretable Evidence F1; disclose this filter.
        qs.sort(key=lambda x: (not bool(x["refs"]), seeded_rank(seed, x["question_id"] or str(x["qa"]["question"]))))
        selected = qs[:per_paper]
        copy = dict(doc)
        copy["qas"] = [q["qa"] for q in selected]
        output[pid] = copy
        manifest.append({"paper_id": f"qasper-{pid}", "dataset_paper_id": pid,
                         "title": doc.get("title", ""), "full_text_chars": chars,
                         "total_candidate_questions": len(qs), "candidate_mapped_questions": mapped_count,
                         "questions": [{"question_id": q["question_id"],
                                        "question": q["qa"]["question"],
                                        "gold_evidence_refs": q["refs"],
                                        "gold_mapping_available": bool(q["refs"])} for q in selected]})
    return output, manifest, len(candidates)


def cmd_prepare(args):
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    raw, provenance = load_dev_json(args, out)
    dataset = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(dataset, dict) or len(dataset) < args.papers:
        raise ValueError("Expected QASPER raw mapping {paper_id: paper_record}; received incompatible JSON")
    selected, papers, candidates = select_papers(dataset, args.papers, args.questions_per_paper, args.seed)
    selected_path = out / "qasper_selected_dev.json"
    manifest_path = out / "selection_manifest.json"
    dump_json(selected_path, selected)
    manifest = {"dataset": "QASPER", "version": "0.3", "split": "dev",
                "created_at_utc": datetime.now(timezone.utc).isoformat(), "seed": args.seed,
                "papers_requested": args.papers, "questions_per_paper": args.questions_per_paper,
                "eligible_papers": candidates, "total_questions": sum(len(p["questions"]) for p in papers),
                "sampling_policy": f"Stable seed hash among dev papers with >={args.questions_per_paper} QA, >=min(2, {args.questions_per_paper}) paragraph-mappable gold; within each paper prefer paragraph-mappable questions; no model outputs used for selection.",
                "source": provenance, "selected_json_sha256": sha256_file(selected_path), "papers": papers}
    dump_json(manifest_path, manifest)
    print(f"Prepared {len(papers)} dev-source papers / {manifest['total_questions']} questions (source: {provenance['source_kind']})")
    for paper in papers:
        print(f" - {paper['dataset_paper_id']} | {paper['title'][:72]} | mapped={sum(q['gold_mapping_available'] for q in paper['questions'])}/{len(paper['questions'])}")
    print(f"Subset:   {selected_path}\nManifest: {manifest_path}")
    return manifest_path


def cmd_run(args):
    backend = args.backend.resolve()
    if not (backend / "app" / "benchmark.py").exists() or not (backend / "app" / "cli.py").exists():
        raise FileNotFoundError(f"Not a Paper Tutor backend directory: {backend}")
    dataset_dir = args.dataset_dir.resolve()
    selected_json = dataset_dir / "qasper_selected_dev.json"
    manifest_path = dataset_dir / "selection_manifest.json"
    if not selected_json.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Run prepare first to create selected_dev and manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256_file(selected_json) != manifest["selected_json_sha256"]:
        raise ValueError("Selected dataset changed after preparation. Prepare again before running.")
    if manifest["total_questions"] != manifest["papers_requested"] * manifest["questions_per_paper"]:
        raise ValueError("Manifest question counts inconsistent")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    results = out / f"qasper_dev_{args.trial}.json"
    if results.exists() and not args.force:
        raise FileExistsError(f"Output already exists; choose --trial trial2 (preferred) or --force: {results}")
    # Separate database entirely outside source project, keyed to immutable chosen question set.
    db_path = (args.runtime_dir or SCRIPT_DIR / "runtime" / manifest["selected_json_sha256"][:12]).resolve()
    env = dict(os.environ)
    env["APP_DATA_DIR"] = str(db_path)
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    if args.model:
        env["MODEL_NAME"] = args.model
    env["ENABLE_RERANK"] = "false"  # Fair, low-cost local RAG comparison
    for mode in ("basic", "hybrid", "adaptive"):
        if mode not in ("basic", "hybrid", "adaptive"):
            raise ValueError(mode)
    cmd_import = [sys.executable, "-m", "app.cli", str(selected_json), "--split", "dev"]
    cmd_benchmark = [sys.executable, "-m", "app.benchmark", "--split", "dev",
                     "--limit", str(manifest["total_questions"]), "--modes", "basic", "hybrid", "adaptive",
                     "--output", str(results)]
    print(f"Backend:    {backend}\nRuntime DB: {db_path}\nOutput:     {results}", flush=True)
    print("Experiment: Basic / Hybrid / Adaptive; use_memory=False, teaching=False, allow_web=False; no reranker; one at a time", flush=True)
    print("Importing selected QASPER dev papers ...", flush=True)
    subprocess.run(cmd_import, cwd=backend, env=env, check=True)
    print("Executing real-model benchmark ...", flush=True)
    subprocess.run(cmd_benchmark, cwd=backend, env=env, check=True)
    print(f"Done. Generate comparison: python day2_qasper.py report --results \"{results}\" --dataset-dir \"{dataset_dir}\" --output-dir \"{out}\"", flush=True)


def as_num(v):
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def fmt(val, d=3):
    return "—" if val is None else f"{val:.{d}f}"


def mean(vals):
    vals = [x for x in vals if x is not None]
    return statistics.mean(vals) if vals else None


def aggregate(rows, modes):
    out = []
    for mode in modes:
        these = [r for r in rows if r.get("mode") == mode]
        nonfailed = [r for r in these if r.get("status") in ("succeeded", "partial") and str(r.get("answer", "")).strip()]
        eligible = [r for r in these if r.get("gold_mapping_available") and as_num((r.get("evidence_metrics") or {}).get("f1")) is not None]
        out.append({
            "mode": mode,
            "n": len(these),
            "successful": sum(r.get("status") == "succeeded" for r in these),
            "partial": sum(r.get("status") == "partial" for r in these),
            "failed": sum(r.get("status") not in ("succeeded", "partial") for r in these),
            "nonempty_rate": sum(bool(str(r.get("answer") or "").strip()) for r in these) / len(these) if these else None,
            "answer_f1_including_failures": mean([as_num(r.get("answer_f1")) or 0.0 for r in these]),
            "answer_f1_completed_only": mean([as_num(r.get("answer_f1")) for r in nonfailed]),
            "evidence_f1_mapped_only": mean([as_num((r.get("evidence_metrics") or {}).get("f1")) for r in eligible]),
            "evidence_eligible_n": len(eligible),
            "latency_s_mean": mean([as_num((r.get("metrics") or {}).get("total_ms")) / 1000 if as_num((r.get("metrics") or {}).get("total_ms")) is not None else None for r in these]),
            "llm_calls_mean": mean([as_num((r.get("metrics") or {}).get("llm_calls")) for r in these]),
            "retrieval_rounds_mean": mean([as_num((r.get("metrics") or {}).get("retrieval_rounds")) for r in these]),
        })
    return out


def cmd_report(args):
    results_path = args.results.resolve()
    manifest = json.loads((args.dataset_dir.resolve() / "selection_manifest.json").read_text(encoding="utf-8"))
    data = json.loads(results_path.read_text(encoding="utf-8"))
    rows = data.get("results", [])
    if not isinstance(rows, list):
        raise ValueError("Benchmark results missing results array")
    expected = {q["question_id"] for p in manifest["papers"] for q in p["questions"]}
    modes = ["basic", "hybrid", "adaptive"]
    seen = Counter((r.get("question_id"), r.get("mode")) for r in rows)
    missing = [(q, mode) for q in sorted(expected) for mode in modes if (q, mode) not in seen]
    duplicates = [(k, n) for k, n in seen.items() if n > 1]
    extras = [r.get("question_id") for r in rows if r.get("question_id") not in expected]
    aggregates = aggregate(rows, modes)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    q2paper = {q["question_id"]: p for p in manifest["papers"] for q in p["questions"]}
    overview = {"dataset": "QASPER", "split": "dev", "dataset_version": "0.3",
                "selection_manifest": str((args.dataset_dir / "selection_manifest.json").resolve()),
                "benchmark_file": str(results_path), "model": data.get("model"),
                "expected_runs": len(expected) * len(modes), "observed_runs": len(rows),
                "missing": missing, "duplicates": duplicates, "unexpected_ids": extras,
                "by_mode": aggregates, "notes": [
                    "Answer F1 is the project's simplified token-overlap proxy, NOT the official QASPER score.",
                    "Evidence F1 excludes unmappable gold cases; recall can be optimistic with multiple retrieval rounds.",
                    "Latency includes model/retrieval overhead and may include first-use embedding initialization.",
                    "The sampled papers/questions were selected with a reproducible eligibility and evidence-mapping filter; do not claim random representativeness of full QASPER."]}
    dump_json(out / "metrics_summary.json", overview)
    csvpath = out / "per_question_results.csv"
    with csvpath.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["paper_id", "paper_title", "question_id", "mode", "status", "answer_f1_proxy", "evidence_f1", "gold_mapping_available", "latency_s", "llm_calls", "retrieval_rounds", "question", "answer", "error"])
        for r in rows:
            m = r.get("metrics") or {}
            paper = q2paper.get(r.get("question_id"))
            gold = next((q for q in paper["questions"] if q["question_id"] == r.get("question_id")), None) if paper else None
            writer.writerow([paper["paper_id"] if paper else "", paper["title"] if paper else "", r.get("question_id"), r.get("mode"), r.get("status"), r.get("answer_f1"), (r.get("evidence_metrics") or {}).get("f1"), r.get("gold_mapping_available"), (as_num(m.get("total_ms")) or 0)/1000, m.get("llm_calls"), m.get("retrieval_rounds"), gold["question"] if gold else "", r.get("answer", ""), r.get("error", "")])
    lines = ["# QASPER v0.3 Development: 三模式对比实验", "", f"- 实测模型：`{data.get('model', '?')}`", f"- 论文数：{len(manifest['papers'])}，问题数：{len(expected)}", f"- 预期调用：{len(expected)*3}，已有：{len(rows)}", "- 设置：Basic/Hybrid/Adaptive；无教学 Skill、Memory、Web；单任务串行执行；关闭 Rerank", "", "## 总体指标", "", "| 模式 | 成功/总数 | 非空率 | Answer F1（含失败）| Evidence F1（可映射）| Evidence样本 | 平均耗时(s) | 平均LLM调用 | 平均检索轮次 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|" ]
    for a in aggregates:
        lines.append(f"| {a['mode']} | {a['successful']}/{a['n']} | {fmt(a['nonempty_rate'])} | {fmt(a['answer_f1_including_failures'])} | {fmt(a['evidence_f1_mapped_only'])} | {a['evidence_eligible_n']} | {fmt(a['latency_s_mean'],2)} | {fmt(a['llm_calls_mean'],2)} | {fmt(a['retrieval_rounds_mean'],2)} |")
    lines += ["", "## 实验限制与判断原则", "", "1. Answer Token F1 是项目简化代理指标，不是官方 QASPER leaderboard 分数。", "2. 无法映射 Gold evidence 的问题不纳入 Evidence F1 平均值；注意记录 eligible 样本数。", "3. 多轮 Adaptive 的累计检索段落范围不与单轮固定 K 的 Recall@K 直接等价。", "4. 一次运行的延迟易受本地 GPU、缓存和 Embedding 冷启动影响，不作严格速度结论。", "5. 样本是固定 Seed 的符合条件子集，偏向能映射证据的题目，不代表整体数据集。", "6. 论文导入为原始 QASPER JSON，未将全文转换为 PDF。", ""]
    if missing or duplicates or extras:
        lines += ["## 数据完整性警告", "", f"- 缺失：{len(missing)}；重复：{len(duplicates)}；未知 question ID：{len(extras)}。", ""]
    lines += ["## 分论文问答结果", ""]
    for p in manifest["papers"]:
        lines += [f"### {p['title']}", "", "| 问题 | Basic F1 | Hybrid F1 | Adaptive F1 |", "|---|---:|---:|---:|"]
        for q in p["questions"]:
            results_by_mode = {r["mode"]: r for r in rows if r.get("question_id") == q["question_id"]}
            f1s = [fmt(as_num(results_by_mode.get(m, {}).get("answer_f1"))) for m in modes]
            qt = q["question"].replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {qt[:100]} | {f1s[0]} | {f1s[1]} | {f1s[2]} |")
        lines.append("")
    report = out / "comparison_report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Summary: {out / 'metrics_summary.json'}\nReport:  {report}\nDetails: {csvpath}")
    for a in aggregates:
        print(f"{a['mode']:>8}  succeeded={a['successful']:>2}/{a['n']}   answer_f1={fmt(a['answer_f1_including_failures'])}   evidence_f1={fmt(a['evidence_f1_mapped_only'])}   latency={fmt(a['latency_s_mean'],2)}s")


def build_parser():
    p = argparse.ArgumentParser(description="External QASPER v0.3: prepare / run / report; no changes to the course source project")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prepare", help="Download SHA-verified official QASPER dev and choose 3 papers x 5 questions")
    src = a.add_mutually_exclusive_group()
    src.add_argument("--archive", type=Path, help="Previously downloaded official train-dev-v0.3.tgz")
    src.add_argument("--dev-json", type=Path, help="Previously extracted qasper-dev-v0.3.json")
    a.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "dataset")
    a.add_argument("--papers", type=int, default=3)
    a.add_argument("--questions-per-paper", type=int, default=5)
    a.add_argument("--seed", type=int, default=6000)
    a.set_defaults(fn=cmd_prepare)
    b = sub.add_parser("run", help="Run project's own benchmark in isolated experiment data directory")
    b.add_argument("--backend", type=Path, required=True, help="Actual inner project backend folder")
    b.add_argument("--dataset-dir", type=Path, default=SCRIPT_DIR / "dataset")
    b.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "results")
    b.add_argument("--runtime-dir", type=Path, default=None, help="Override isolated APP_DATA_DIR")
    b.add_argument("--trial", default="trial1", help="Unique name e.g. trial2")
    b.add_argument("--model", default=None, help="Override MODEL_NAME; defaults to project setting")
    b.add_argument("--force", action="store_true", help="Overwrite existing result; prefer a new trial")
    b.set_defaults(fn=cmd_run)
    c = sub.add_parser("report", help="Summarize existing project benchmark JSON; create Markdown/CSV/JSON")
    c.add_argument("--results", type=Path, required=True)
    c.add_argument("--dataset-dir", type=Path, default=SCRIPT_DIR / "dataset")
    c.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "results")
    c.set_defaults(fn=cmd_report)
    return p


def main():
    args = build_parser().parse_args()
    try:
        args.fn(args)
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
