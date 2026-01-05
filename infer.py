#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ProteinGym-style zero-shot inference with LucaOne (local weights or auto-download).

This script reads one or more CSV files containing deep mutational scanning (DMS)
variants and evaluates how well LucaOne zero-shot scores correlate with the
experimental DMS measurements.

Expected input columns (per row / variant):
  - `mutant`: mutation string in the common format like "A123G"
              (wildtype AA, 1-based position, mutant AA).
  - `mutated_sequence`: the full *mutant* protein sequence.
  - `DMS_score`: the experimental fitness/score for this variant.

For each variant, we:
  1) Recover the wildtype sequence by replacing the mutated residue back to the
     wildtype amino acid at the given position.
  2) Create a masked sequence by replacing that position with "[MASK]".
  3) Run LucaOne to obtain log-probabilities at the masked position.
  4) Compute a delta log-probability ("masked marginal"):
        Δ = log P(mutant_aa | context) - log P(wildtype_aa | context)
  5) Save per-variant scores and a Spearman correlation summary.

Outputs:
  - For each input CSV: a new CSV with a `lucaone_delta_logp` column.
  - A `summary.csv` aggregating per-file Spearman statistics.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import torch

try:
    import pandas as pd
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Missing dependency: pandas."
    ) from e

try:
    from scipy.stats import spearmanr
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Missing dependency: scipy."
    ) from e


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
SRC_DIR = THIS_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


try:
    from src.embedding.get_embedding import load_model as luca_load_model
    from src.models.alphabet import Alphabet
    from src.utils import download_trained_checkpoint_lucaone
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Failed to import LucaOne code. Ensure this script is run from the LucaOne code directory and "
        "that `src/` is present."
    ) from e


try:
    from importlib.metadata import distributions
except Exception:  # pragma: no cover

    def distributions():  # type: ignore[override]
        return []


REQUIRED_COLS = {"mutant", "mutated_sequence", "DMS_score"}
AA20 = "ACDEFGHIKLMNPQRSTVWY"


def compute_spearman(pred_scores, true_scores) -> tuple[float | None, float | None]:
    rho, pval = spearmanr(pred_scores, true_scores, nan_policy="omit")
    rho_val = None if rho is None or (isinstance(rho, float) and math.isnan(rho)) else float(rho)
    pval_val = None if pval is None or (isinstance(pval, float) and math.isnan(pval)) else float(pval)
    return rho_val, pval_val


def _fmt_float(x: float | None, *, fmt: str) -> str:
    return "nan" if x is None else format(x, fmt)


def collect_installed_packages() -> list[str]:
    items: list[str] = []
    for dist in distributions():
        name = None
        try:
            name = dist.metadata.get("Name")
        except Exception:
            name = None
        if not name:
            continue
        items.append(f"{name}=={dist.version}")
    return sorted(set(items), key=str.lower)


def print_runtime_environment() -> None:
    print("========== Runtime ==========")
    print(f"Python:        {sys.version.replace(os.linesep, ' ')}")
    print(f"Executable:    {sys.executable}")
    print(f"Platform:      {sys.platform}")
    print("Packages:")
    for item in collect_installed_packages():
        print(f"  - {item}")
    print("=============================\n")


def parse_mutant(mut_str: str) -> tuple[str, int, str]:
    wt_aa = mut_str[0]
    mut_aa = mut_str[-1]
    pos1 = int(mut_str[1:-1])
    return wt_aa, pos1, mut_aa


def recover_wt_sequence(mut_seq: str, wt_aa: str, pos1: int) -> str:
    return mut_seq[: pos1 - 1] + wt_aa + mut_seq[pos1:]


def resolve_csv_paths(*, data_dir: Path, csv: str | None) -> list[Path]:
    if csv is None:
        return sorted(p for p in data_dir.glob("*.csv") if p.is_file())

    candidate = Path(csv)
    if not candidate.is_absolute():
        candidate = (data_dir / candidate).resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"CSV not found: {candidate}")
    return [candidate]


def _checkpoint_paths(
    *,
    model_root: Path,
    llm_type: str,
    llm_version: str,
    llm_step: str,
) -> tuple[Path, Path]:
    log_path = model_root / "logs" / llm_type / llm_version / "logs.txt"
    ckpt_dir = model_root / "models" / llm_type / llm_version / f"checkpoint-step{llm_step}"
    return log_path, ckpt_dir


def load_lucaone_for_zeroshot(
    *,
    model_root: Path,
    llm_type: str,
    llm_version: str,
    llm_step: str | None,
    base_url: str,
) -> tuple[dict, object, torch.nn.Module, Alphabet, str]:
    download_trained_checkpoint_lucaone(
        llm_dir=str(model_root),
        llm_type=llm_type,
        llm_version=llm_version,
        llm_step=llm_step,
        base_url=base_url,
    )
    resolved_step = llm_step
    if not resolved_step:
        resolved_step = "36000000" if llm_version == "lucaone" else ("36800000" if llm_version == "lucaone-gene" else "30000000")
    log_path, ckpt_dir = _checkpoint_paths(model_root=model_root, llm_type=llm_type, llm_version=llm_version, llm_step=resolved_step)
    args_info, model_config, model, tokenizer = luca_load_model(str(log_path), str(ckpt_dir), embedding_inference=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return args_info, model_config, model, tokenizer, device


def preprocess_dataset(
    *,
    df: pd.DataFrame,
    progress_every: int,
) -> tuple[list[str], list[str], list[int], list[str], list[float]]:
    masked_seqs: list[str] = []
    wt_aas: list[str] = []
    mut_aas: list[str] = []
    pos1s: list[int] = []
    true_scores: list[float] = []

    for i, row in enumerate(df.itertuples(index=False), start=1):
        mut_str = row.mutant
        mut_seq = row.mutated_sequence
        dms = row.DMS_score

        wt_aa, pos1, mut_aa = parse_mutant(mut_str)
        wt_seq = recover_wt_sequence(mut_seq=mut_seq, wt_aa=wt_aa, pos1=pos1)

        if mut_seq[pos1 - 1] != mut_aa:
            raise ValueError(f"Mut AA mismatch in {mut_str}: sequence has {mut_seq[pos1-1]}")

        masked_seq = wt_seq[: pos1 - 1] + "[MASK]" + wt_seq[pos1:]
        masked_seqs.append(masked_seq)
        wt_aas.append(wt_aa)
        pos1s.append(pos1)
        mut_aas.append(mut_aa)
        true_scores.append(dms)

        if progress_every > 0 and i % progress_every == 0:
            print(f"  preprocessed {i}/{len(df)}")

    return masked_seqs, wt_aas, pos1s, mut_aas, true_scores


def _encode_batch_prot(
    *,
    tokenizer: Alphabet,
    model_config,
    args_info: dict,
    masked_seqs: list[str],
) -> dict[str, torch.Tensor]:
    encoded_list: list[list[int]] = []
    max_len = 0
    for seq in masked_seqs:
        ids = tokenizer.encode(seq_type="prot", seq=seq)
        if args_info.get("max_length"):
            ids = ids[: int(args_info["max_length"])]
        encoded_list.append(ids)
        max_len = max(max_len, len(ids))

    prepend = int(getattr(tokenizer, "prepend_bos", True))
    append = int(getattr(tokenizer, "append_eos", True))
    processed_len = max_len + prepend + append

    input_ids = torch.empty((len(masked_seqs), processed_len), dtype=torch.int64)
    input_ids.fill_(tokenizer.padding_idx)

    position_ids = None
    if not getattr(model_config, "no_position_embeddings", True):
        position_ids = torch.empty((len(masked_seqs), processed_len), dtype=torch.int64)
        position_ids.fill_(tokenizer.padding_idx)

    token_type_ids = None
    if not getattr(model_config, "no_token_type_embeddings", False):
        token_type_ids = torch.empty((len(masked_seqs), processed_len), dtype=torch.int64)
        token_type_ids.fill_(tokenizer.padding_idx)

    for i, ids in enumerate(encoded_list):
        if prepend:
            input_ids[i, 0] = tokenizer.cls_idx
        seq_tensor = torch.tensor(ids, dtype=torch.int64)
        start = prepend
        input_ids[i, start : start + len(ids)] = seq_tensor
        end = start + len(ids)
        if append:
            input_ids[i, end] = tokenizer.eos_idx
            end += 1

        if position_ids is not None:
            position_ids[i, :end] = torch.arange(0, end, dtype=torch.int64)
        if token_type_ids is not None:
            token_type_ids[i, :end] = 1

    batch: dict[str, torch.Tensor] = {
        "input_ids_b": input_ids,
        "return_dict": True,
        "output_keys_b": {"token_level": {"mlm"}},
    }
    if position_ids is not None:
        batch["position_ids_b"] = position_ids
    if token_type_ids is not None:
        batch["token_type_ids_b"] = token_type_ids
    return batch


@torch.no_grad()
def debug_alignment_lucaone(
    *,
    model,
    tokenizer: Alphabet,
    model_config,
    args_info: dict,
    device: str,
    mutant: str,
    mutated_sequence: str,
    token_window: int,
) -> None:
    wt_aa, pos1, mut_aa = parse_mutant(mutant)
    print("\n========== Alignment Debug (LucaOne) ==========")
    print(f"mutant parsed: (wt_aa={wt_aa!r}, pos1={pos1}, mut_aa={mut_aa!r})")
    print(f"mutated_sequence[pos1-1] == mut_aa ? {mutated_sequence[pos1-1] == mut_aa}  ({mutated_sequence[pos1-1]!r})")

    wt_seq = recover_wt_sequence(mut_seq=mutated_sequence, wt_aa=wt_aa, pos1=pos1)
    print(f"recovered_wt[pos1-1] == wt_aa ?     {wt_seq[pos1-1] == wt_aa}  ({wt_seq[pos1-1]!r})")

    masked_seq = wt_seq[: pos1 - 1] + "[MASK]" + wt_seq[pos1:]
    batch = _encode_batch_prot(tokenizer=tokenizer, model_config=model_config, args_info=args_info, masked_seqs=[masked_seq])
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    input_ids = batch["input_ids_b"][0].tolist()
    tokens = [tokenizer.get_tok(i) for i in input_ids]
    print("\ninput_ids -> tokens:")
    print(f"token[0]: {tokens[0]!r}  (expected [CLS])")
    print(f"candidate token indices: pos1-1={pos1-1}, pos1={pos1}")
    if 0 <= pos1 - 1 < len(tokens):
        print(f"token[pos1-1]: {tokens[pos1-1]!r}")
    if 0 <= pos1 < len(tokens):
        print(f"token[pos1]:   {tokens[pos1]!r}  (expected [MASK])")
    lo = max(0, pos1 - token_window)
    hi = min(len(tokens), pos1 + token_window + 1)
    print("\ntoken window around pos1:")
    for i in range(lo, hi):
        mark = "<==" if i == pos1 else "   "
        print(f"  {i:5d}: {tokens[i]!r} {mark}")

    wt_id = tokenizer.get_idx(wt_aa)
    mut_id = tokenizer.get_idx(mut_aa)
    print("\nVocab alignment:")
    print(f"wt_token_id={wt_id}  id->token={tokenizer.get_tok(wt_id)!r}")
    print(f"mut_token_id={mut_id} id->token={tokenizer.get_tok(mut_id)!r}")

    out = model(**batch)
    logits = out.outputs_b["token_level"]["mlm"][0, pos1, :]
    probs = torch.softmax(logits, dim=-1)
    print("\nLogits alignment (mask site):")
    print(
        "logits stats:",
        f"min={logits.min().item():.4g}",
        f"max={logits.max().item():.4g}",
        f"mean={logits.mean().item():.4g}",
        f"std={logits.std().item():.4g}",
    )
    print(f"logits[wt_id]={logits[wt_id].item():.4g}  p(wt)={probs[wt_id].item():.4g}")
    print(f"logits[mut_id]={logits[mut_id].item():.4g} p(mut)={probs[mut_id].item():.4g}")

    aa_ids = [tokenizer.get_idx(aa) for aa in AA20]
    aa_probs = probs[torch.tensor(aa_ids, device=probs.device)].detach().cpu().tolist()
    aa_rank = sorted(zip(AA20, aa_probs), key=lambda x: x[1], reverse=True)[:10]
    print("top-10 AA (within 20):")
    for aa, p in aa_rank:
        print(f"  {aa}\t{p:.4g}")

    print("============================================\n")


@torch.no_grad()
def score_delta_logp(
    *,
    model,
    tokenizer: Alphabet,
    model_config,
    args_info: dict,
    device: str,
    batch_size: int,
    masked_seqs: list[str],
    wt_aas: list[str],
    pos1s: list[int],
    mut_aas: list[str],
    progress_every: int,
) -> list[float]:
    if not (len(masked_seqs) == len(wt_aas) == len(pos1s) == len(mut_aas)):
        raise ValueError("Input list lengths must match: masked_seqs, wt_aas, pos1s, mut_aas")

    deltas: list[float] = []
    total = len(masked_seqs)
    processed = 0

    for start in range(0, total, batch_size):
        end = min(total, start + batch_size)
        batch_seqs = masked_seqs[start:end]

        batch = _encode_batch_prot(tokenizer=tokenizer, model_config=model_config, args_info=args_info, masked_seqs=batch_seqs)
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

        out = model(**batch)
        logits = out.outputs_b["token_level"]["mlm"]  # (B, L, vocab)

        for j in range(len(batch_seqs)):
            pos1 = int(pos1s[start + j])
            if pos1 <= 0 or pos1 >= logits.shape[1]:
                raise ValueError(f"pos1 out of range for sample {start+j}: pos1={pos1}, logits_len={logits.shape[1]}")

            site_logits = logits[j, pos1, :]
            log_probs = torch.log_softmax(site_logits, dim=-1)
            wt_id = tokenizer.get_idx(wt_aas[start + j])
            mut_id = tokenizer.get_idx(mut_aas[start + j])
            delta = (log_probs[mut_id] - log_probs[wt_id]).item()
            deltas.append(float(delta))

        processed += (end - start)
        if progress_every > 0 and processed % progress_every == 0:
            print(f"  scored {processed}/{total}")

    return deltas


def run_one_csv(
    *,
    csv_path: Path,
    output_dir: Path,
    output_suffix: str,
    model,
    tokenizer: Alphabet,
    model_config,
    args_info: dict,
    device: str,
    batch_size: int,
    progress_every: int,
    debug_alignment: bool,
    debug_rows: int,
    debug_token_window: int,
) -> dict:
    print(f"Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    if not REQUIRED_COLS.issubset(df.columns):
        missing = sorted(REQUIRED_COLS.difference(df.columns))
        raise ValueError(f"Missing required columns in {csv_path.name}: {missing}")

    masked_seqs, wt_aas, pos1s, mut_aas, true_scores = preprocess_dataset(df=df, progress_every=progress_every)

    if debug_alignment:
        n = min(len(df), max(1, int(debug_rows)))
        for i in range(n):
            debug_alignment_lucaone(
                model=model,
                tokenizer=tokenizer,
                model_config=model_config,
                args_info=args_info,
                device=device,
                mutant=str(df.loc[i, "mutant"]),
                mutated_sequence=str(df.loc[i, "mutated_sequence"]),
                token_window=max(1, int(debug_token_window)),
            )

    print("Running zero-shot predictions...")
    pred_scores = score_delta_logp(
        model=model,
        tokenizer=tokenizer,
        model_config=model_config,
        args_info=args_info,
        device=device,
        batch_size=batch_size,
        masked_seqs=masked_seqs,
        wt_aas=wt_aas,
        pos1s=pos1s,
        mut_aas=mut_aas,
        progress_every=progress_every,
    )
    df["lucaone_delta_logp"] = pred_scores

    rho, pval = compute_spearman(pred_scores, true_scores)

    out_csv = output_dir / f"{csv_path.stem}{output_suffix}"
    df.to_csv(out_csv, index=False)

    print("\n========== ProteinGym zero-shot ==========")
    print("Model:        LucaOne")
    print(f"CSV:          {csv_path.name}")
    print(f"Variants:     {len(df)}")
    print(f"Spearman ρ:   {_fmt_float(rho, fmt='.4f')}")
    print(f"P-value:      {_fmt_float(pval, fmt='.2e')}")
    print(f"Saved to:     {out_csv}")
    print("==========================================\n")

    return {
        "model": f"{args_info.get('model_type', 'lucaone')}:{args_info.get('num_hidden_layers', '')}",
        "csv": csv_path.name,
        "variants": int(len(df)),
        "spearman_rho": rho,
        "p_value": pval,
        "output_csv": out_csv.name,
        "score_column": "lucaone_delta_logp",
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input_csv",
        default=None,
        help="Only process this CSV (basename under data_dir, or an absolute path).",
    )
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--progress_every", type=int, default=100, help="Print progress every N variants (0 disables).")

    p.add_argument("--llm_type", default="lucaone")
    p.add_argument("--llm_version", default="lucaone-prot", choices=["lucaone", "lucaone-gene", "lucaone-prot"])
    p.add_argument("--llm_step", default=None, help="Checkpoint step (default depends on llm_version).")
    p.add_argument("--download_base_url", default="http://47.93.21.181/lucaone/TrainedCheckPoint/latest/")

    p.add_argument("--data_dir", default="/opt/ml/processing/input/data")
    p.add_argument("--model_dir", default="/opt/ml/processing/input/model")
    p.add_argument("--output_dir", default="/opt/ml/processing/output")
    p.add_argument("--output_suffix", default="_lucaone_zeroshot.csv")

    p.add_argument("--debug_alignment", action="store_true", help="Print alignment diagnostics for the first rows.")
    p.add_argument("--debug_rows", type=int, default=1, help="How many rows to debug when --debug_alignment is set.")
    p.add_argument("--debug_token_window", type=int, default=3, help="Token window size around the site.")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    print_runtime_environment()

    csv_paths = resolve_csv_paths(data_dir=data_dir, csv=args.input_csv)
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found under: {data_dir}")

    args_info, model_config, model, tokenizer, device = load_lucaone_for_zeroshot(
        model_root=model_dir,
        llm_type=str(args.llm_type),
        llm_version=str(args.llm_version),
        llm_step=None if args.llm_step in (None, "", "None") else str(args.llm_step),
        base_url=str(args.download_base_url),
    )

    summary: list[dict] = []
    for csv_path in csv_paths:
        summary.append(
            run_one_csv(
                csv_path=csv_path,
                output_dir=output_dir,
                output_suffix=args.output_suffix,
                model=model,
                tokenizer=tokenizer,
                model_config=model_config,
                args_info=args_info,
                device=device,
                batch_size=max(1, int(args.batch_size)),
                progress_every=max(0, int(args.progress_every)),
                debug_alignment=bool(args.debug_alignment),
                debug_rows=max(1, int(args.debug_rows)),
                debug_token_window=max(1, int(args.debug_token_window)),
            )
        )

    summary_path = output_dir / "summary.csv"
    pd.DataFrame(summary).to_csv(summary_path, index=False)
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
