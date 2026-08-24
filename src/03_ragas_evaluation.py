"""
Bước 3 — RAGAS Evaluation
===========================
NHIỆM VỤ:
  1. Chạy 50 QA pairs qua CẢ 2 prompt version, lưu answers + contexts
  2. Tạo EvaluationDataset với các SingleTurnSample object
  3. Đánh giá với 4 RAGAS metrics: faithfulness, answer_relevancy,
     context_recall, context_precision
  4. In bảng so sánh V1 vs V2
  5. Lưu kết quả vào data/ragas_report.json

DELIVERABLE: faithfulness ≥ 0.8 cho ít nhất 1 prompt version
             + file data/ragas_report.json được tạo ra

⏰ LƯU Ý: Bước này mất ~15-30 phút. Hãy bắt đầu sớm!
"""
import sys
import json
import copy
import os
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config  # ⚠️ phải import trước LangChain

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from ragas.run_config import RunConfig

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import QA_PAIRS

# Một số Gemini free-tier models không hỗ trợ n>1 candidates. Một candidate
# vẫn là cấu hình RAGAS hợp lệ và giúp phép chấm có thể chạy nhất quán.
answer_relevancy.strictness = 1


# ── 1. Prompt Templates (copy từ Bước 2) ──────────────────────────────────
# DONE: Copy SYSTEM_V1 và SYSTEM_V2 mà bạn đã viết ở file 02_prompt_hub_ab_routing.py
SYSTEM_V1 = (
    "Bạn là trợ lý AI thân thiện. Chỉ trả lời dựa trên context được cung cấp. "
    "Trả lời ngắn gọn, rõ ràng trong 2-4 câu; nếu thiếu thông tin, hãy nói thẳng là không biết. "
    "Context:\n{context}"
)
PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human",  "{question}"),
])

SYSTEM_V2 = (
    "Bạn là chuyên gia phân tích thông tin. Chỉ sử dụng các dữ kiện trong context. "
    "Hãy trả lời có cấu trúc trong 3-5 câu: nêu kết luận chính, giải thích bằng dữ kiện liên quan, "
    "và nêu mức độ chắc chắn. Không suy đoán khi context không đủ. Context:\n{context}"
)
PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human",  "{question}"),
])

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2}
METRIC_NAMES = [
    "faithfulness",
    "answer_relevancy",
    "context_recall",
    "context_precision",
]
RAGAS_METRICS = [faithfulness, answer_relevancy, context_recall, context_precision]


# ── 2. Setup Vectorstore ───────────────────────────────────────────────────
def setup_vectorstore():
    """Tái sử dụng — tạo FAISS vectorstore từ knowledge base."""
    embeddings  = get_embeddings()
    text        = load_knowledge_base()
    chunks      = split_text(text, chunk_size=600, chunk_overlap=50)
    return build_vectorstore(chunks, embeddings)


# ── 3. Chạy RAG và thu thập kết quả ───────────────────────────────────────
def run_rag(retriever, llm, prompt, question: str) -> dict:
    """
    Chạy RAG chain cho 1 câu hỏi.

    ⚠️ QUAN TRỌNG: trả về contexts là LIST of strings, KHÔNG phải string đã ghép!
    RAGAS cần từng đoạn riêng để tính context_recall và context_precision.

    Trả về: {"answer": str, "contexts": list[str]}
    """
    # DONE: Retrieve documents từ retriever
    docs = retriever.invoke(question)

    # DONE: Tạo contexts là danh sách page_content (KHÔNG ghép chuỗi ở đây)
    # Gợi ý: contexts = [doc.page_content for doc in docs]
    contexts = [doc.page_content for doc in docs]

    # DONE: Ghép contexts thành 1 string để truyền vào {context} của prompt
    ctx_str = "\n\n".join(contexts)

    # DONE: Chạy chain (prompt | llm | StrOutputParser()).invoke(...)
    answer = (prompt | llm | StrOutputParser()).invoke({
        "context": ctx_str,
        "question": question,
    })

    # DONE: Trả về dict với answer và contexts (list)
    return {"answer": answer, "contexts": contexts}


def collect_rag_outputs(vectorstore, prompt_version: str) -> list:
    """
    Chạy tất cả 50 QA pairs qua prompt version được chỉ định.
    Trả về: list of dict với keys: question, reference, answer, contexts
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm       = get_llm()
    prompt    = PROMPTS[prompt_version]

    checkpoint_path = (
        Path(__file__).parent.parent / "data" / f"rag_outputs_{prompt_version}.json"
    )
    results = []
    if checkpoint_path.exists():
        try:
            cached = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if isinstance(cached, list):
                results = cached
                print(f"♻️  Tiếp tục từ checkpoint: {len(results)}/50 câu")
        except (json.JSONDecodeError, OSError):
            print("⚠️  Checkpoint lỗi, bắt đầu lại từ đầu.")

    completed_questions = {r["question"] for r in results}
    print(f"\n🚀 Đang chạy 50 câu hỏi với prompt {prompt_version} ...")

    for i, qa in enumerate(QA_PAIRS, 1):
        if qa["question"] in completed_questions:
            print(f"  [{i:02d}/50] Đã có checkpoint: {qa['question'][:45]}")
            continue

        # DONE: Gọi run_rag() cho câu hỏi hiện tại
        out = run_rag(retriever, llm, prompt, qa["question"])

        # DONE: Append vào results dict với 4 keys
        results.append({
            "question":  qa["question"],
            "reference": qa["reference"],
            "answer":    out["answer"],
            "contexts":  out["contexts"],
        })
        checkpoint_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"  [{i:02d}/50] {qa['question'][:60]}")

    return results


# ── 4. Tạo RAGAS EvaluationDataset ────────────────────────────────────────
def build_ragas_dataset(rag_results: list) -> EvaluationDataset:
    """
    Chuyển đổi kết quả RAG thành RAGAS EvaluationDataset.

    Mỗi SingleTurnSample cần 4 trường:
      user_input         → câu hỏi
      response           → câu trả lời đã tạo
      retrieved_contexts → list[str] các đoạn đã retrieve
      reference          → đáp án chuẩn (ground truth)
    """
    # DONE: Tạo list các SingleTurnSample từ rag_results
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["reference"],
        )
        for r in rag_results
    ]

    # DONE: Wrap thành EvaluationDataset và trả về
    return EvaluationDataset(samples=samples)


# ── 5. Chạy RAGAS Evaluation ──────────────────────────────────────────────
def run_ragas_eval(rag_results: list, version: str) -> dict:
    """
    Đánh giá kết quả RAG với 4 RAGAS metrics.
    Trả về: dict {metric_name: mean_score}

    Lưu ý: evaluate() thực hiện rất nhiều lần gọi LLM → mất 5-10 phút / version.
    """
    print(f"\n📐 Đang đánh giá RAGAS cho prompt {version} ... (vui lòng chờ ~5-10 phút)")

    # Chấm theo nhóm nhỏ và checkpoint điểm thô. Cách này vẫn gọi đủ bốn
    # metrics cho mọi sample, đồng thời không mất hàng giờ tiến độ nếu free-tier
    # quota hoặc kết nối Gemini bị ngắt giữa chừng.
    checkpoint_path = (
        Path(__file__).parent.parent / "data" / f"ragas_metrics_{version}.json"
    )
    # Có thể dùng một LLM local nhẹ chỉ cho phần judge, trong khi embeddings
    # vẫn theo PROVIDER chính. Điều này tránh phải dựng lại FAISS/index.
    eval_provider = (os.getenv("RAGAS_EVAL_PROVIDER") or config.PROVIDER).lower()
    evaluator_id = (
        f"ollama:{config.OLLAMA_MODEL}"
        if eval_provider == "ollama"
        else config.GEMINI_MODEL if eval_provider == "gemini" else eval_provider
    )
    state = {
        "scores": {name: [None] * len(rag_results) for name in METRIC_NAMES},
        "attempts": [0] * len(rag_results),
        "evaluator": evaluator_id,
    }
    if checkpoint_path.exists():
        try:
            cached = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if (
                isinstance(cached, dict)
                and set(cached.get("scores", {})) == set(METRIC_NAMES)
                and all(len(cached["scores"][name]) == len(rag_results) for name in METRIC_NAMES)
                and len(cached.get("attempts", [])) == len(rag_results)
            ):
                state = cached
                previous_evaluator = state.get("evaluator")
                if previous_evaluator != evaluator_id:
                    for i in range(len(rag_results)):
                        if any(state["scores"][name][i] is None for name in METRIC_NAMES):
                            state["attempts"][i] = 0
                    print(
                        f"🔄 Đổi evaluator {previous_evaluator or 'unknown'} → {evaluator_id}; "
                        "reset retry cho các điểm còn thiếu."
                    )
                state["evaluator"] = evaluator_id
                complete = sum(
                    all(state["scores"][name][i] is not None for name in METRIC_NAMES)
                    for i in range(len(rag_results))
                )
                print(f"♻️  Tiếp tục metric checkpoint: {complete}/{len(rag_results)} mẫu hoàn chỉnh")
        except (json.JSONDecodeError, OSError, TypeError):
            print("⚠️  Metric checkpoint lỗi, chấm lại từ đầu.")

    while True:
        pending = [
            i for i in range(len(rag_results))
            if state["attempts"][i] < 3
            and any(state["scores"][name][i] is None for name in METRIC_NAMES)
        ]
        if not pending:
            break

        indices = pending[:5]
        dataset = build_ragas_dataset([rag_results[i] for i in indices])
        print(f"  → Chấm samples {indices[0] + 1}-{indices[-1] + 1} ...")
        # RAGAS tạo event loop riêng cho mỗi evaluate(). Dùng client và metric
        # instances mới để không tái sử dụng async gRPC channel đã bị đóng.
        llm_eval = get_llm(
            provider=eval_provider,
            temperature=0,
            max_tokens=2048 if eval_provider == "ollama" else 8192,
            json_mode=True,
            gemini_openai_compat=eval_provider == "gemini",
        )
        emb_eval = get_embeddings()
        chunk_metrics = copy.deepcopy(RAGAS_METRICS)
        result = evaluate(
            dataset,
            metrics=chunk_metrics,
            llm=llm_eval,
            embeddings=emb_eval,
            run_config=RunConfig(
                timeout=300,
                max_retries=1,
                max_wait=10,
                max_workers=1 if eval_provider == "ollama" else 4,
            ),
            batch_size=4 if eval_provider == "ollama" else 20,
        )

        made_progress = False
        for local_index, source_index in enumerate(indices):
            sample_progress = False
            for name in METRIC_NAMES:
                value = result[name][local_index]
                if value is not None and np.isfinite(float(value)):
                    state["scores"][name][source_index] = float(value)
                    made_progress = True
                    sample_progress = True
            if sample_progress:
                state["attempts"][source_index] += 1

        checkpoint_path.write_text(
            json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if not made_progress:
            raise RuntimeError(
                "Evaluator không trả về điểm nào; có thể model đã hết quota. "
                "Checkpoint đã được giữ lại để tiếp tục bằng model khác."
            )

    scores = {}
    for name in METRIC_NAMES:
        numeric = np.asarray(
            [value for value in state["scores"][name] if value is not None],
            dtype=float,
        )
        if numeric.size == 0:
            raise RuntimeError(f"Không có điểm hợp lệ cho metric {name}.")
        scores[name] = float(np.nanmean(numeric))
        missing = len(rag_results) - numeric.size
        if missing:
            print(f"  ⚠️  {name}: {missing} mẫu lỗi parser sau 3 lần thử")

    # In kết quả
    print(f"\n📊 Kết quả RAGAS — Prompt {version.upper()}:")
    for k, v in scores.items():
        star = " ⭐" if k == "faithfulness" and v >= 0.8 else ""
        print(f"  {k:30s}: {v:.4f}{star}")

    return scores


# ── 6. Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Bước 3: RAGAS Evaluation")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    root = Path(__file__).parent.parent
    checkpoint_paths = {
        version: root / "data" / f"rag_outputs_{version}.json"
        for version in ("v1", "v2")
    }
    cached_results = {}
    for version, path in checkpoint_paths.items():
        if path.exists():
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, list) and len(loaded) == len(QA_PAIRS):
                cached_results[version] = loaded

    if len(cached_results) == 2:
        print("♻️  Đã tải đủ 100 outputs từ checkpoint; bỏ qua bước gọi RAG.")
        v1_results = cached_results["v1"]
        v2_results = cached_results["v2"]
    else:
        vectorstore = setup_vectorstore()
        config.wait_for_gemini_embedding_quota()
        v1_results = collect_rag_outputs(vectorstore, "v1")
        v2_results = collect_rag_outputs(vectorstore, "v2")

    # Chạy RAGAS evaluation
    v1_scores = run_ragas_eval(v1_results, "v1")
    v2_scores = run_ragas_eval(v2_results, "v2")

    # In bảng so sánh
    print("\n" + "=" * 65)
    print(f"  {'Metric':30s}  {'V1':>8}  {'V2':>8}  Winner")
    print("=" * 65)
    for metric in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        s1, s2  = v1_scores[metric], v2_scores[metric]
        winner  = "← V1" if s1 > s2 else "← V2"
        print(f"  {metric:30s}  {s1:>8.4f}  {s2:>8.4f}  {winner}")

    # Kiểm tra mục tiêu
    best_faith = max(v1_scores["faithfulness"], v2_scores["faithfulness"])
    if best_faith >= 0.8:
        print(f"\n✅ Đạt mục tiêu: faithfulness = {best_faith:.4f} ≥ 0.8")
    else:
        print(f"\n⚠️  Chưa đạt mục tiêu ({best_faith:.4f} < 0.8).")
        print("   Gợi ý: giảm chunk_size, tăng k, hoặc điều chỉnh prompt.")

    # DONE: Lưu báo cáo vào data/ragas_report.json
    report = {
        "prompt_v1_scores": v1_scores,
        "prompt_v2_scores": v2_scores,
        "target_met": best_faith >= 0.8,
    }
    report_path = Path(__file__).parent.parent / "data" / "ragas_report.json"
    # DONE: Ghi report vào file bằng json.dumps hoặc json.dump
    # Gợi ý: report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"💾 Đã lưu báo cáo vào {report_path}")


if __name__ == "__main__":
    main()
