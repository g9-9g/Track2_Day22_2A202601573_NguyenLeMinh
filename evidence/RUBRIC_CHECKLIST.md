# Rubric checklist — Day 22

## Nhiệm vụ 1 — 25/25

- [x] 1.1: Knowledge base được chia thành chunks và index bằng FAISS trong
  `src/utils/data_loader.py` và `src/01_langsmith_rag_pipeline.py`.
- [x] 1.2: LCEL chain đúng thứ tự retriever → prompt → LLM →
  `StrOutputParser`.
- [x] 1.3: `@traceable(name="rag-query")`; LangSmith API xác nhận 82 root runs
  mang tên `rag-query` (yêu cầu tối thiểu 50).
- [x] 1.4: ảnh `01_langsmith_traces.png` thể hiện input/output và các child run
  retriever; log chạy nằm ở `01_rag_pipeline_log.txt`.

## Nhiệm vụ 2 — 25/25

- [x] 2.1: V1 ngắn gọn 2–4 câu; V2 phân tích có cấu trúc 3–5 câu.
- [x] 2.2: ảnh `02_prompt_hub.png` hiển thị rõ hai prompt, mỗi prompt 1 commit.
- [x] 2.3: log xác nhận cả hai prompt được pull từ Hub trước khi chạy.
- [x] 2.4: routing tất định bằng MD5 của `request_id`.
- [x] 2.5: đủ 50 nhãn routing: V1=19, V2=31; LangSmith có 50 root runs
  `ab-rag-query`.

## Nhiệm vụ 3 — 25/25, cộng 2 điểm phân tích

- [x] 3.1: đủ 50 QA cho cả V1 và V2 (100 RAG outputs).
- [x] 3.2: `EvaluationDataset` dùng đúng `SingleTurnSample` với
  `user_input`, `response`, `retrieved_contexts`, `reference`.
- [x] 3.3: đủ `faithfulness`, `answer_relevancy`, `context_recall`,
  `context_precision`; checkpoint xác nhận 50 điểm/metric/version.
- [x] 3.4: faithfulness tốt nhất 0.9593, đạt yêu cầu >= 0.8.
- [x] 3.5: `data/ragas_report.json` và bản sao evidence đều tồn tại.
- [x] Phân tích V1/V2 nằm trong `evidence/README.md`.
- [ ] Thưởng faithfulness >= 0.9 cho cả hai: V2 đạt 0.8127 nên không nhận 3 điểm này.

## Nhiệm vụ 4 — 25/25

- [x] 4.1: custom `PIIDetector` dùng `@register_validator`.
- [x] 4.2: regex phát hiện email, phone, SSN và credit card.
- [x] 4.3: `PIIDetector(on_fail=OnFailAction.FIX)` trả chuỗi đã redact.
- [x] 4.4: 6 test case PII, gồm clean và multi-PII.
- [x] 4.5: custom `JSONFormatter` kiểm tra bằng `json.loads`.
- [x] 4.6: sửa markdown fences, single quotes và trailing comma.
- [x] 4.7: trả JSON fallback khi không thể sửa.
- [x] 4.8: 5 test case JSON bao phủ toàn bộ trường hợp rubric.

## Evidence và nộp bài

- [x] Đủ 7 file evidence bắt buộc.
- [x] `evidence/README.md` có phân tích A/B.
- [x] GitHub repository là public.
- [x] Không commit `.env`; quét source không phát hiện API key thật.
- [ ] LangSmith project hiện vẫn private; cần bật public/share nếu cổng chấm yêu
  cầu người chấm truy cập không cần tài khoản organization.
- [ ] Các thay đổi hiện tại cần được commit và push trước khi nộp.
