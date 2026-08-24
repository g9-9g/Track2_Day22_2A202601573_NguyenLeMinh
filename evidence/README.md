# Evidence — Day 22 Lab

## Kết quả A/B

Đã chạy đủ 50 QA pairs qua cả V1 và V2. V1 thắng về `faithfulness`
(0.9593 so với 0.8127) và `answer_relevancy` (0.9694 so với 0.8926), phù
hợp với prompt ngắn gọn, ít thêm chi tiết ngoài trọng tâm. V2 thắng nhẹ về
`context_precision` (0.9283 so với 0.9083), cho thấy cấu trúc phân tích giúp tận
dụng các đoạn context liên quan tốt hơn. Cả hai cùng đạt `context_recall = 1.0`.
Mục tiêu rubric được đạt vì faithfulness tốt nhất là 0.9593, cao hơn 0.8.

## Danh sách bằng chứng bắt buộc

- `01_langsmith_traces.png`: ảnh giao diện project `day22-lab`; trace có input,
  retriever output và LLM output. Phần chú thích ghi số đếm đọc trực tiếp qua
  LangSmith API: 570 root runs, 82 `rag-query` và 50 `ab-rag-query`.
- `02_prompt_hub.png`: ảnh giao diện Prompt Hub thật, hiển thị rõ hai prompt V1/V2,
  loại `ChatPromptTemplate`, visibility và commit tương ứng.
- `02_ab_routing_log.txt`: đủ 50 truy vấn có nhãn `prompt-v1`/`prompt-v2`, đồng
  thời ghi URL push và thao tác pull của cả hai prompt.
- `03_ragas_scores.png`: bảng so sánh điểm RAGAS từ lần chạy thật.
- `03_ragas_report.json`: bản sao báo cáo máy đọc được trong `data/`.
- `04_pii_demo_log.txt`: demo PII sạch và các loại email, điện thoại, SSN, thẻ.
- `04_json_demo_log.txt`: demo JSON hợp lệ, fences, nháy đơn, trailing comma và
  fallback.

## Prompt Hub

- `lab22-rag-prompt-v1`, commit `4ecf2e44`
- `lab22-rag-prompt-v2`, commit `0aa0ebc4`

Chi tiết URL Prompt Hub nằm ở đầu `02_ab_routing_log.txt` để người chấm đối
chiếu trực tiếp.

Đối chiếu đầy đủ từng tiêu chí chấm điểm: `RUBRIC_CHECKLIST.md`.
