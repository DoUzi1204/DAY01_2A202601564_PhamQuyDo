# K4 — Ngày 1: Bài Tập & Phản Ánh

## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 14h00–18h00
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature

Gọi `call_openai` với temperature 0.0, 0.7, 1.2 và 1.8 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Hà Nội."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi? Ở mức nào phản hồi bắt đầu
kém mạch lạc?** (2–3 câu)

> Ở temperature 0.0, phản hồi mang tính xác định cao, chính xác và nhất quán (nói về 36 phố phường). Ở mức 0.7, câu trả lời tự nhiên và chọn chủ đề độc đáo hơn (xóm đường tàu). Mức 1.2 bắt đầu mở rộng nhưng xuất hiện chi tiết chưa hoàn toàn chuẩn xác. Khi lên tới 1.8, phân bố xác suất token bị làm phẳng quá mức khiến phản hồi dễ bị lan man hoặc kém mạch lạc giữa các từ ngữ.

### Câu 1.2 — Chọn temperature cho sản phẩm

**Bạn sẽ đặt temperature bao nhiêu cho trợ lý soạn thảo hợp đồng pháp lý,
và bao nhiêu cho trợ lý viết slogan quảng cáo? Giải thích khác biệt.**

> Tôi sẽ đặt temperature = 0.0 (hoặc 0.1) cho trợ lý hợp đồng pháp lý nhằm đảm bảo tính chính xác tuyệt đối, nhất quán và tránh việc mô hình tự bịa đặt thông tin gây rủi ro pháp lý. Ngược lại, tôi sẽ đặt temperature = 0.8–1.0 cho trợ lý viết slogan để kích thích sự sáng tạo, đa dạng phong cách từ vựng và tạo ra nhiều ý tưởng độc đáo khác biệt.

### Câu 1.3 — Đánh đổi chi phí

Kịch bản: 20.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 2 lần,
mỗi lần trung bình ~500 token đầu ra.

**Ước tính chi phí mỗi ngày của model lớn so với model nhỏ cho workload này
(dựa trên bảng giá trong template). Nêu một trường hợp model lớn xứng đáng
với chi phí và một trường hợp model nhỏ là lựa chọn đúng:**

> Với workload 20.000.000 token đầu ra/ngày: Model lớn (GPT-4o) tốn ~$200/ngày, trong khi model nhỏ (GPT-4o-mini) chỉ tốn ~$12/ngày (chênh lệch gần 17 lần). Model lớn xứng đáng chi phí trong bài toán tư vấn pháp lý/y tế hoặc phân tích logic phức tạp cần độ chính xác cao. Model nhỏ là lựa chọn đúng cho tác vụ đơn giản như phân loại ý kiến khách hàng, tóm tắt văn bản ngắn hoặc chatbot FAQ định hướng.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona

Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích máy học (machine learning) là gì?"** nhưng hai system prompt
khác nhau:

- "Bạn là một nhà thơ, trả lời mọi thứ bằng hình ảnh ví von, tránh thuật ngữ."
- "Bạn là kỹ sư phần mềm senior, trả lời chính xác, có ví dụ code khi phù hợp."

**Hai phản hồi khác nhau như thế nào (giọng văn, độ dài, mức kỹ thuật)?
Từ đó rút ra system prompt điều khiển được những khía cạnh nào của phản hồi?**
(3–4 câu)

> Phản hồi của nhà thơ mang giọng văn tự sự mộc mạc dạng vần thơ, dùng hình ảnh ví von (cánh đồng cỏ, giọt mưa) và hoàn toàn không chứa thuật ngữ. Phản hồi của kỹ sư senior mang tính kỹ thuật cao, phân loại rõ ràng các nhóm thuật toán và đính kèm code minh họa. Từ đó rút ra System Prompt giúp điều khiển:(1) Giọng văn và vai trò của trợ lý (Tone of voice), (2) Độ sâu kiến thức/ngôn ngữ chuyên ngành, và (3) Cấu trúc/định dạng đầu ra của phản hồi (thơ, đoạn văn hay khối mã lệnh code).

### Câu 2.2 — tiktoken vs đếm từ

Chọn một đoạn văn tiếng Việt ~150 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Nếu dùng ước lượng thô để dự
toán ngân sách API cho ứng dụng tiếng Việt, bạn sẽ dự toán thiếu hay thừa —
và vì sao?**

> Qua thực nghiệm với đoạn văn tiếng Việt 197 từ, ước lượng thô (`số từ / 0.75`) thu được 262.7 tokens, trong khi tiktoken đếm thực tế qua hàm `count_tokens` (dùng model gpt-4o) là 268 tokens — chênh lệch thực tế khoảng +2.0%. Nếu dùng ước lượng thô cho ứng dụng tiếng Việt, chúng ta vẫn sẽ dự toán THIẾU ngân sách, vì các từ tiếng Việt có dấu thường bị băm thành nhiều sub-word tokens hơn so với tiếng Anh.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming

**Xét ba ứng dụng: (a) chatbot văn bản, (b) trợ lý giọng nói đọc to phản hồi,
(c) pipeline dịch tài liệu chạy ngầm ban đêm. Ứng dụng nào hưởng lợi nhiều
nhất từ streaming, ứng dụng nào không cần — và tại sao?** (1 đoạn văn)

> Ứng dụng chatbot văn bản và trợ lý giọng nói hưởng lợi nhiều nhất từ streaming vì giúp người dùng nhận phản hồi ngay lập tức, đồng thời hệ thống giọng nói có thể đọc luôn từng chunk đầu tiên. Ngược lại, pipeline dịch tài liệu chạy ngầm ban đêm không cần streaming vì đây là tác vụ xử lý hàng loạt không tương tác trực tiếp với người dùng, hệ thống chỉ quan tâm đến file dịch hoàn chỉnh cuối cùng.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?

**Khi API quá tải và hàng nghìn client cùng retry, exponential backoff giúp
gì so với delay cố định? Tra cứu thêm: kỹ thuật "jitter" (thêm độ trễ ngẫu
nhiên) giải quyết vấn đề gì còn sót lại?**

> Exponential backoff giúp dãn cách thời gian thử lại theo cấp số nhân (0.1s, 0.2s, 0.4s...), giúp làm giảm mật độ yêu cầu dồn dập lên server đang quá tải thay vì lặp lại truy vấn với nhịp cố định. Kỹ thuật "jitter" (thêm độ trễ ngẫu nhiên) giúp làm lệch pha thời điểm retry của hàng nghìn client khác nhau, tránh hiện tượng các client ngẫu nhiên trùng thời điểm gửi yêu cầu gây nghẽn lại server.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona

**Viết lại system prompt bạn dùng cho trợ lý của mình. Chỉ ra 2 chỗ trong
prompt mà nếu xóa đi, hành vi trợ lý sẽ thay đổi rõ rệt — và mô tả thay đổi
đó:**

> System prompt dùng cho trợ lý: "Bạn là trợ giảng thân thiện của khóa AI, trả lời ngắn gọn bằng tiếng Việt dưới 3 câu và luôn đính kèm 1 ví dụ minh họa."
> Chỗ 1 (Ràng buộc "dưới 3 câu"): Nếu xóa đi, trợ lý sẽ trả lời dài dòng thành nhiều đoạn văn phân tích chi tiết (tốn nhiều token hơn hẳn).
> Chỗ 2 (Ràng buộc "luôn đính kèm 1 ví dụ minh họa"): Nếu xóa đi, trợ lý chỉ giải thích định nghĩa lý thuyết thuần túy mà hoàn toàn không đưa ra ví dụ tình huống thực tế nào.

### Câu 4.2 — Hạn chế & cải thiện

**Trợ lý của bạn giữ history 4 lượt cuối. Hãy mô tả một tình huống hội thoại
cụ thể mà giới hạn này khiến trợ lý trả lời sai/mất ngữ cảnh, và đề xuất một
cách khắc phục (ví dụ: tóm tắt các lượt cũ, tăng giới hạn có chọn lọc...):**

> Tình huống thực tế: Ở lượt 1, người dùng khai báo "Tôi tên là An, đang làm chatbot bán hàng cho cửa hàng quần áo X". Sau 5 lượt trao đổi tiếp theo về streaming và retry, ở lượt 6 người dùng hỏi "Tên tôi là gì và tôi đang làm ứng dụng gì?". Vì history chỉ giữ 4 lượt gần nhất (8 messages), thông tin ở lượt 1 đã bị xóa khỏi bộ nhớ khiến trợ lý trả lời mất ngữ cảnh và không thể nhớ tên người dùng.

> Cách khắc phục: Triển khai kỹ thuật Conversation Summary Memory — dùng một model nhỏ (như gpt-4o-mini) tự động tóm tắt các thông tin mấu chốt của các lượt chat cũ bị cắt bỏ thành 1 câu tóm tắt ngữ cảnh (ví dụ: "User tên An, làm chatbot shop X") và đính kèm cố định đoạn tóm tắt này vào System Prompt ở tất cả các lượt về sau.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên GitHub cá nhân và nộp link repo vào vlearn (theo hướng dẫn README)
