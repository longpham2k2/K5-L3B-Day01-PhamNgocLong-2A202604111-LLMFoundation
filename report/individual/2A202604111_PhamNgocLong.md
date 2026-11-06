# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202604111
- **Họ và tên:** Phạm Ngọc Long
- **SEED** (in ở ô đầu notebook): 9493 → đề thử thách **#2**

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ (10/10 check TODO 1.1 → 4.4 đạt, đủ 10 dự đoán) |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☐ *(TODO: ảnh hiện tại là trang giấy trống, cần chụp lại trang có lời giải)* |

Dữ liệu sau khi bỏ trùng: 5.169 tin (653 spam, 4.516 ham) · train 3.619 (455 spam) · val 775 · test 775 (95 spam) · SUB 40 tin (20/20).

## 2. Kết quả chính

| Cách tiếp cận | Precision | Recall | F1 (test) | F1 (SUB 40) | p95 ms/tin | tin/s (cả lô) | USD / 1k tin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1. BoW + LogReg (tự code) | 1.000 | 0.747 | 0.855 | 0.889 | 14.94 | 306 | 0 |
| 2. BoW + MLP | 0.965 | 0.874 | 0.917 | 0.889 | **7.20** | 394 | 0 |
| 3a. MiniLM + kNN | 0.909 | 0.947 | 0.928 | **0.923** | 107.02 | 15 | 0 |
| 3b. MiniLM + LogReg | 0.989 | 0.926 | **0.957** | 0.919 | 40.35 | 53 | 0 |
| 3c. MiniLM + MLP head | 0.989 | 0.926 | **0.957** | 0.919 | 32.06 | 53 | 0 |
| 4a. LLM nhỏ zero-shot (gpt-6-luna) | 0.000 | 0.000 | — | 0.000 ⚠️ | 3447.14 | 0.84 | 0.00495 |
| 4b. LLM lớn zero-shot (gpt-6.1-terra) | 0.000 | 0.000 | — | 0.000 ⚠️ | 1306.90 | 0.98 | 0.04950 |
| 4c. LLM nhỏ few-shot | 0.000 | 0.000 | — | 0.000 ⚠️ | 1254.47 | 1.22 | 0.01588 |

- Đường Pareto (theo 1 − F1 SUB và p95): **BoW + MLP → MiniLM + MLP head → MiniLM + kNN**.
- Đạt SLA p95 < 50 ms: BoW + LogReg, BoW + MLP, MiniLM + LogReg, MiniLM + MLP head.
- ⚠️ **Hàng LLM không phản ánh chất lượng của model**: cả 3 lần chạy đều có `invalid=40/40`, tức `parse_label` không đọc được câu trả lời nào, nên mọi tin bị gán mặc định là `ham` (accuracy đúng bằng 0.5 trên SUB cân bằng). Lệnh thử "Say hello" với `max_tokens=20` thì trả lời bình thường, còn lúc phân loại chỉ có `max_completion_tokens=3`. Giả thuyết của tôi là 3 token không đủ nên nội dung trả về bị rỗng hoặc bị cắt. Cần in `response` thô ra để kiểm chứng rồi chạy lại. Phần dưới đây chỉ dùng số độ trễ và chi phí của LLM, không dùng F1.

## 3. Dự đoán và thực tế

**(1) `llm_models` — model lớn vs model nhỏ.**
Dự đoán: gpt-6.1-terra hơn gpt-6-luna khoảng 1–5 điểm F1. Thực tế: cả hai đều F1 = 0.000 với 40/40 câu trả lời không hợp lệ. Dự đoán lệch vì tôi chỉ nghĩ đến chất lượng model mà bỏ qua đường ống xung quanh nó: giới hạn token, định dạng output, cách parse. Model "giỏi" đến đâu thì một bước parse hỏng cũng đưa F1 về 0. Bài học là phải kiểm tra `n_invalid` và đọc output thô trước khi so sánh model. Riêng chi phí thì khớp kỳ vọng: model lớn đắt gấp khoảng 10 lần ($0.0495 so với $0.00495 / 1k tin).

**(2) `emb_tricky` — BoW vs Encoder + LogReg trên 6 tin "ngụy trang".**
Dự đoán: Encoder + LogReg đúng nhiều hơn. Thực tế: **BoW 5/6, Encoder + LogReg 4/6**. Embedding đoán sai tin *"Your mobile number has been chosen for a complimentary holiday. Reply YES to confirm"*, trong khi BoW vẫn bắt được nhờ các token `mobile`, `reply`, `your` có trọng số spam cao (`your` nằm trong top 12 từ spam của tôi). Lý do lệch: embedding chỉ là đặc trưng, còn ranh giới quyết định vẫn do LogReg học từ 3.619 tin train. Encoder nén cả câu thành một vector, nên phần "lịch sự, trung tính" của câu có thể kéo vector về phía ham. Tin *"Last chance! Your account rewards expire tonight…"* thì cả hai đều sai, vì đây là kiểu phishing gần như không có trong dữ liệu SMS 2011.

**(3) `tradeoff` — ai nằm trên đường Pareto.**
Dự đoán: Pareto gồm BoW + LogReg (nhanh nhất), Encoder + LogReg (cân bằng) và LLM lớn (chính xác nhất), còn MLP sẽ bị lấn át. Thực tế thì gần như ngược lại: Pareto là **BoW + MLP, MiniLM + MLP head, MiniLM + kNN**. BoW + MLP vừa nhanh hơn (p95 7.2 ms so với 14.9 ms) vừa cho F1 SUB bằng BoW + LogReg, nên chính BoW + LogReg mới bị lấn át. MiniLM + MLP head lấn át MiniLM + LogReg (cùng F1, p95 32 ms so với 40 ms). Lý do lệch: tôi giả định "model phức tạp hơn thì chậm hơn", nhưng ở đây chi phí chính nằm ở **bước biểu diễn** (`vectorize` viết bằng vòng lặp Python, encoder MiniLM) chứ không ở đầu phân loại. MLP chạy bằng PyTorch đã vector hoá nên còn nhanh hơn `predict_proba` numpy. Ngoài ra trên 40 tin, chênh dưới khoảng 0.05 F1 nên được coi là hoà.

*(Cũng lệch nhiều là `hard_eval`: tôi đoán LLM đứng đầu trên 50 case khó, thực tế LLM chỉ đạt accuracy 0.42 = tỉ lệ ham trong bộ, tức đoán toàn `ham`, vẫn do lỗi parse ở trên.)*

## 4. Phân tích 50 case khó

| Model | F1 SUB (test thường) | F1 50 case khó | Tụt | Acc 50 case khó |
| --- | ---: | ---: | ---: | ---: |
| 1. BoW + LogReg | 0.889 | 0.378 | **−0.511** | 0.54 |
| 2. BoW + MLP | 0.889 | 0.378 | **−0.511** | 0.54 |
| 3a. Emb + kNN | 0.923 | 0.772 | −0.151 | **0.74** |
| 3b. Emb + LogReg | 0.919 | 0.565 | −0.354 | 0.60 |
| 3c. Emb + MLP head | 0.919 | 0.596 | −0.323 | 0.62 |
| 4a / 4d. LLM nhỏ | (lỗi parse) | 0.000 | — | 0.42 |

**Model tụt nhiều nhất là hai model BoW** (−0.51 F1), đúng với dự đoán. Điểm test thường quá lạc quan vì test lấy cùng phân phối với train, và còn những spam mẫu chỉ khác số điện thoại hoặc mã thưởng nằm ở cả hai tập. BoW học thuộc từ khoá của spam năm 2011 (`call`, `txt`, `free`, `stop`, `www`, `uk`…) nên khi gặp spam kiểu mới (giọng tự nhiên, mạo danh ngân hàng, bưu điện, sếp) thì không còn tín hiệu để dựa vào. Bất ngờ là **kNN lại bền nhất** (0.74), dù trên test thường nó có precision thấp nhất. Bỏ phiếu theo hàng xóm gần nhất trong không gian nghĩa ít bị một ranh giới tuyến tính "học thuộc" chi phối hơn LogReg.

Có 5 case mà **tất cả** model đều sai: #12, #13, #16, #19 (spam giọng tự nhiên) và #31 (né từ khoá). Hai case cụ thể:

- **#31 — "U have been s.e.l.e.c.t.e.d for a c.a.s.h r.e.w.a.r.d, c.a.l.l 09061 701 461"** (nhóm né từ khoá). Hàm `tokenize` dùng regex `[a-z0-9£$€]+` nên dấu chấm tách `c.a.s.h` thành các token một chữ cái `c`, `a`, `s`, `h`. Các từ `cash`, `selected`, `call`, vốn có trọng số spam cao, biến mất khỏi vector BoW. Còn lại `u`, `have`, `for`, `a`, đều là từ hay gặp trong tin ham. Với MiniLM, tokenizer WordPiece cũng băm chuỗi thành các mảnh chữ cái rời, nên embedding không còn mang nghĩa "nhận tiền thưởng". Cả hai cách biểu diễn đều mất thông tin ngay từ bước tiền xử lý, trước khi tới bộ phân loại.
- **#19 — "Hey sweetie, missed you lately, I've got a little something special for you tonight, just reply and I'll send the link 😘"** (spam giọng tự nhiên). Câu không có từ khoá spam nào, toàn là từ của hội thoại thân mật (`i`, `you`, `missed`, `tonight`), đúng nhóm từ mà BoW học là kéo về ham (`i`, `me`, `my`… nằm trong top từ ham). Về mặt ngữ nghĩa, MiniLM đặt câu này gần cụm tin nhắn tình cảm giữa bạn bè, người yêu trong tập train, nên kNN, LogReg và MLP head đều đoán ham. Dấu hiệu spam thật sự ở đây là *ý định* (dụ người nhận trả lời để gửi link), và phải hiểu ngữ dụng mới thấy được. Đây là thứ LLM lẽ ra làm tốt hơn, nhưng trong lần chạy này chưa đo được vì lỗi parse.

Ngoài ra LLM nhỏ (4a và cả 4d có prompt phòng thủ) sai 29/50, đúng bằng số case spam. Đây là dấu hiệu rõ ràng của việc "mọi câu trả lời đều không parse được, mặc định 0", chứ chưa phải bằng chứng LLM bị prompt injection lừa.

## 5. Thử thách cá nhân

**Đề #2** (SEED 9493 % 6 = 1): *ML · Giảm vocab còn 100 từ và tăng lên 5000 từ (`build_vocab(..., max_size)`). Train lại, báo F1 test. Giải thích xu hướng.*

---

| `max_size` | F1 test | Precision | Recall |
| ---: | ---: | ---: | ---: |
| 100 | *(điền)* | *(điền)* | *(điền)* |
| 2000 (mặc định) | 0.855 | 1.000 | 0.747 |
| 5000 | *(điền)* | *(điền)* | *(điền)* |

*Cách làm:* ---

*Giải thích xu hướng:* ---

## 6. Phương án cho sản phẩm thật

**Yêu cầu:** 10 triệu SMS/ngày ≈ 116 tin/giây trung bình (giờ cao điểm có thể gấp vài lần), p95 < 50 ms/tin.

**Lọc theo độ trễ** (bảng ở mục 2): chỉ có BoW + LogReg (14.9 ms), BoW + MLP (7.2 ms), MiniLM + LogReg (40.4 ms) và MiniLM + MLP head (32.1 ms) đạt SLA. kNN (107 ms) và mọi LLM (p95 1.3–3.4 **giây**) đều vượt SLA từ 2 đến 70 lần.

**Lọc tiếp theo chất lượng:** BoW + MLP nhanh nhất nhưng tụt còn F1 0.378 trên 50 case khó, tức là sẽ lọt phần lớn spam kiểu mới. MiniLM + MLP head có F1 test cao nhất (0.957), giữ được 0.596 trên case khó, p95 32 ms vẫn còn khoảng 18 ms dư so với SLA.

**Chọn: MiniLM + MLP head (3c) làm bộ lọc chính**, có thể kèm router:

1. **Tầng 1 (đồng bộ, < 50 ms):** MiniLM + MLP head chấm mọi tin. Throughput đo trên Colab là khoảng 53 tin/s cho mỗi tiến trình, nên cần ít nhất ~3 instance cho mức trung bình và khoảng 6–10 instance (hoặc 1 GPU, batching) cho giờ cao điểm.
2. **Tầng 2 (bất đồng bộ):** tin có xác suất lưng chừng (p ∈ [0.3, 0.7]) vẫn được giao ngay theo nhãn tầng 1, đồng thời được đưa vào hàng đợi cho LLM chấm lại. Nhãn của LLM dùng để gắn cảnh báo hoặc chặn số gửi, và để tạo dữ liệu gán nhãn cho lần train sau. Vì LLM không nằm trên đường xử lý đồng bộ nên p95 của nó không ảnh hưởng tới SLA.
3. **Tiền xử lý chung:** chuẩn hoá chữ bị ngụy trang (`0→o`, `€→e`, bỏ dấu chấm hoặc khoảng trắng chen giữa các chữ cái), để cứu được những case như #29–#34.

**Chi phí nếu dùng LLM cho toàn bộ lưu lượng:** theo số đo của tôi, gpt-6-luna tốn $0.00495 / 1k tin, tức 10.000 × 0.00495 ≈ **$50/ngày ≈ $1.500/tháng**. gpt-6.1-terra tốn khoảng **$495/ngày ≈ $15.000/tháng**. Nếu theo giả định của bài tính tay (100 token input + 1 token output) thì gpt-6-luna tốn khoảng $105/ngày ≈ $3.150/tháng. Tiền không phải rào cản chính. Rào cản là **độ trễ** (p95 > 1 giây). Nếu router chỉ gửi khoảng 5% số tin sang LLM thì chi phí giảm còn khoảng $75–160/tháng.

**Đánh đổi chấp nhận:** (a) chấp nhận F1 thấp hơn LLM về lý thuyết trên các tin phải hiểu ý định (như #19), đổi lấy đáp ứng được SLA. (b) Phải vận hành encoder khoảng 90 MB và nhiều instance thay vì một model BoW rất nhẹ. (c) Tầng LLM bất đồng bộ nghĩa là một phần spam "khó" vẫn tới tay người dùng trước khi bị gắn cờ. (d) Cần retrain định kỳ, vì 50 case khó cho thấy model học từ dữ liệu 2011 bị lỗi thời với lừa đảo kiểu mới.

## 7. AI log và điều đã học

| Phần | Hỏi AI gì | AI trả lời / viết gì | Tôi đã sửa / kiểm chứng gì |
| --- | --- | --- | --- |
| Phần 4 (vùng được dùng AI) | Tôi đã nhờ Copilot soạn giúp tôi hàm | AI đã trả về đủ hàm | Tôi kiểm chứng rằng hàm thực sự ra kết quả trên jupyter |
| Câu trả lời giải thích trong notebook | Các câu hỏi từ phần 2 trở đi, tôi đã nhờ AI viết ngắn gọn câu trả lời | AI đã cho tôi câu trả lời ngắn gọn | Tôi đã kiểm tra rằng câu trả lời có đủ ý kiến tôi muốn diễn đạt |
| Báo cáo này (`2A202604111_PhamNgocLong.md`) | Nhờ Claude Code "giúp hoàn thành báo cáo" | Claude Code đọc notebook và `lab1_report.json`, soạn các mục 2, 3, 4, 6 từ số liệu của tôi, và phát hiện lỗi LLM `invalid=40/40` cùng ảnh `hand_calc.jpg` bị trống. Mục 5 và phần "điều đã học" được để trống cho tôi tự làm. | Tôi chưa sửa, hết giờ làm bài rồi |

**Điều tôi hiểu ra hôm nay:** Có nhiều cách tiếp cận để giải bài toán lọc spam trong hòm thư, mỗi cách sẽ có ưu nhược riêng. Bằng việc kiểm chứng kết quả sau mỗi phần - mỗi cách thực hiện, tôi nắm rõ hơn điểm mạnh, điểm yếu của các cách tiêp cận. 
