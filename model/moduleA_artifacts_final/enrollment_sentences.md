# Câu mẫu Enrollment — Trợ lý ảo xe thông minh

**Mục đích:** dùng trong luồng đăng ký giọng nói (Module B, M5 Enrollment) — người dùng đọc lần lượt 7 câu này để hệ thống thu đủ mẫu giọng cho SV/SID.

**Lưu ý về độ chặt chẽ ngôn ngữ học:** đây là bản thiết kế thực dụng, ưu tiên phủ đa dạng thanh điệu + nhóm phụ âm đầu + nguyên âm đôi phổ biến trong tiếng Việt, dựa trên kiến thức ngữ âm cơ bản — **không phải kết quả đối chiếu chính thức với tài liệu chuyên sâu** (2 bài báo giảng viên chia sẻ về enrollment selection mà nhóm chưa truy cập được full-text, theo ghi chú trong `00_Workflow_And_TechStack.pdf` mục 8). Nên ghi rõ giới hạn này trong báo cáo.

---

## Danh sách 7 câu

| # | Câu | Ngữ cảnh xe | Thanh điệu chính xuất hiện | Phụ âm đầu đáng chú ý |
|---|---|---|---|---|
| 1 | "Xin chào, tôi tên là Khánh, đây là giọng nói của tôi." | Giới thiệu (mở đầu tự nhiên) | ngang, huyền, sắc, hỏi | kh, ch, gi, đ |
| 2 | "Hôm nay trời nắng đẹp, tôi muốn nghe một bản nhạc nhẹ nhàng." | Yêu cầu phát nhạc (general) | huyền, sắc, nặng, ngang | tr, ng, nh, b |
| 3 | "Làm ơn mở cửa sổ bên trái và bật điều hòa mát mẻ." | Điều khiển tiện nghi trong xe | hỏi, sắc, huyền, ngang | l, m, c, tr, đ |
| 4 | "Chỉ đường cho tôi đến quán cà phê gần nhất được không?" | Chỉ đường (general) | ngã, hỏi, sắc, huyền | ch, đ, q, ph, kh |
| 5 | "Tôi thường lái xe vào buổi sáng sớm để tránh kẹt xe." | Thói quen lái xe (cá nhân hoá SID) | huyền, sắc, nặng, ngang | th, l, x, v, tr |
| 6 | "Xin vui lòng ghi nhớ giọng nói này để nhận diện tôi sau này." | Nhắc mục đích enrollment | ngã, nặng, huyền, ngang | v, gh, nh, d |
| 7 | "Bây giờ hãy phát danh sách nhạc yêu thích của tôi." | Lệnh cá nhân hoá cuối cùng (SID) | huyền, ngã, sắc, nặng | b, gi, h, ph, th |

## Rationale tổng hợp

- **Phủ đủ 6 thanh điệu** (ngang, huyền, sắc, hỏi, ngã, nặng) — mỗi câu chứa tối thiểu 3-4 thanh khác nhau, tổng hợp cả 7 câu đảm bảo không thanh nào bị thiếu vắng hoàn toàn.
- **Đa dạng phụ âm đầu:** bao gồm cả phụ âm đơn (b, m, l, h, đ, x, v, d) và tổ hợp phụ âm đặc trưng tiếng Việt (ch, tr, kh, ph, th, ngh, nh, gi, gh, qu) — nhóm phụ âm dễ nhầm lẫn khi nhận dạng (ví dụ tr/ch, x/s không xuất hiện trong danh sách này do giới hạn ngữ cảnh câu tự nhiên, có thể bổ sung nếu cần chặt chẽ hơn).
- **Nguyên âm đôi/ba phổ biến:** "đường", "được", "muốn", "điều hòa", "nhất" chứa các tổ hợp `ươ`, `uô`, `iê`, `ương` — nhóm nguyên âm phức tạp thường ảnh hưởng đến chất lượng đặc trưng giọng nói.
- **Ngữ cảnh xe hơi tự nhiên:** câu 2-5, 7 lồng ghép tình huống thực tế của trợ lý ảo xe (phát nhạc, điều hoà, chỉ đường, thói quen lái xe), giúp trải nghiệm enrollment không bị khô khan như đọc câu vô nghĩa, đồng thời câu 5, 7 có thể tái sử dụng làm ví dụ minh hoạ cho use case SID trong phần demo.
- **Độ dài mỗi câu** ước tính 4-6 giây khi đọc tự nhiên — đủ dài để trích embedding ổn định (khớp khuyến nghị `min_duration_sec` trong pipeline train), không quá dài gây khó chịu khi enroll.

## Khuyến nghị bổ sung cho Module B (M1 — check_audio_quality)

Khi Hiếu code phần kiểm tra chất lượng mẫu enrollment, có thể dùng chính 7 câu này để đối chiếu qua ASR (WER so với câu mẫu) như đã thiết kế trong `00_Workflow_And_TechStack.pdf` mục 8 (Tầng 1 — Audio Quality). Nội dung câu ở trên đã cố định, không nên đổi giữa chừng vì sẽ ảnh hưởng đến logic đối chiếu WER nếu đã code cứng.
