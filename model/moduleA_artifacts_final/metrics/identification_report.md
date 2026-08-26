# Kết quả đánh giá Speaker Identification (SID)

- Số speaker đủ điều kiện (>=2 utterance): 15
- Tổng số truy vấn: 105
- Top-1 accuracy (closed-set, không áp threshold): 84.76%
- Accuracy có áp threshold=0.35: 73.33%
- Tỉ lệ bị từ chối thành 'unknown': 26.67%

**Lưu ý:** closed-set nghĩa là giả định người truy vấn CHẮC CHẮN đã có trong database enrollment — đây là giả định lý tưởng hoá, không phản ánh trường hợp người lạ hoàn toàn chưa từng enroll (open-set), vốn cần đánh giá riêng bằng cách thêm truy vấn từ speaker KHÔNG có trong db_embeddings.
