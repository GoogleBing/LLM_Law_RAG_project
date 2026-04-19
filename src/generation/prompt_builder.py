"""
prompt_builder.py - Convert a query + retrieved contexts into an LLM prompt.

The prompt enforces three behaviors the Qwen baseline failed at:
  - structured citation  ([Số hiệu, Điều X khoản Y]) — raises BLEU/ROUGE vs
    reference answers that are extractive.
  - honest expiry reporting: if a chunk carries a HẾT HIỆU LỰC note, the LLM
    must surface it and prefer an active doc in context if one exists.
  - refusal when context is insufficient — avoids hallucinated citations.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MAX_CONTEXT_CHARS

SYSTEM_INSTRUCTION = (
    "Bạn là trợ lý pháp lý Việt Nam. Chỉ trả lời dựa trên VĂN BẢN PHÁP LUẬT "
    "được cung cấp ở dưới.\n\n"
    "QUY TẮC TRÍCH DẪN (BẮT BUỘC):\n"
    " - Mỗi luận điểm phải kèm trích dẫn trong ngoặc vuông theo đúng format: "
    "[Số hiệu: <số hiệu>, Điều <n>, Khoản <n>].\n"
    " - Ví dụ đúng: [Số hiệu: 50/2024/TT-NHNN, Điều 11, Khoản 3].\n"
    " - KHÔNG viết 'điều 11', 'Đ.11', 'điều thứ 11', 'Art. 11'. Luôn viết 'Điều <n>'.\n"
    " - KHÔNG trích dẫn số hiệu hoặc số Điều không xuất hiện trong phần VĂN BẢN PHÁP LUẬT bên dưới.\n\n"
    "QUY TẮC NỘI DUNG:\n"
    " - Nếu một văn bản có ghi chú đã HẾT HIỆU LỰC hoặc 'Đã được thay thế bởi ...', "
    "hãy nêu rõ điều đó và ưu tiên dẫn văn bản thay thế nếu nó cũng có trong ngữ cảnh.\n"
    " - Nếu ngữ cảnh không đủ để trả lời, hãy nói 'Ngữ cảnh được cung cấp chưa đủ "
    "để trả lời câu hỏi này.' thay vì suy đoán."
)


def build_prompt(query: str, contexts: list[dict]) -> str:
    parts = []
    total_chars = 0
    for i, ch in enumerate(contexts, 1):
        so_hieu = ch.get("so_hieu") or ""
        tieu_de = ch.get("tieu_de") or ""
        status  = ch.get("tinh_trang") or "(không rõ tình trạng)"
        header = f"[{i}] Số hiệu: {so_hieu} | Tình trạng: {status}"
        if tieu_de:
            header += f"\n    Tiêu đề: {tieu_de}"
        note = ch.get("_freshness_note") or ""
        if note:
            header += f"\n    {note}"

        remaining = MAX_CONTEXT_CHARS - total_chars
        if remaining <= 0:
            break
        text = ch.get("text") or ""
        if len(text) > remaining:
            text = text[:remaining] + " [...]"
        total_chars += len(text)
        parts.append(f"{header}\n{text}")

    context_str = "\n\n---\n\n".join(parts)
    return (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"=== VĂN BẢN PHÁP LUẬT ===\n{context_str}\n\n"
        f"=== CÂU HỎI ===\n{query}\n\n"
        f"=== TRẢ LỜI ===\n"
    )
