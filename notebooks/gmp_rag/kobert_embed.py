"""KoBERT(skt/kobert-base-v1) 문장 임베딩 — GMP RAG용.

- 토크나이저 : 허깅페이스 저장소의 spiece.model 을 sentencepiece로 직접 읽어 KoBERTTokenizer(SKTBrain kobert_hf)와
  같은 방식으로 [CLS] 문장 [SEP] 을 만듦 → kobert_tokenizer(git 설치) 없이 동작
- 문장 벡터 : 마지막 은닉층의 mean pooling (패딩 제외) 후 L2 정규화 → 내적 = 코사인 유사도

주의 : KoBERT는 문장 유사도용으로 미세조정된 모델이 아님 (MLM 사전학습만). 검색 품질은 5why.ipynb 의 점검 셀로 확인.
"""
from __future__ import annotations

import unicodedata

import numpy as np

MODEL_ID = "skt/kobert-base-v1"
MAX_LEN = 512


class KoBERTEmbedder:
    def __init__(self, model_id: str = MODEL_ID, device: str | None = None):
        import sentencepiece as spm
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import AutoModel

        self.torch = torch
        self.sp = spm.SentencePieceProcessor(model_file=hf_hub_download(model_id, "spiece.model"))
        self.cls_id, self.sep_id, self.pad_id = (self.sp.piece_to_id(t) for t in ("[CLS]", "[SEP]", "[PAD]"))
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()

    def _ids(self, text: str) -> list[int]:
        text = " ".join(unicodedata.normalize("NFC", text).split())   # remove_space
        return [self.cls_id] + self.sp.encode(text)[: MAX_LEN - 2] + [self.sep_id]

    def encode(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        torch = self.torch
        out = []
        for i in range(0, len(texts), batch_size):
            ids = [self._ids(t) for t in texts[i:i + batch_size]]
            n = max(map(len, ids))
            input_ids = torch.tensor([x + [self.pad_id] * (n - len(x)) for x in ids], device=self.device)
            mask = torch.tensor([[1] * len(x) + [0] * (n - len(x)) for x in ids], device=self.device)
            with torch.no_grad():
                h = self.model(input_ids=input_ids, attention_mask=mask,
                               token_type_ids=torch.zeros_like(input_ids)).last_hidden_state
            m = mask.unsqueeze(-1).float()
            v = (h * m).sum(1) / m.sum(1)
            out.append(torch.nn.functional.normalize(v, dim=-1).cpu().numpy())
        return np.vstack(out).astype(np.float32)
