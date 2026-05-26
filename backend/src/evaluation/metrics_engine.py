from typing import List, Dict, Any
import re
from collections import Counter


class MetricsEngine:
    def exact_match(self, prediction: str, reference: str) -> float:
        return 1.0 if prediction.strip() == reference.strip() else 0.0

    def f1_score(self, prediction: str, reference: str) -> float:
        pred_tokens = self._tokenize(prediction)
        ref_tokens = self._tokenize(reference)
        
        if not pred_tokens and not ref_tokens:
            return 1.0
        if not pred_tokens or not ref_tokens:
            return 0.0

        common = Counter(pred_tokens) & Counter(ref_tokens)
        num_same = sum(common.values())
        
        if num_same == 0:
            return 0.0

        precision = num_same / len(pred_tokens)
        recall = num_same / len(ref_tokens)
        
        return 2 * precision * recall / (precision + recall)

    def bleu_score(self, prediction: str, reference: str, n: int = 4) -> float:
        pred_tokens = self._tokenize(prediction)
        ref_tokens = self._tokenize(reference)
        
        if not pred_tokens:
            return 0.0

        scores = []
        for i in range(1, n + 1):
            score = self._bleu_n(pred_tokens, ref_tokens, i)
            scores.append(score)
        
        if any(s == 0 for s in scores):
            return 0.0
        
        import math
        geometric_mean = math.exp(sum(math.log(s) for s in scores) / n)
        
        bp = min(1, math.exp(1 - len(ref_tokens) / len(pred_tokens))) if len(pred_tokens) > 0 else 0
        
        return bp * geometric_mean

    def rouge_1(self, prediction: str, reference: str) -> float:
        return self._rouge_n(prediction, reference, 1)

    def rouge_2(self, prediction: str, reference: str) -> float:
        return self._rouge_n(prediction, reference, 2)

    def rouge_l(self, prediction: str, reference: str) -> float:
        pred_tokens = self._tokenize(prediction)
        ref_tokens = self._tokenize(reference)
        
        lcs_length = self._lcs_length(pred_tokens, ref_tokens)
        
        if lcs_length == 0:
            return 0.0
        
        recall = lcs_length / len(ref_tokens)
        precision = lcs_length / len(pred_tokens)
        
        if recall + precision == 0:
            return 0.0
        
        return 2 * recall * precision / (recall + precision)

    def keyword_matching(self, prediction: str, keywords: List[str]) -> float:
        if not keywords:
            return 1.0
        
        prediction_lower = prediction.lower()
        matched = sum(1 for kw in keywords if kw.lower() in prediction_lower)
        
        return matched / len(keywords)

    def semantic_similarity(self, prediction: str, reference: str) -> float:
        try:
            import sentence_transformers
            # 禁用语义相似度，避免网络连接问题
            pass
        except ImportError:
            pass
        return self.f1_score(prediction, reference)

    def calculate_all(self, prediction: str, reference: str, keywords: List[str] = None) -> Dict[str, float]:
        keywords = keywords or []
        return {
            'exact_match': self.exact_match(prediction, reference),
            'f1': self.f1_score(prediction, reference),
            'bleu': self.bleu_score(prediction, reference),
            'rouge_1': self.rouge_1(prediction, reference),
            'rouge_2': self.rouge_2(prediction, reference),
            'rouge_l': self.rouge_l(prediction, reference),
            'keyword_matching': self.keyword_matching(prediction, keywords),
            'semantic_similarity': self.semantic_similarity(prediction, reference),
        }

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        # 对于中文文本，采用逐个字符分割（简单实现）
        if any('\u4e00' <= c <= '\u9fff' for c in text):
            return [c for c in text if not c.isspace()]
        return text.split()

    def _bleu_n(self, pred_tokens: List[str], ref_tokens: List[str], n: int) -> float:
        pred_ngrams = self._get_ngrams(pred_tokens, n)
        ref_ngrams = self._get_ngrams(ref_tokens, n)
        
        if not pred_ngrams:
            return 0.0
        
        common = Counter(pred_ngrams) & Counter(ref_ngrams)
        num_same = sum(common.values())
        
        return num_same / len(pred_ngrams)

    def _rouge_n(self, prediction: str, reference: str, n: int) -> float:
        pred_tokens = self._tokenize(prediction)
        ref_tokens = self._tokenize(reference)
        
        pred_ngrams = self._get_ngrams(pred_tokens, n)
        ref_ngrams = self._get_ngrams(ref_tokens, n)
        
        if not ref_ngrams:
            return 0.0
        
        common = Counter(pred_ngrams) & Counter(ref_ngrams)
        num_same = sum(common.values())
        
        return num_same / len(ref_ngrams)

    def _get_ngrams(self, tokens: List[str], n: int) -> List[tuple]:
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    def _lcs_length(self, a: List[str], b: List[str]) -> int:
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i-1] == b[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]