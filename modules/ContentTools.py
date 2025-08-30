import logging
import re
from rapidfuzz import fuzz, process

logger = logging.getLogger("ContentTools")

class ContentTools:
    @staticmethod
    def compute_similarity(text1: str, text2: str) -> float:
        """
        计算两个字符串的相似度分数 (0~1)
        """
        return fuzz.ratio(text1.strip(), text2.strip()) / 100.0

    @staticmethod
    def strip_timestamps(lrc_text: str) -> str:
        """
        去掉 LRC 时间戳
        """
        return re.sub(r"\[\d{2}:\d{2}\.\d{2}\]\s*", "", lrc_text).strip()

    @staticmethod
    def split_sentences(text: str) -> list[str]:
        """
        参考文本按句子切分（句号、问号、感叹号）
        """
        sentences = re.split(r'(?<=[。！？.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]

    @staticmethod
    def evaluate_accuracy(asr_text: str, ref_text: str, threshold: float = 0.85, mode: str = "sentence"):
        """
        评估字幕和参考文本的相似度

        :param asr_text: ASR 生成的字幕（可能含 LRC 时间戳）
        :param ref_text: 参考文本
        :param threshold: 相似度阈值
        :param mode: "global" | "sentence"
        """
        clean_asr = ContentTools.strip_timestamps(asr_text)

        if mode == "global":
            score = ContentTools.compute_similarity(clean_asr, ref_text)
            logger.info("整体准确率统计：")
            logger.info(f"整体相似度: {score:.4f}")
            if score < threshold:
                logger.warning(f"整体准确率低于阈值 ({threshold})")
            return score

        elif mode == "sentence":
            asr_lines = [line.strip() for line in clean_asr.split("\n") if line.strip()]
            ref_sentences = ContentTools.split_sentences(ref_text)

            similarities = []
            low_detail = []

            for i, asr in enumerate(asr_lines, start=1):
                # 在参考句子中找最接近的一句
                match, score, _ = process.extractOne(asr, ref_sentences, scorer=fuzz.ratio)
                score = score / 100.0
                similarities.append(score)

                if score < threshold:
                    detail = (
                        f"行 {i} | 相似度={score:.4f}\n"
                        f"  最终结果: {asr}\n"
                        f"  匹配参考: {match}"
                    )
                    logger.debug(detail)
                    low_detail.append(detail)

            if similarities:
                avg = sum(similarities) / len(similarities)
                logger.info("整体准确率统计：")
                logger.info(f"平均相似度: {avg:.4f}")
                logger.info(f"最低相似度: {min(similarities):.4f}")
                logger.info(f"最高相似度: {max(similarities):.4f}")

                if low_detail:
                    logger.warning(f"共有 {len(low_detail)} 行低于阈值 ({threshold})")
                    logger.warning("低分详情如下：\n" + "\n".join(low_detail))

            return similarities
