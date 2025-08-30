import whisperx
import torch
import re
import logging
from modules.ContentTools import ContentTools
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

# ========== 日志配置 ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

class SubtitleCorrector:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        whisper_model: str = "medium.en",
        align_language: str = "en",
        max_retry: int = 3,
        use_reference: bool = True,
        llm_model: str = "deepseek-reasoner",
        num_threads: int = 1
    ):
        # API
        self.client = OpenAI(api_key=api_key, base_url=base_url)

        # 设备
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"使用设备: {self.device}")

        # 配置
        self.whisper_model = whisper_model
        self.align_language = align_language
        self.max_retry = max_retry
        self.use_reference = use_reference
        self.llm_model = llm_model
        self.num_threads = num_threads

        # 模型
        logger.info(f"加载 WhisperX 模型: {self.whisper_model}")
        self.model = whisperx.load_model(self.whisper_model, self.device)
        self.model_a, self.metadata = whisperx.load_align_model(
            language_code=self.align_language, device=self.device
        )

    def _merge_words_to_sentences(self, aligned_words):
        """单词合并成句子"""
        sentences = []
        current_sentence = []
        start_time = None

        for item in aligned_words:
            word = item["word"]
            time = item["time"]

            if start_time is None:
                start_time = time

            if re.match(r'^[.,!?;:]+$', word):
                if current_sentence:
                    current_sentence[-1] += word
            else:
                current_sentence.append(word)

            if word.endswith(('.', '?', '!')):
                sentences.append((start_time, " ".join(current_sentence)))
                current_sentence = []
                start_time = None

        if current_sentence:
            sentences.append((start_time, " ".join(current_sentence)))

        return sentences

    def _transcribe_single(self, audio_file: str, reference_text: str = None):
        """处理单个音频 → 修正后的 LRC"""
        logger.info(f"正在转写音频: {audio_file}")
        asr_result = self.model.transcribe(audio_file)

        aligned_result = whisperx.align(
            asr_result["segments"], self.model_a, self.metadata, audio_file, self.device
        )

        word_lrc = [{"time": w["start"], "word": w["word"]} for w in aligned_result["word_segments"]]
        sentence_lrc = self._merge_words_to_sentences(word_lrc)

        # 如果不开启大模型校对，直接拼 LRC 返回
        if not self.use_reference or not reference_text:
            logger.info(f"未启用参考文本校对，直接返回 ASR LRC: {audio_file}")
            return self._format_lrc(sentence_lrc)

        # 调用大模型校对
        return self._correct_with_reference(sentence_lrc, reference_text)

    def _correct_with_reference(self, sentence_lrc, reference_text: str):
        """用大模型对齐参考文本"""
        asr_text = "\n".join(s for _, s in sentence_lrc)

        check_prompt = f"""
        你是一个字幕校对专家。  
        我有两份文本：  
        - 【ASR字幕】是通过语音识别得到的，包含时间戳对应的句子，切分粒度不能改变。  
        - 【参考文本】是正确的文字版本，但没有时间戳，切分方式可能不同。  
        
        你的任务：  
        1. 逐行检查【ASR字幕】，对照【参考文本】，修复拼写、大小写、标点、词汇错误。  
        2. **禁止改变行数**，每一行只能修改文字，不允许新增或删除行。  
        3. 保留与输入相同的行顺序，输出的行数必须和【ASR字幕】完全一致。  
        4. 只输出字幕文字，不要时间戳。  
        
        【ASR字幕】(共 {len(sentence_lrc)} 行):
        {asr_text}
        
        【参考文本】:
        {reference_text}
        
        【输出字幕】:
        """

        corrected_lines = None
        for attempt in range(1, self.max_retry + 1):
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": check_prompt}],
                stream=False
            )
            corrected_text = response.choices[0].message.content.strip()
            corrected_lines = corrected_text.split("\n")

            if len(corrected_lines) == len(sentence_lrc):
                logger.info(f"第 {attempt} 次字幕校对成功")
                break
            else:
                logger.warning(
                    f"第 {attempt} 次校对失败: 输出 {len(corrected_lines)} 行，需要 {len(sentence_lrc)} 行"
                )

        if corrected_lines is None:
            logger.error("字幕校对失败，返回原始 ASR 结果")
            return self._format_lrc(sentence_lrc)

        # 拼接回 LRC
        final_lrc = []
        for (t, _), new_text in zip(sentence_lrc, corrected_lines):
            minutes = int(t // 60)
            seconds = int(t % 60)
            centis = int((t % 1) * 100)
            final_lrc.append(f"[{minutes:02d}:{seconds:02d}.{centis:02d}] {new_text}")

        return final_lrc

    @staticmethod
    def _format_lrc(sentence_lrc):
        """格式化 LRC 输出"""
        return [
            f"[{int(t // 60):02d}:{int(t % 60):02d}.{int((t % 1) * 100):02d}] {s}"
            for t, s in sentence_lrc
        ]

    def process_files(self, audio_files: list, reference_text: str = None):
        """批量处理多个音频文件，支持多线程"""
        results = {}
        if self.num_threads > 1:
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                future_to_file = {
                    executor.submit(self._transcribe_single, f, reference_text): f
                    for f in audio_files
                }
                for future in as_completed(future_to_file):
                    file = future_to_file[future]
                    try:
                        results[file] = future.result()
                    except Exception as e:
                        logger.error(f"处理 {file} 出错: {e}")
                        results[file] = []
        else:
            for f in audio_files:
                results[f] = self._transcribe_single(f, reference_text)

        if reference_text:
            for file, lrc_lines in results.items():
                if not lrc_lines:
                    continue
                lrc_text = "\n".join(lrc_lines)  # LRC 转成文本
                logger.info(f"==== 准确率评估: {file} ====")
                ContentTools.evaluate_accuracy(lrc_text, reference_text)

        return results