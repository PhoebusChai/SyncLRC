# 📖 SyncLRC  
基于 **WhisperX + DeepSeek LLM** 的智能字幕同步与校正工具  

支持 **音频自动转录、字幕格式化（LRC）、内容准确率评估、多线程加速和日志追踪**。  

---

## ✨ 功能特性

- 🎙️ **自动语音识别 (ASR)**：使用 WhisperX 进行高精度转录  
- 🤖 **大模型校正**：调用 DeepSeek 推理模型，确保字幕与参考文本一致  
- 📝 **字幕输出 (LRC)**：生成带时间戳的 `.lrc` 歌词/字幕文件  
- 📊 **准确率评估**：逐行计算相似度，统计平均/最低/最高准确率  
- ⚡ **多线程支持**：批量文件处理，加速字幕生成  
- 🛠️ **参数可配置**：模型大小、线程数、是否使用大模型校验均可灵活设置  
- 📑 **专业日志系统**：使用标准 `logging`，分级记录 `INFO / WARNING / ERROR`  

---

## 📂 项目结构

```angular2html
SyncLRC/
│── main.py # 主入口
│── modules/
│ │── SubtitleCorrector.py # 核心字幕处理模块
│ │── ContentTools.py # 内容评估工具 (准确率计算)
│── test/
│ │── *.mp3 # 测试音频文件
│── README.md
```
---
## ⚙️ 安装依赖

1. 克隆项目  
```bash
git clone https://github.com/yourname/SyncLRC.git

cd SyncLRC
```

2. 必要依赖：
- torch
- whisperx
- httpx
- rapidfuzz
- tqdm

## 🚀 使用方法
1. 启动处理
```
python main.py
```

2. 参数配置

在 SubtitleCorrector 初始化时可指定：
```
corrector = SubtitleCorrector(
    api_key="your_deepseek_api_key",
    model_size="medium.en",        # WhisperX 模型大小
    use_llm_validation=True,       # 是否启用大模型校验
    llm_model="deepseek-reasoner", # 大模型名称
    num_threads=4                  # 多线程数量
)
```

# 📈 准确率评估

调用：

`ContentTools.evaluate_accuracy(asr_text, ref_text)`

功能：

- 返回每行的 相似度分数

- 输出整体的 平均、最低、最高准确率

- 对低于阈值 (默认 0.85) 的行，重点标记

# 🛠️ 开发计划

>支持 字幕对齐（自动切分长文本）X

>支持 多语种模型（不仅限于 English）X

>添加 可视化界面 (Streamlit/WebUI) X

>集成 字幕翻译 功能 X
