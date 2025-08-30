from modules.SubtitleCorrector import SubtitleCorrector

if __name__ == "__main__":
    api_key = "you_api_key"

    audio_files = [
        r"D:\PythonProject\SyncLRC\test\test.mp3"
    ]

    reference_text = """
    China Travel: The New Global Trend. Have you ever seen videos of foreigners exploring famous Chinese attractions? Their excited faces clearly show that more and more people are enjoying trips to China.
    A major reason for that is China's new visa-free policy. South Korean tourists especially love this. Many young Koreans now spend weekends in Shanghai, relaxing in cafés with views of the Bund and enjoying barbecue. Others take their parents to Zhangjiajie, a city famous for its breathtaking mountains and often called by Koreans "a place you must take your parents at least once."
    In March 2025, this trend reached a new high. IShowSpeed, one of America's most popular YouTubers with over 38 million followers, visited China for the first time. In Beijing, he toured the Great Wall and the Forbidden City, performing his famous backflip in traditional Chinese clothes. He also tasted tanghulu and was surprised by its sweet-and-sour flavor. On the high-speed train, he found it unbelievable that the WiFi stayed strong even in tunnels.  His livestream from Beijing attracted five million viewers worldwide. Some fans commented, "China is safe, friendly and clean. It is much better than we imagined."
    For a long time, Western textbooks and media described China as poor, closed-off, and unsafe for travelling. Now, these foreign tourists are breaking stereotypes with their own experiences, showing the world a real and wonderful China.
    """

    sc = SubtitleCorrector(
        api_key=api_key,
        whisper_model="medium.en",
        use_reference=False,
        llm_model="deepseek-chat",
        num_threads=2,
        max_retry=3,
    )

    results = sc.process_files(audio_files, reference_text)

    for file, lrc in results.items():
        print(f"{file} 修正结果:\n" + "\n".join(lrc))