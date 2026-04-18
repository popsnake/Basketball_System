把你的知识文档放到这个目录里即可，第一版推荐使用 UTF-8 编码的 .txt 文件。

建议做法：
1. 一个主题一份文件。
2. 文件名直接表达主题，例如：
   shooting_basics.txt
   elbow_alignment.txt
   training_faq.txt
3. 每个文件尽量聚焦一个主题，内容按自然段组织。
4. 文档里尽量写事实、规则、步骤、FAQ，不要把多个完全不同主题混在一份文件里。

导入命令：
D:\base\.venv\Scripts\python.exe -m agents.baseline.kb_ingest

导入完成后，/v1/chat 会自动优先检索这些知识块。
