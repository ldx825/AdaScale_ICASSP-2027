# LG-CLIP 修改记录（保持可追踪）

## 2026-09-11: 补充缺失文件（bug fix，非功能性改动）

官方 repo commit e0a3c6f 的 `clip/` 目录缺少 `bpe_simple_vocab_16e6.txt.gz`（CLIP tokenizer 词表），
导致 `import clip` 立即失败。已从 OpenAI CLIP 官方仓库下载同名文件补入：

    third_party/LG-CLIP/clip/bpe_simple_vocab_16e6.txt.gz

SHA256: TBD（后续补录）
影响：无算法改动；仅为补全依赖。

## 后续 instrumentation 计划

见 notes/code_map.md §3。优先使用官方脚本产生的 per-scale HDF5 缓存（无计算修改）离线分析；
仅在需要受控随机 crop 时新增 instrumented 脚本（放 src/analysis/ 下，不改 third_party 计算逻辑）。
924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a  third_party/LG-CLIP/clip/bpe_simple_vocab_16e6.txt.gz
