# 100-game evaluation record

[中文](#中文) | [English](#english)

## 中文

这是上一代模型 `value-transformer-20260920-161113-612005` 的完整 100 局行为评估。模型固定使用 depth 1 expectimax、4096 搜索展开上限和 512 推理 batch。100 局均使用固定模型运行至终局。

| 指标 | 上一代学习模型 | 历史 mixed4 基线 |
|---|---:|---:|
| 完成局数 | 100 | 100 |
| 平均分 | 121,442.04 | 77,019.64 |
| 平均分 95% bootstrap 区间 | 113,300–129,789 | 71,208–82,924 |
| 中位数 | 130,266 | 78,414 |
| 最低 / 最高分 | 37,972 / 250,604 | 27,576 / 169,588 |
| 达到 4096 | 99% | 80% |
| 达到 8192 | 56% | 18% |

平均分差为 +44,422（+57.7%），独立局 bootstrap 95% 区间为 +34,291 至 +54,650。两组不是同种子配对实验；历史 mixed4 基线与最初训练语料有重叠，因此这些数字用于描述对局行为，不应解释为独立留出集上的泛化保证。当前发布模型比该模型更新，但只留下 19 局阶段评估，所以首页没有把它标成 100 局结果。

## English

This is the complete 100-game behavioral evaluation of the previous model, `value-transformer-20260920-161113-612005`. It used depth-1 expectimax, a 4,096-expansion search cap and an inference batch size of 512. The same pinned model played every game to termination.

| Metric | Previous learned model | Historical mixed4 baseline |
|---|---:|---:|
| Completed games | 100 | 100 |
| Mean score | 121,442.04 | 77,019.64 |
| Mean 95% bootstrap interval | 113,300–129,789 | 71,208–82,924 |
| Median | 130,266 | 78,414 |
| Minimum / maximum score | 37,972 / 250,604 | 27,576 / 169,588 |
| Reached 4096 | 99% | 80% |
| Reached 8192 | 56% | 18% |

The mean-score difference was +44,422 (+57.7%), with an independent-game bootstrap 95% interval of +34,291 to +54,650. The rows were not evaluated on paired seeds, and the historical mixed4 baseline overlaps the original training corpus. These figures describe playing behavior rather than held-out generalization. The current release model is newer, but only a 19-game interim evaluation was retained, so the repository does not present it as a 100-game result.
