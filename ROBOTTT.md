# RoboTTT / GR00T on RLinf

仓库位于 H100 的 `/home/xuzhexuan/project/repos/robottt`，基于
`3cb54c2eb2e355378ba049d50f7ec0cb8acc2da0`，按原 training/eval 仓库的组件位置实现。

```bash
source /home/xuzhexuan/project/.venv_robottt_robocasa/bin/activate
cd /home/xuzhexuan/project/repos/robottt

bash examples/sft/run_vla_sft.sh robocasa_bowlandcup_gr00t_baseline_h100.yaml
bash examples/sft/run_vla_sft.sh robocasa_bowlandcup_robottt_h100.yaml

bash evaluations/run_eval.sh robocasa365_bowlandcup_gr00t_baseline_h100.yaml
bash evaluations/run_eval.sh robocasa365_bowlandcup_robottt_h100.yaml
```

两组需依次运行，各使用 8 张 H100。训练配置位于 `examples/sft/config/`，
评测配置位于 `evaluations/robocasa365/`；支持配置文件全路径及 Hydra overrides。
训练默认 30K、峰值 LR 5e-5、warmup 120、30K cosine、全参数、104 episodes、真实 H40 / 执行 8、TBPTT 16。
评测默认读取已经完成的历史 500-step 权重；新训练权重通过
`runner.ckpt_path=.../actor/model_state_dict/full_weights.pt` 指定。

- 核心 TTT、DiT 和 flow：`rlinf/models/embodiment/gr00t/gr00t_n1d7/`。
- 数据：`rlinf/data/datasets/gr00t_n1d7/`，接入 `FSDPVlaRoboTTTWorker(FSDPVlaSftWorker)`。
- 评测：原生 `EnvWorker` ↔ `MultiStepRolloutWorker` ↔ `Robocasa365Env`。
- [train/eval 配置复用、30K 与独立 worker](docs/robottt/30k_worker_refactor.md)。
- [全部文件修改与目的](docs/robottt/file_changes.md)。
- [组件重组与 FSDP 审查](docs/robottt/native_integration.md)。
- [数学协议、运行与恢复命令](docs/robottt/reproduction.md)。
- [参考提交审查](docs/robottt/reference_review.md) 和 [逐文件映射](docs/robottt/reference_commit_map.csv)。
- [本轮 30K/独立 worker 短运行验收](artifacts/worker30k/report.md)。
- [上阶段原生集成验收](artifacts/reorganization/reorganization_report.md)。
- [历史 500-step 验收](artifacts/acceptance/acceptance_report.md)：来自重组前实现，保留原始证据。

原独立 `rlinf/robottt` trainer/evaluator 及 `scripts/robottt` 启动脚本已移除，
有用的审计工具迁至 `toolkits/robottt`。训练集 loss 和短 rollout 不作为泛化证据。
