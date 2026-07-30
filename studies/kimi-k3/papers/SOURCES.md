# Sources — Kimi K3

The report is committed alongside these notes. To re-fetch or update it:

```bash
curl -sL -o k3_tech_report.pdf \
  https://github.com/MoonshotAI/Kimi-K3/raw/main/k3_tech_report.pdf
```

| | |
|---|---|
| Technical report | [`MoonshotAI/Kimi-K3`](https://github.com/MoonshotAI/Kimi-K3) — 47pp, rev. 27 Jul 2026 |
| Weights | [`moonshotai/Kimi-K3`](https://huggingface.co/moonshotai/Kimi-K3) — Kimi K3 License |
| Blog | [kimi.com/blog/kimi-k3](https://www.kimi.com/blog/kimi-k3) |

## Open-sourced stack referenced by the report

| Component | Repo | §  |
|---|---|---|
| FlashKDA — CUTLASS chunkwise KDA kernels | [MoonshotAI/FlashKDA](https://github.com/MoonshotAI/FlashKDA) | 5.1.1 |
| MoonEP — perfectly balanced expert parallelism | [MoonshotAI/MoonEP](https://github.com/MoonshotAI/MoonEP) | 5.2.1 |
| AgentENV — microVM sandboxes for agentic RL | [kvcache-ai/AgentENV](https://github.com/kvcache-ai/AgentENV) | 5.3.2 |
| MiniTriton — case-study compiler K3 wrote | [MoonshotAI/minitriton](https://github.com/MoonshotAI/minitriton) | 7 |
| nano-kpu — inference chip K3 designed | [MoonshotAI/nano-kpu](https://github.com/MoonshotAI/nano-kpu) | 7 |

## Direct antecedents worth reading next

- [Kimi Linear](https://arxiv.org/abs/2510.26692) — KDA's parent; the chunkwise form and UT transform
- [LatentMoE](https://arxiv.org/abs/2601.18089) — routing in a compressed latent space
- [Gated DeltaNet](https://openreview.net/forum?id=r8H7xhYPwz) — delta rule + gating
- [DeepSeek-V2](https://arxiv.org/abs/2405.04434) — MLA
- [DeepSeek-V3](https://arxiv.org/abs/2412.19437) — auxiliary-loss-free bias routing, the QB baseline
- [Muon](https://kellerjordan.github.io/posts/muon/) — the optimizer K3 makes per-head
- Attention Residuals — Kimi Team preprint 2026, cited as [58]; no public URL in the report
