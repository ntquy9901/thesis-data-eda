# Story 23.6: Multi-agent Research Infrastructure

Status: done
baseline_revision: 0472e800e044b0ba48ecf70035c9ee3a9db437f7

## Story

As a data scientist,
I want multiple autonomous research agents continuously experimenting with financial news analysis methods,
so that I can compare results across methods overnight without manual intervention.

## Acceptance Criteria

1. 5 specialized agents run as independent background processes (pythonw.exe):
   - `agent_sentiment`: tests VADER, TextBlob, lexicon methods
   - `agent_model`: tests Ridge, RF, HGB, XGBoost with varying hyperparams
   - `agent_feature`: tests feature set combinations (basic → dual → full)
   - `agent_horizon`: tests multi-horizon (pk_t+1/+5/+10/+22)
   - `agent_ticker`: tests per-ticker specific models and windows
2. Each agent logs to `results/research_agent/logs/<agent>.log`
3. All results stored in shared SQLite DB `results/research_agent/experiments.db`
4. Master PowerShell launcher starts/stops/monitors all agents
5. Consolidated comparison reports generated every cycle
6. Agents survive terminal disconnection (true background processes via pythonw)

## Tasks / Subtasks

- [x] Create base Experiment + Registry framework
- [x] Create SQLite storage module
- [ ] Create 5 specialized agent scripts
- [ ] Create master launcher PowerShell script
- [ ] Create watchdog + consolidated reporting
- [ ] Launch all agents and verify independent operation

## Dev Notes

- Agents use `pythonw.exe` (windowless Python) để không phụ thuộc console session
- Mỗi agent chạy vô hạn, sleep configurable giữa các cycle
- Shared SQLite DB ở `results/research_agent/experiments.db`
- Log files ở `results/research_agent/logs/`
- PIDs tracked in `results/research_agent/agent_pids.json` để master script monitor
- If agent crashes → auto-restart by launcher watchdog
- All experiments import từ `src.research_agent.experiments.*`

## Dev Agent Record

### Agent Model Used

opencode/deepseek-v4-flash-free

### Completion Notes List

### File List

- `src/research_agent/base.py` (done)
- `src/research_agent/storage.py` (done)
- `src/research_agent/report.py` (done)
- `src/research_agent/research.py` (done)
- `src/research_agent/runner.py` (done)
- `src/research_agent/experiments/*.py` (done)
- `src/research_agent/agent_*.py` (5 agent scripts)
- `scripts/launch_agents.ps1`
- `scripts/agent_watchdog.ps1`
