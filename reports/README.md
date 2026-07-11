# Analysis Reports

Directory for generated analysis reports and summaries.

## Convention

Per CLAUDE.md Definition of Done, all changes should generate a summary report:

```
<YYYY-MM-DD_HHMM>_summaryOfUpdate_report.md
```

## Report Contents

Each report should include (as applicable):

- **What changed**: Brief description of the change
- **Files changed**: Path → purpose mapping
- **Tests + coverage**: Coverage % for changed lines (diff-coverage)
- **Code review result**: Findings and actions taken
- **Commands run**: What was actually executed (honest reporting)
- **Risks/Follow-ups**: Any identified risks or next steps
- **Definition of Done checklist**: Verification of all DoD criteria

## Example

```markdown
# 2026-07-11_1430_summaryOfUpdate_report.md

## What changed
Added initial data loading module for Vietnam stock prices.

## Files changed
- `src/data/load_prices.py` → Main data loading logic
- `tests/unit/test_load_prices.py` → Unit tests

## Tests + Coverage
- Coverage: 87.5% (diff-coverage)
- All tests passing: `pytest -v`

## Code Review
Findings:
- [LOW] Missing type hints for helper functions → Added
- [INFO] Consider caching → Added as follow-up

## Commands Run
```bash
pytest tests/                    # PASS
ruff check .                    # PASS
mypy src/                       # PASS
pytest -m smoke                 # PASS
```

## Definition of Done
- [x] Code directly satisfies request
- [x] Tests written and run (87.5% coverage)
- [x] All checks passed
- [x] Code reviewed and findings addressed
- [x] Summary report generated
- [x] Smoke tests passing

## Risks/Follow-ups
- Consider adding data caching for large datasets
- Add integration tests with real data files
```
