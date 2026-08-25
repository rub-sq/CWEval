# Scripts

Everything written for the thesis, in the order the study ran. All paths are
relative to the repository root. The three shell scripts drive the experiments
and need the evaluation container; the Python scripts read the stored results
and are read-only regarding the evaluation data.

The results themselves live under `evals/` and are not part of this repository.
They are shipped separately as an archive and have to be unpacked to `evals/`
before any of the Python scripts can run.

## Running the experiments

| Script | Delivers | Output |
|---|---|---|
| `../run_models.sh` | the generation runs of the five proprietary models, with the sample count, temperature, prompt variant and token budget that Section 4.3 states | `evals/eval_<model>/` |
| `../run_eval.sh` | one evaluation pass over the generated code, in the environment Section 4.4 describes. Run once with an empty `RESTORE_FROM` for Run A, then again with a different `OUT_DIR` and `RESTORE_FROM` pointing at Run A for the second pass that Section 4.4 uses to measure run-to-run variation | `evals/eval_<model>/res_all.json`, `evals/_run_<X>_<date>/`, `evals/eval_logs/` |
| `../benchmark_check.sh` | the sanity check of Section 4.4 that every reference implementation passes its oracles, and the separate reference run of the three `cwe_1333_0` variants behind Section 3.7 | `evals/audit/benchmark_check.log` |

## Reading the results

| Script | Delivers | Output |
|---|---|---|
| `data_basis_report.py` | Section 5.1, Data Basis and Evaluation Coverage, and the reporting thresholds used throughout Chapter 5. Also the two per-model figures that Section 3.6 and Chapter 7 rest on, namely the share of samples that are secure without being plausible and the smallest number of graded samples any single task carries. | `evals/data_basis/` |
| `passk_report.py` | func@k and func-sec@k per model over all 119 tasks, at every $k$ its arm supports. Everything Chapter 5 rests on. Gap@k is formed from these two, and the 2024 baseline values come from Table I of the CWEval paper, which the table generator carries as a literal. | `evals/passk_all_models.csv` |
| `flip_report.py` | Section 5.3, Task-Level Security Regressions. PFR, NFR and the two noise floors per pair and scope, plus the share of each rate that the five largest tasks carry. | `evals/flip_report.csv`, `evals/flip_concentration.csv` |
| `breakdown_report.py` | Section 5.4, Security Failures by Programming Language, and Section 5.5, Security Failures by Weakness Type. Also the task variants on which no sample of any current model is plausible, which Section 5.5 has to qualify before it reads such a rate as a property of a model. | `evals/breakdowns/` |
| `audit_report.py` | the token statistics of Section 5.7, including the longest response and the single response carrying reasoning tokens. The response shapes behind Section 3.7. And the responses that lost their code to the token budget together with their length, which Chapter 7 discusses under internal validity. | `evals/audit/` |
| `check_parser_equivalence.py` | the measurement Chapter 7 reports under internal validity, that both parsers select the same code block for all 11,900 responses of the five proprietary models | stdout |
| `redos_oracle_check.py` | the reference-test counts and the checker verdicts behind Section 3.7. Needs `timeout` from GNU coreutils and therefore the evaluation container. | `evals/audit/redos_oracle.csv` |

Section 5.6 is deliberately not backed by a script. It is an analysis of
individual artifacts, selected by the criteria of Section 4.8 from the values
already reported in Sections 5.4 and 5.5, and each artifact was read directly
from the corresponding `generated_N/res.json`.

Scripts shipped with CWEval and not written for this thesis: `rmc.sh`,
`table_report.py`, `test_all.sh`. Written for this thesis but committed with
the framework changes rather than here: `token_report.py`, which produces the
per-model and per-language token averages that Section 3.7 refers to.
