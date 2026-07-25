# Validated alternative-objective winners

Improvement sign convention: positive always means better (throughput `cand/base-1`, latency `1-cand/base`). Classification uses bootstrap 95 % CIs, never a fixed threshold.


## lfm25 — R_concurrent_decode

| objective_role                   |   config_id |   cap |   chunk | policy   |   mem |   request_throughput_delta |   ttft_p95_ms_delta |   tpot_p95_ms_delta |   e2e_p95_ms_delta | classification        |   repeat_count |
|:---------------------------------|------------:|------:|--------:|:---------|------:|---------------------------:|--------------------:|--------------------:|-------------------:|:----------------------|---------------:|
| cookbook                         |          74 |    32 |      -1 | lpm      |  0.85 |                        0   |                 0   |                 0   |                0   | FLAT                  |              5 |
| request_throughput_best          |         100 |    48 |      -1 | fcfs     |  0.75 |                        1.5 |                 3.4 |                 1.1 |                1.3 | STRICT_ALL_METRIC_WIN |              5 |
| ttft_p95_best                    |         125 |    64 |      -1 | fcfs     |  0.8  |                        0.4 |                 1.4 |                 0.3 |                0.2 | WIN                   |              5 |
| tpot_p95_best                    |           5 |     8 |      -1 | fcfs     |  0.8  |                      -64.1 |             -2131.1 |                24.3 |             -180   | TRADE-OFF             |              5 |
| e2e_p95_best                     |         100 |    48 |      -1 | fcfs     |  0.75 |                        1.5 |                 3.4 |                 1.1 |                1.3 | STRICT_ALL_METRIC_WIN |              5 |
| constrained_throughput_best_3pct |         100 |    48 |      -1 | fcfs     |  0.75 |                        1.5 |                 3.4 |                 1.1 |                1.3 | STRICT_ALL_METRIC_WIN |              5 |
| maximin_balanced_best            |         100 |    48 |      -1 | fcfs     |  0.75 |                        1.5 |                 3.4 |                 1.1 |                1.3 | STRICT_ALL_METRIC_WIN |              5 |
| pareto_knee_candidate            |         100 |    48 |      -1 | fcfs     |  0.75 |                        1.5 |                 3.4 |                 1.1 |                1.3 | STRICT_ALL_METRIC_WIN |              5 |
| strict_all_metric_candidate      |         100 |    48 |      -1 | fcfs     |  0.75 |                        1.5 |                 3.4 |                 1.1 |                1.3 | STRICT_ALL_METRIC_WIN |              5 |

## lfm25 — R_long_prefill

| objective_role                   |   config_id |   cap |   chunk | policy   |   mem |   request_throughput_delta |   ttft_p95_ms_delta |   tpot_p95_ms_delta |   e2e_p95_ms_delta | classification        |   repeat_count |
|:---------------------------------|------------:|------:|--------:|:---------|------:|---------------------------:|--------------------:|--------------------:|-------------------:|:----------------------|---------------:|
| cookbook                         |          74 |    32 |      -1 | lpm      |  0.85 |                        0   |                 0   |                 0   |                0   | FLAT                  |              5 |
| request_throughput_best          |          60 |    24 |    2048 | fcfs     |  0.75 |                       58.1 |                54.9 |                -6   |               37.4 | STRICT_ALL_METRIC_WIN |              5 |
| ttft_p95_best                    |          60 |    24 |    2048 | fcfs     |  0.75 |                       58.1 |                54.9 |                -6   |               37.4 | STRICT_ALL_METRIC_WIN |              5 |
| tpot_p95_best                    |          42 |    16 |    8192 | lpm      |  0.85 |                       20.6 |                25.2 |                 6.9 |               17   | WIN                   |              5 |
| e2e_p95_best                     |          60 |    24 |    2048 | fcfs     |  0.75 |                       58.1 |                54.9 |                -6   |               37.4 | STRICT_ALL_METRIC_WIN |              5 |
| constrained_throughput_best_3pct |          60 |    24 |    2048 | fcfs     |  0.75 |                       58.1 |                54.9 |                -6   |               37.4 | STRICT_ALL_METRIC_WIN |              5 |
| maximin_balanced_best            |          42 |    16 |    8192 | lpm      |  0.85 |                       20.6 |                25.2 |                 6.9 |               17   | STRICT_ALL_METRIC_WIN |              5 |
| pareto_knee_candidate            |          59 |    24 |    2048 | lpm      |  0.9  |                       63.1 |                55.5 |                -3.6 |               37.7 | STRICT_ALL_METRIC_WIN |              5 |
| strict_all_metric_candidate      |          60 |    24 |    2048 | fcfs     |  0.75 |                       58.1 |                54.9 |                -6   |               37.4 | STRICT_ALL_METRIC_WIN |              5 |

## lfm25 — R_medium_balanced

| objective_role                   |   config_id |   cap |   chunk | policy   |   mem |   request_throughput_delta |   ttft_p95_ms_delta |   tpot_p95_ms_delta |   e2e_p95_ms_delta | classification        |   repeat_count |
|:---------------------------------|------------:|------:|--------:|:---------|------:|---------------------------:|--------------------:|--------------------:|-------------------:|:----------------------|---------------:|
| cookbook                         |          74 |    32 |      -1 | lpm      |  0.85 |                        0   |                 0   |                 0   |                0   | FLAT                  |              5 |
| request_throughput_best          |          32 |    16 |    2048 | lpm      |  0.75 |                       -1.4 |                -6.2 |                -1.1 |               -0.3 | REGRESSION            |              5 |
| ttft_p95_best                    |         170 |   128 |      -1 | lpm      |  0.85 |                        0.6 |                 3.6 |                 0.1 |                0.5 | STRICT_ALL_METRIC_WIN |              5 |
| tpot_p95_best                    |         168 |   128 |      -1 | lpm      |  0.75 |                       -0.1 |                 1   |                 0   |                0.1 | FLAT                  |              5 |
| e2e_p95_best                     |          36 |    16 |    2048 | fcfs     |  0.75 |                        1.2 |                -2.4 |                 1.4 |                2.5 | STRICT_ALL_METRIC_WIN |              5 |
| constrained_throughput_best_3pct |         118 |    48 |    8192 | fcfs     |  0.85 |                        1.2 |                 1.6 |                 0.1 |                0.3 | STRICT_ALL_METRIC_WIN |              5 |
| maximin_balanced_best            |         118 |    48 |    8192 | fcfs     |  0.85 |                        1.2 |                 1.6 |                 0.1 |                0.3 | STRICT_ALL_METRIC_WIN |              5 |
| pareto_knee_candidate            |         118 |    48 |    8192 | fcfs     |  0.85 |                        1.2 |                 1.6 |                 0.1 |                0.3 | STRICT_ALL_METRIC_WIN |              5 |
| strict_all_metric_candidate      |         118 |    48 |    8192 | fcfs     |  0.85 |                        1.2 |                 1.6 |                 0.1 |                0.3 | STRICT_ALL_METRIC_WIN |              5 |

## lfm25 — R_short_decode

| objective_role                   |   config_id |   cap |   chunk | policy   |   mem |   request_throughput_delta |   ttft_p95_ms_delta |   tpot_p95_ms_delta |   e2e_p95_ms_delta | classification   |   repeat_count |
|:---------------------------------|------------:|------:|--------:|:---------|------:|---------------------------:|--------------------:|--------------------:|-------------------:|:-----------------|---------------:|
| cookbook                         |          74 |    32 |      -1 | lpm      |  0.85 |                        0   |                 0   |                 0   |                0   | FLAT             |              5 |
| request_throughput_best          |         167 |    96 |    8192 | fcfs     |  0.9  |                       -0.2 |                12.9 |                -0.6 |                0.3 | REGRESSION       |              5 |
| ttft_p95_best                    |          21 |     8 |    8192 | fcfs     |  0.8  |                       -0   |                12.3 |                -0.1 |                0.5 | REGRESSION       |              5 |
| tpot_p95_best                    |         105 |    48 |    2048 | lpm      |  0.8  |                        0   |                13   |                -0.1 |                0.5 | REGRESSION       |              5 |
| e2e_p95_best                     |           5 |     8 |      -1 | fcfs     |  0.8  |                       -0.1 |                18.6 |                -0.4 |                0.6 | TRADE-OFF        |              5 |
| constrained_throughput_best_3pct |         167 |    96 |    8192 | fcfs     |  0.9  |                       -0.2 |                12.9 |                -0.6 |                0.3 | REGRESSION       |              5 |
| maximin_balanced_best            |           9 |     8 |    2048 | lpm      |  0.8  |                       -0.5 |                 9.3 |                -0.4 |                0.1 | REGRESSION       |              5 |
| pareto_knee_candidate            |           5 |     8 |      -1 | fcfs     |  0.8  |                       -0.1 |                18.6 |                -0.4 |                0.6 | TRADE-OFF        |              5 |
| strict_all_metric_candidate      |          78 |    32 |      -1 | fcfs     |  0.85 |                       -0.5 |                10.6 |                -1.4 |               -0.8 | REGRESSION       |              5 |

## lfm25 — shared_prefix

| objective_role                   |   config_id |   cap |   chunk | policy   |   mem |   request_throughput_delta |   ttft_p95_ms_delta |   tpot_p95_ms_delta |   e2e_p95_ms_delta | classification        |   repeat_count |
|:---------------------------------|------------:|------:|--------:|:---------|------:|---------------------------:|--------------------:|--------------------:|-------------------:|:----------------------|---------------:|
| cookbook                         |          74 |    32 |      -1 | lpm      |  0.85 |                        0   |                 0   |                 0   |                0   | FLAT                  |              5 |
| request_throughput_best          |         152 |    96 |    2048 | lpm      |  0.75 |                       89.1 |                93.5 |                -4.4 |               73   | TRADE-OFF             |              5 |
| ttft_p95_best                    |         152 |    96 |    2048 | lpm      |  0.75 |                       89.1 |                93.5 |                -4.4 |               73   | TRADE-OFF             |              5 |
| tpot_p95_best                    |           8 |     8 |    2048 | lpm      |  0.75 |                      -44.9 |               -89.3 |                46.8 |              -66.6 | TRADE-OFF             |              5 |
| e2e_p95_best                     |         152 |    96 |    2048 | lpm      |  0.75 |                       89.1 |                93.5 |                -4.4 |               73   | TRADE-OFF             |              5 |
| constrained_throughput_best_3pct |         106 |    48 |    2048 | lpm      |  0.85 |                       54.7 |                67.1 |                 5.9 |               53.4 | STRICT_ALL_METRIC_WIN |              5 |
| maximin_balanced_best            |          83 |    32 |    2048 | lpm      |  0.9  |                       45.8 |                54   |                28   |               47.1 | STRICT_ALL_METRIC_WIN |              5 |
| pareto_knee_candidate            |         179 |   128 |    2048 | lpm      |  0.9  |                       90.3 |                93.8 |                -4.2 |               73.3 | TRADE-OFF             |              5 |
| strict_all_metric_candidate      |         106 |    48 |    2048 | lpm      |  0.85 |                       54.7 |                67.1 |                 5.9 |               53.4 | STRICT_ALL_METRIC_WIN |              5 |

## lfm25 — tool_agent

| objective_role                   |   config_id |   cap |   chunk | policy   |   mem |   request_throughput_delta |   ttft_p95_ms_delta |   tpot_p95_ms_delta |   e2e_p95_ms_delta | classification        |   repeat_count |
|:---------------------------------|------------:|------:|--------:|:---------|------:|---------------------------:|--------------------:|--------------------:|-------------------:|:----------------------|---------------:|
| cookbook                         |          74 |    32 |      -1 | lpm      |  0.85 |                        0   |                 0   |                 0   |                0   | FLAT                  |              5 |
| request_throughput_best          |         105 |    48 |    2048 | lpm      |  0.8  |                        0   |                41.9 |              -114.1 |                6.7 | WIN                   |              5 |
| ttft_p95_best                    |          41 |    16 |    8192 | lpm      |  0.8  |                        0.2 |                29.2 |               -16.4 |                3.4 | STRICT_ALL_METRIC_WIN |              5 |
| tpot_p95_best                    |          17 |     8 |    8192 | lpm      |  0.8  |                       -0.3 |              -161.5 |                66.4 |              -10.5 | TRADE-OFF             |              5 |
| e2e_p95_best                     |         191 |   128 |    8192 | fcfs     |  0.9  |                       -0   |                46.6 |                40.6 |                8.4 | STRICT_ALL_METRIC_WIN |              5 |
| constrained_throughput_best_3pct |          68 |    24 |    8192 | fcfs     |  0.75 |                        0.1 |                35.8 |               -11.3 |                6.5 | WIN                   |              5 |
| maximin_balanced_best            |          68 |    24 |    8192 | fcfs     |  0.75 |                        0.1 |                35.8 |               -11.3 |                6.5 | STRICT_ALL_METRIC_WIN |              5 |
| pareto_knee_candidate            |          25 |    16 |      -1 | lpm      |  0.8  |                       -0   |                 3.3 |                24.2 |               -1.3 | TRADE-OFF             |              5 |
| strict_all_metric_candidate      |          68 |    24 |    8192 | fcfs     |  0.75 |                        0.1 |                35.8 |               -11.3 |                6.5 | STRICT_ALL_METRIC_WIN |              5 |

## qwen — R_concurrent_decode

| objective_role                   |   config_id |   cap |   chunk | policy   |   mem |   request_throughput_delta |   ttft_p95_ms_delta |   tpot_p95_ms_delta |   e2e_p95_ms_delta | classification        |   repeat_count |
|:---------------------------------|------------:|------:|--------:|:---------|------:|---------------------------:|--------------------:|--------------------:|-------------------:|:----------------------|---------------:|
| cookbook                         |          74 |    32 |      -1 | lpm      |  0.85 |                        0   |                 0   |                 0   |                0   | FLAT                  |              5 |
| request_throughput_best          |         174 |   128 |      -1 | fcfs     |  0.85 |                        0.4 |                 4.4 |                 0.1 |                0.3 | STRICT_ALL_METRIC_WIN |              5 |
| ttft_p95_best                    |         169 |   128 |      -1 | lpm      |  0.8  |                       -0.1 |                 0.9 |                -0.1 |               -0.1 | REGRESSION            |              5 |
| tpot_p95_best                    |           4 |     8 |      -1 | fcfs     |  0.75 |                      -61.9 |             -4560.7 |                30.5 |             -163.2 | TRADE-OFF             |              5 |
| e2e_p95_best                     |         169 |   128 |      -1 | lpm      |  0.8  |                       -0.1 |                 0.9 |                -0.1 |               -0.1 | REGRESSION            |              5 |
| constrained_throughput_best_3pct |         174 |   128 |      -1 | fcfs     |  0.85 |                        0.4 |                 4.4 |                 0.1 |                0.3 | STRICT_ALL_METRIC_WIN |              5 |
| maximin_balanced_best            |         179 |   128 |    2048 | lpm      |  0.9  |                        0.5 |                10   |                 0   |                0.4 | STRICT_ALL_METRIC_WIN |              5 |
| pareto_knee_candidate            |         174 |   128 |      -1 | fcfs     |  0.85 |                        0.4 |                 4.4 |                 0.1 |                0.3 | STRICT_ALL_METRIC_WIN |              5 |
| strict_all_metric_candidate      |         174 |   128 |      -1 | fcfs     |  0.85 |                        0.4 |                 4.4 |                 0.1 |                0.3 | STRICT_ALL_METRIC_WIN |              5 |

## qwen — R_long_prefill

| objective_role                   |   config_id |   cap |   chunk | policy   |   mem |   request_throughput_delta |   ttft_p95_ms_delta |   tpot_p95_ms_delta |   e2e_p95_ms_delta | classification        |   repeat_count |
|:---------------------------------|------------:|------:|--------:|:---------|------:|---------------------------:|--------------------:|--------------------:|-------------------:|:----------------------|---------------:|
| cookbook                         |          74 |    32 |      -1 | lpm      |  0.85 |                        0   |                 0   |                 0   |                0   | FLAT                  |              5 |
| request_throughput_best          |         120 |    64 |      -1 | lpm      |  0.75 |                       17.9 |                32.8 |                17.8 |               15.5 | STRICT_ALL_METRIC_WIN |              5 |
| ttft_p95_best                    |          98 |    48 |      -1 | lpm      |  0.85 |                       18.5 |                33.7 |                17.9 |               15.7 | STRICT_ALL_METRIC_WIN |              5 |
| tpot_p95_best                    |          71 |    24 |    8192 | fcfs     |  0.9  |                       17.3 |                32.5 |                17.9 |               15.2 | STRICT_ALL_METRIC_WIN |              5 |
| e2e_p95_best                     |          98 |    48 |      -1 | lpm      |  0.85 |                       18.5 |                33.7 |                17.9 |               15.7 | STRICT_ALL_METRIC_WIN |              5 |
| constrained_throughput_best_3pct |         120 |    64 |      -1 | lpm      |  0.75 |                       17.9 |                32.8 |                17.8 |               15.5 | STRICT_ALL_METRIC_WIN |              5 |
| maximin_balanced_best            |         120 |    64 |      -1 | lpm      |  0.75 |                       17.9 |                32.8 |                17.8 |               15.5 | STRICT_ALL_METRIC_WIN |              5 |
| pareto_knee_candidate            |         137 |    64 |    8192 | lpm      |  0.8  |                       20.3 |                37.4 |                17.9 |               17.3 | STRICT_ALL_METRIC_WIN |              5 |
| strict_all_metric_candidate      |         120 |    64 |      -1 | lpm      |  0.75 |                       17.9 |                32.8 |                17.8 |               15.5 | STRICT_ALL_METRIC_WIN |              5 |

## qwen — R_medium_balanced

| objective_role                   |   config_id |   cap |   chunk | policy   |   mem |   request_throughput_delta |   ttft_p95_ms_delta |   tpot_p95_ms_delta |   e2e_p95_ms_delta | classification        |   repeat_count |
|:---------------------------------|------------:|------:|--------:|:---------|------:|---------------------------:|--------------------:|--------------------:|-------------------:|:----------------------|---------------:|
| cookbook                         |          74 |    32 |      -1 | lpm      |  0.85 |                        0   |                 0   |                 0   |                0   | FLAT                  |              5 |
| request_throughput_best          |         165 |    96 |    8192 | fcfs     |  0.8  |                       -0.2 |                -2.1 |                -0   |               -0.3 | REGRESSION            |              5 |
| ttft_p95_best                    |         165 |    96 |    8192 | fcfs     |  0.8  |                       -0.2 |                -2.1 |                -0   |               -0.3 | REGRESSION            |              5 |
| tpot_p95_best                    |          72 |    32 |      -1 | lpm      |  0.75 |                        0.4 |                 3.7 |                 0.1 |                0.2 | STRICT_ALL_METRIC_WIN |              5 |
| e2e_p95_best                     |          83 |    32 |    2048 | lpm      |  0.9  |                        0.1 |                 1.7 |                 0   |               -0   | FLAT                  |              5 |
| constrained_throughput_best_3pct |         165 |    96 |    8192 | fcfs     |  0.8  |                       -0.2 |                -2.1 |                -0   |               -0.3 | REGRESSION            |              5 |
| maximin_balanced_best            |          83 |    32 |    2048 | lpm      |  0.9  |                        0.1 |                 1.7 |                 0   |               -0   | FLAT                  |              5 |
| pareto_knee_candidate            |          83 |    32 |    2048 | lpm      |  0.9  |                        0.1 |                 1.7 |                 0   |               -0   | FLAT                  |              5 |
| strict_all_metric_candidate      |         102 |    48 |      -1 | fcfs     |  0.85 |                       -0.3 |                -2.9 |                -0.3 |               -0.3 | REGRESSION            |              5 |

## qwen — R_short_decode

| objective_role                   |   config_id |   cap |   chunk | policy   |   mem |   request_throughput_delta |   ttft_p95_ms_delta |   tpot_p95_ms_delta |   e2e_p95_ms_delta | classification        |   repeat_count |
|:---------------------------------|------------:|------:|--------:|:---------|------:|---------------------------:|--------------------:|--------------------:|-------------------:|:----------------------|---------------:|
| cookbook                         |          74 |    32 |      -1 | lpm      |  0.85 |                        0   |                 0   |                 0   |                0   | FLAT                  |              5 |
| request_throughput_best          |         102 |    48 |      -1 | fcfs     |  0.85 |                        0.1 |                -4.2 |                 0.2 |               -0   | REGRESSION            |              5 |
| ttft_p95_best                    |         139 |    64 |    8192 | lpm      |  0.9  |                       -0.1 |                -0.9 |                 0   |               -0   | FLAT                  |              5 |
| tpot_p95_best                    |          67 |    24 |    8192 | lpm      |  0.9  |                       -0.2 |                -1.6 |                -0.1 |               -0.2 | REGRESSION            |              5 |
| e2e_p95_best                     |          39 |    16 |    2048 | fcfs     |  0.9  |                        0.4 |                 0.5 |                 0.3 |                0.3 | STRICT_ALL_METRIC_WIN |              5 |
| constrained_throughput_best_3pct |         102 |    48 |      -1 | fcfs     |  0.85 |                        0.1 |                -4.2 |                 0.2 |               -0   | REGRESSION            |              5 |
| maximin_balanced_best            |          39 |    16 |    2048 | fcfs     |  0.9  |                        0.4 |                 0.5 |                 0.3 |                0.3 | STRICT_ALL_METRIC_WIN |              5 |
| pareto_knee_candidate            |          98 |    48 |      -1 | lpm      |  0.85 |                        0.4 |                -5.1 |                 0.1 |                0.1 | TRADE-OFF             |              5 |
| strict_all_metric_candidate      |         102 |    48 |      -1 | fcfs     |  0.85 |                        0.1 |                -4.2 |                 0.2 |               -0   | REGRESSION            |              5 |

## qwen — shared_prefix

| objective_role                   |   config_id |   cap |   chunk | policy   |   mem |   request_throughput_delta |   ttft_p95_ms_delta |   tpot_p95_ms_delta |   e2e_p95_ms_delta | classification   |   repeat_count |
|:---------------------------------|------------:|------:|--------:|:---------|------:|---------------------------:|--------------------:|--------------------:|-------------------:|:-----------------|---------------:|
| cookbook                         |          74 |    32 |      -1 | lpm      |  0.85 |                        0   |                 0   |                 0   |                0   | FLAT             |              5 |
| request_throughput_best          |         184 |   128 |    8192 | lpm      |  0.75 |                       23.1 |                84.7 |               -48.1 |               22.9 | TRADE-OFF        |              5 |
| ttft_p95_best                    |         169 |   128 |      -1 | lpm      |  0.8  |                       21.7 |                83.2 |               -49   |               21.4 | TRADE-OFF        |              5 |
| tpot_p95_best                    |           9 |     8 |    2048 | lpm      |  0.8  |                      -63.7 |              -662.6 |                27.4 |             -388.4 | TRADE-OFF        |              5 |
| e2e_p95_best                     |         165 |    96 |    8192 | fcfs     |  0.8  |                       21.9 |                83.3 |               -49.1 |               21.5 | TRADE-OFF        |              5 |
| constrained_throughput_best_3pct |          77 |    32 |      -1 | fcfs     |  0.8  |                       -0.4 |                -6.1 |                 5.1 |               -0.7 | TRADE-OFF        |              5 |
| maximin_balanced_best            |          77 |    32 |      -1 | fcfs     |  0.8  |                       -0.4 |                -6.1 |                 5.1 |               -0.7 | TRADE-OFF        |              5 |
| pareto_knee_candidate            |         169 |   128 |      -1 | lpm      |  0.8  |                       21.7 |                83.2 |               -49   |               21.4 | TRADE-OFF        |              5 |
| strict_all_metric_candidate      |          77 |    32 |      -1 | fcfs     |  0.8  |                       -0.4 |                -6.1 |                 5.1 |               -0.7 | TRADE-OFF        |              5 |

## qwen — tool_agent

| objective_role                   |   config_id |   cap |   chunk | policy   |   mem |   request_throughput_delta |   ttft_p95_ms_delta |   tpot_p95_ms_delta |   e2e_p95_ms_delta | classification   |   repeat_count |
|:---------------------------------|------------:|------:|--------:|:---------|------:|---------------------------:|--------------------:|--------------------:|-------------------:|:-----------------|---------------:|
| cookbook                         |          74 |    32 |      -1 | lpm      |  0.85 |                        0   |                 0   |                 0   |                0   | FLAT             |              5 |
| request_throughput_best          |         139 |    64 |    8192 | lpm      |  0.9  |                       -0.2 |               -31   |              -195   |               -4   | REGRESSION       |              5 |
| ttft_p95_best                    |         115 |    48 |    8192 | lpm      |  0.9  |                       -0.2 |               -27.6 |              -168.9 |               -4.7 | REGRESSION       |              5 |
| tpot_p95_best                    |           5 |     8 |      -1 | fcfs     |  0.8  |                      -11.8 |             -3779.9 |                 7.5 |             -103.3 | TRADE-OFF        |              5 |
| e2e_p95_best                     |          98 |    48 |      -1 | lpm      |  0.85 |                       -0.1 |                -0.4 |                -3.4 |               -0.3 | FLAT             |              5 |
| constrained_throughput_best_3pct |          73 |    32 |      -1 | lpm      |  0.8  |                       -0.1 |               -41.2 |              -153   |               -4.7 | REGRESSION       |              5 |
| maximin_balanced_best            |          75 |    32 |      -1 | lpm      |  0.9  |                       -0   |                -4.7 |                -1.8 |               -0.6 | REGRESSION       |              5 |
| pareto_knee_candidate            |          48 |    24 |      -1 | lpm      |  0.75 |                       -0.1 |               -28.8 |               -62   |               -4.9 | REGRESSION       |              5 |
| strict_all_metric_candidate      |          75 |    32 |      -1 | lpm      |  0.9  |                       -0   |                -4.7 |                -1.8 |               -0.6 | REGRESSION       |              5 |
