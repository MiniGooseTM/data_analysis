# Results

## Test conditions

| ID | Target velocity | Surface |
|----|-----------------|---------|
| C1 |        0        |rpl floor|
| C2 |        0        |rpl meeting room carper|
| C3 |                 |         |
| C4 |                 |         |

## Tracking error (MAE)

Linear velocity MAE in m/s, angular velocity MAE in rad/s.

| Policy     | C1 lin MAE | C1 ang MAE | C2 lin MAE | C2 ang MAE | C3 lin MAE | C3 ang MAE | C4 lin MAE | C4 ang MAE |
|------------|-----------:|-----------:|-----------:|-----------:|-----------:|-----------:|-----------:|-----------:|
| no_randomization|  |            |            |            |            |            |            |            |
| policy_2   |            |            |            |            |            |            |            |            |
| policy_3   |            |            |            |            |            |            |            |            |

## Policy specification
- no_randomization - trained for 30 000 000 steps, no domain randomization, only linear and angular velocity commands