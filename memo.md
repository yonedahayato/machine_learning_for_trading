# L21 memo

1. 1st step
Select x1, x2, ,,, ,xn
predictive factors
1. 2nd step
Select y
These both become our data.
1. 3rd step
Time period, stock universe
1. 4th step
Train
1. 5th step
Predict

# L23 memo
1. 6th step
### metric
  1. RMS error
  1. Correlation

# L25

# L26 memo
## Q-Learning Recap
### Building a model
1. define states, actions, rewards
1. choose in-sample trading period
1. iterate:Q-table update
1. backtest
1. repat「3, 4」

### Testing a model
1. backtest on later data

# L27
## Q-Learn -> expensive
1. init Q-table
1. observe s
1. execute a, observe s', r
1. update Q with <s, a, s', r>
1. repeat 2, 3, 4

## Dyna-Q -> cheap
1. Learn modle(TR)
  1. T'[s, a, s'] = ?
  1. R'[s, a] = ?
1. Hallucinate experience
  1. s = random
  1. a = random
  1. s' = infer from T[]
  1. r = R[s, a]
1. update Q
  1. update Q w/<s, a, s', r>
