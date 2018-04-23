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

# L25 Reinforcement learning
## The RL problem
1. Environment = the market
1. actions = buying, selling, holding
1. state = factors about our stocks that we might observe and know about
1. reward = the return we get for making the proper trades

## Trading as an RL problem
| |S|A|R|
|:--|:--:|:--:|:--:|
|BUY|   | v |   |
|SELL|   | v |   |
|Holding Long| v |  |   |
|Bollinger value| v |  |  |
|return from trade|   |  | v |
|daily return| v |  | v ||

## mapping trading to RL
state -> market features, prices, hold or not hold  
action -> buy, sell, do nothing

## Markov decision problem
1. set of states S
1. set of actions A
1. Transition function T[s, a, s']
1. Reward function R[s, a]

find policy that will maximize reward

1. policy iteration
2. value iteration

## unknown transition and rewards
1. model based reinforcement learning  
        build model of transition function
        build model of reward function

        value iteration
        policy iteration

2. model free
        Q-learning

## What to optimize?
1. infinite horizon
2. finite horizon
3. discounted reward

## Which approach gets $1M

## summary
1. RL algos solve MDP
2. S, A, T[s, a, s'], R[s, a]
3. find pi(s) -> a
4. map trading to RL
        transition function is the market
        reward function is how much money we get at the end of a trade

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
