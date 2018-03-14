# 4x3_world_AI_Textbook

This code is to calculate the utilities at each state for "4x3 world" problem using Value Iteration. <br />
The more detailed information is at the page 645 to 653 in AI textbook. <br />

Download Link : https://www.ics.uci.edu/~rickl/courses/cs-171/aima-resources/Artificial%20Intelligence%20A%20Modern%20Approach%20(3rd%20Edition).pdf

## Visualization of 4x3 world problem 

```
3 [__, __, __, +1]
2 [__, wa, __, -1]
1 [sp, __, __, __]
    1   2   3   4
```

### Description

sp : a start point (1, 1)

wa : a wall which is blocked

+1 : reward +1

-1 : reward -1

__ : a space which the agent can stay

horizon first, vertical second i.e. coordinate order: left, up

e.g., +1 => (4, 3)
      -1 => (4, 2)

### Action

An agent can act to go one step.

The cases are "Up", "Right", "Down", "Left"

### Noisy

An agent do action to go one step, and it has noisies,

Go straight with probability 0.8             P(gs) = 0.8   That is an intention to go forward

Left unintentionally with probability 0.1    P(l) = 0.1    An noisy to go left unintentionally

Right unintentionally with probability 0.1   P(r) = 0.1    Ao noisy to go right unintentionally

### Reward

Except the states (4,3) and (4,2) (those are +1, -1), R(s) = -0.04

### Optimal Policy

An Optimal policy for the R(s) = -0.04 is

```
3 [Ri, Ri, Ri,  1]
2 [Up, xx, Up, -1]
1 [Up, Le, Le, Le]
    1   2   3   4
```

### Value Iteration Pseudo code
refer to the page 653 in AI textbook.
