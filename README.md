# 4x3_world_AI_Textbook

This code is to calculate the utilities at each state for "4x3 world" problem using Value Iteration. <br />
The more detailed information is at the page 645 to 653 in AI textbook. <br />
Download Link : https://www.ics.uci.edu/~rickl/courses/cs-171/aima-resources/Artificial%20Intelligence%20A%20Modern%20Approach%20(3rd%20Edition).pdf <br />

## Visualization of 4x3 world problem 

```
3 [__, __, __, +1]
2 [__, wa, __, -1]
1 [sp, __, __, __]
    1   2   3   4
```

- Description <br />
    sp : a start point (1, 1) <br />
    wa : a wall which is blocked <br />
    +1 : reward +1 <br />
    -1 : reward -1 <br />
    __ : a space which the agent can stay <br />

    coordinate order: left, up i.e. horizon first, verticality second <br />
    e.g., +1 => (4, 3)
          -1 => (4, 2)

- Action <br />
    An agent can act to go one step. <br />
    The cases are "Up", "Right", "Down", "Left" <br />

- Noisy <br />
    An agent do action to go one step, and it has noisies, <br />
    Go straight with probability 0.8             P(gs) = 0.8   That is an intention to go forward <br />
    Left unintentionally with probability 0.1    P(l) = 0.1    An noisy to go left unintentionally <br />
    Right unintentionally with probability 0.1   P(r) = 0.1    Ao noisy to go right unintentionally <br />

- Reward <br />
    Except the states (4,3) and (4,2) (those are +1, -1), R(s) = -0.04

- Optimal Policy <br />
    An Optimal policy for the R(s) = -0.04 is

    ```
    3 [Ri, Ri, Ri,  1]
    2 [Up, xx, Up, -1]
    1 [Up, Le, Le, Le]
        1   2   3   4
    ```
    Ri : Right, Le : Left

### Value Iteration Pseudo code

refer to the page 653 in AI textbook.

### Value Iteration Results

```
Iteration to converge : 33

[0.811  0.868  0.918   1.0]
[0.761  _____  0.66   -1.0]
[0.705  0.655  0.611 0.387]
```
