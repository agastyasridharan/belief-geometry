# Belief State Visualization: A Mathematical Explainer

*Created by Agastya Sridharan*

This document explains the mathematics behind the interactive belief state visualizations, covering both the MESS3 process and the Golden Mean × Even hierarchical process.

---

## Table of Contents

1. [What is a Belief State?](#1-what-is-a-belief-state)
2. [Hidden Markov Models (HMMs)](#2-hidden-markov-models-hmms)
3. [The Belief Simplex](#3-the-belief-simplex)
4. [Bayesian Belief Updates](#4-bayesian-belief-updates)
5. [Tab 1: The MESS3 Process](#5-tab-1-the-mess3-process)
6. [Tab 2: Golden Mean × Even Process](#6-tab-2-golden-mean--even-process)
7. [Understanding the Visualizations](#7-understanding-the-visualizations)
8. [Why Fractals Emerge](#8-why-fractals-emerge)

---

## 1. What is a Belief State?

A **belief state** represents an observer's uncertainty about a hidden variable, expressed as a probability distribution.

Imagine you're watching a sequence of colored lights flash: red, green, blue, red, red, blue... You suspect there's a hidden mechanism determining these colors, but you can't see it directly. Your **belief state** captures your current best guess about what that hidden mechanism is doing.

Formally, if the hidden state can take values in a finite set $\mathcal{S} = \{s_1, s_2, \ldots, s_n\}$, then the belief state is:

$$\mathbf{b}(t) = \begin{pmatrix} P(S_t = s_1 \mid y_{1:t}) \\ P(S_t = s_2 \mid y_{1:t}) \\ \vdots \\ P(S_t = s_n \mid y_{1:t}) \end{pmatrix}$$

where $y_{1:t} = (y_1, y_2, \ldots, y_t)$ denotes all observations up to time $t$.

**Key properties:**
- Each component $b_i(t) \geq 0$
- The components sum to 1: $\sum_i b_i(t) = 1$
- The belief state is a **sufficient statistic** for prediction: knowing $\mathbf{b}(t)$ is as good as knowing the entire history $y_{1:t}$

---

## 2. Hidden Markov Models (HMMs)

A **Hidden Markov Model** consists of:

1. **Hidden states** $S_t \in \mathcal{S}$ that evolve according to a Markov chain
2. **Observations** $Y_t \in \mathcal{Y}$ that are generated probabilistically from the current hidden state

The model is specified by two matrices:

### Transition Matrix $\mathbf{T}$

$$T_{ij} = P(S_{t+1} = j \mid S_t = i)$$

This gives the probability of moving from state $i$ to state $j$. Each row sums to 1.

### Emission Matrix $\mathbf{E}$

$$E_{ij} = P(Y_t = j \mid S_t = i)$$

This gives the probability of observing output $j$ when in hidden state $i$. Each row sums to 1.

### The Inference Problem

Given a sequence of observations $y_1, y_2, \ldots, y_t$, we want to compute the belief state $\mathbf{b}(t)$—our posterior distribution over hidden states.

---

## 3. The Belief Simplex

Since belief states are probability distributions, they live in a constrained space called a **simplex**.

### Definition

The $(n-1)$-simplex is the set of all probability distributions over $n$ outcomes:

$$\Delta^{n-1} = \left\{ \mathbf{b} \in \mathbb{R}^n : b_i \geq 0, \sum_{i=1}^n b_i = 1 \right\}$$

### Geometric Interpretation

| States | Simplex | Geometry |
|--------|---------|----------|
| 2 | $\Delta^1$ | Line segment |
| 3 | $\Delta^2$ | Triangle |
| 4 | $\Delta^3$ | Tetrahedron |
| $n$ | $\Delta^{n-1}$ | $(n-1)$-dimensional polytope |

The **vertices** of the simplex correspond to **certainty**—being 100% sure the system is in a particular state.

For a 3-state system:
- Vertex $(1, 0, 0)$: certain the hidden state is $s_1$
- Vertex $(0, 1, 0)$: certain the hidden state is $s_2$
- Vertex $(0, 0, 1)$: certain the hidden state is $s_3$
- Center $(1/3, 1/3, 1/3)$: maximum uncertainty (uniform distribution)

---

## 4. Bayesian Belief Updates

When a new observation $y_t$ arrives, we update our belief in two steps:

### Step 1: Bayes' Rule (Observation Update)

Incorporate the likelihood of seeing $y_t$ from each state:

$$\tilde{b}_i = b_i(t-1) \cdot P(y_t \mid S_t = i) = b_i(t-1) \cdot E_{i, y_t}$$

This is element-wise multiplication of the prior belief with the emission column for observation $y_t$.

### Step 2: Normalization

Ensure the result is a valid probability distribution:

$$b_i^{\text{posterior}} = \frac{\tilde{b}_i}{\sum_j \tilde{b}_j}$$

### Step 3: Chapman-Kolmogorov (Transition Propagation)

Propagate the belief forward through the transition dynamics:

$$b_i(t) = \sum_j b_j^{\text{posterior}} \cdot T_{ji}$$

### Combined Update (Matrix Form)

$$\mathbf{b}(t) = \text{normalize}\Big( (\mathbf{b}(t-1) \odot \mathbf{E}_{:, y_t}) \cdot \mathbf{T} \Big)$$

where $\odot$ denotes element-wise multiplication and $\mathbf{E}_{:, y_t}$ is the column of the emission matrix for observation $y_t$.

---

## 5. Tab 1: The MESS3 Process

### Process Definition

MESS3 is a symmetric 3-state HMM with:

**Transition Matrix** (parameterized by self-loop probability $p$):

$$\mathbf{T} = \begin{pmatrix} p & \frac{1-p}{2} & \frac{1-p}{2} \\ \frac{1-p}{2} & p & \frac{1-p}{2} \\ \frac{1-p}{2} & \frac{1-p}{2} & p \end{pmatrix}$$

Default: $p = 0.90$, so states are "sticky" (90% chance to stay, 5% chance to transition to each other state).

**Emission Matrix** (parameterized by accuracy $\alpha$):

$$\mathbf{E} = \begin{pmatrix} \alpha & \frac{1-\alpha}{2} & \frac{1-\alpha}{2} \\ \frac{1-\alpha}{2} & \alpha & \frac{1-\alpha}{2} \\ \frac{1-\alpha}{2} & \frac{1-\alpha}{2} & \alpha \end{pmatrix}$$

Default: $\alpha = 0.85$, so each state preferentially emits its "own" observation (state 0 usually emits 0, etc.).

### Visualization

The belief state has 3 components summing to 1, so it lives in a **2-simplex (triangle)**.

We visualize this directly:
- The three vertices represent certainty about each hidden state ($S_0$, $S_1$, $S_2$)
- The cyan dot shows the current belief state
- The orange circle marks the true hidden state
- Purple points show the fractal attractor (the long-run distribution of beliefs)

### Coordinates

We use a standard 2D projection of the simplex:
- $x = b_0 + 0.5 \cdot b_1$
- $y = b_1$

This maps the simplex to an upward-pointing triangle.

---

## 6. Tab 2: Golden Mean × Even Process

This is a **hierarchical** or **compositional** process with two coupled components.

### The Driver: Golden Mean Process

The Golden Mean process has 2 states (A and B) and emits a **hidden** signal $x_t \in \{0, 1\}$.

**Key constraint:** No consecutive 1s are allowed.

**Dynamics:**
- From state A: emit 0 and stay in A (prob $1-p$), OR emit 1 and go to B (prob $p$)
- From state B: emit 0 and go to A (prob 1)—**must** emit 0

The transition-emission structure:

| From | To | Emission $x$ | Probability |
|------|-----|--------------|-------------|
| A | A | 0 | $1 - p$ |
| A | B | 1 | $p$ |
| B | A | 0 | 1 |

**Why "Golden Mean"?** The name comes from the constraint matrix having the golden ratio as an eigenvalue. The process generates the "Fibonacci" constraint: no two 1s in a row.

### The Transducer: Switching Even Process

The Even process has 2 states (A = Even, B = Odd) and emits the **observed** signal $y_t \in \{0, 1\}$.

**Key constraint:** 1s come in runs of even length (0, 2, 4, ...).

**Switching behavior:** The transition probabilities depend on the driver's hidden output $x_t$:
- When $x_t = 1$: use parameter $p_1$
- When $x_t = 0$: use parameter $p_2$

**Dynamics** (for parameter $p_i$):
- From state A (Even): emit 0 and stay in A (prob $1-p_i$), OR emit 1 and go to B (prob $p_i$)
- From state B (Odd): emit 1 and go to A (prob 1)—**must** emit 1

The constraint ensures that once you start emitting 1s (entering state B), you must emit at least one more 1 before you can emit 0 again.

### The Joint State Space

Since the driver has 2 states and the transducer has 2 states, the **joint** hidden state has $2 \times 2 = 4$ possibilities:

| Index | Joint State | Meaning |
|-------|-------------|---------|
| 0 | (A, A) | Driver in A, Transducer in A (Even) |
| 1 | (A, B) | Driver in A, Transducer in B (Odd) |
| 2 | (B, A) | Driver in B, Transducer in A (Even) |
| 3 | (B, B) | Driver in B, Transducer in B (Odd) |

The belief state is now a 4-component vector:

$$\mathbf{b}(t) = \begin{pmatrix} P(\text{Driver}=A, \text{Trans}=A \mid y_{1:t}) \\ P(\text{Driver}=A, \text{Trans}=B \mid y_{1:t}) \\ P(\text{Driver}=B, \text{Trans}=A \mid y_{1:t}) \\ P(\text{Driver}=B, \text{Trans}=B \mid y_{1:t}) \end{pmatrix}$$

### The Joint Transition-Emission Tensor

Because the transducer's parameters depend on the driver's emission $x_t$, we need to marginalize over the hidden $x_t$ to get the joint dynamics.

Define $T[y, s, s']$ = probability of emitting $y$ and transitioning from joint state $s$ to joint state $s'$:

$$T[y, s, s'] = \sum_{x \in \{0,1\}} P(s'_{\text{driver}}, x \mid s_{\text{driver}}) \cdot P(s'_{\text{trans}}, y \mid s_{\text{trans}}, x)$$

**Worked example: From state (A, A)**

The driver is in A, the transducer is in A.

| Driver transition | $x$ | Prob | Trans transition | $y$ | Prob | Combined | Next state |
|-------------------|-----|------|------------------|-----|------|----------|------------|
| A → A | 0 | $1-p$ | A → A | 0 | $1-p_2$ | $(1-p)(1-p_2)$ | (A,A) |
| A → A | 0 | $1-p$ | A → B | 1 | $p_2$ | $(1-p)p_2$ | (A,B) |
| A → B | 1 | $p$ | A → A | 0 | $1-p_1$ | $p(1-p_1)$ | (B,A) |
| A → B | 1 | $p$ | A → B | 1 | $p_1$ | $p \cdot p_1$ | (B,B) |

So from (A,A):
- $T[0, \text{(A,A)}, \text{(A,A)}] = (1-p)(1-p_2)$
- $T[0, \text{(A,A)}, \text{(B,A)}] = p(1-p_1)$
- $T[1, \text{(A,A)}, \text{(A,B)}] = (1-p)p_2$
- $T[1, \text{(A,A)}, \text{(B,B)}] = p \cdot p_1$

The full tensor is computed similarly for all 4 starting states.

### Belief Update for the Joint Process

Given prior $\mathbf{b}(t-1)$ and observation $y_t$:

$$\tilde{b}_{s'}(t) = \sum_{s} b_s(t-1) \cdot T[y_t, s, s']$$

$$b_{s'}(t) = \frac{\tilde{b}_{s'}(t)}{\sum_{s''} \tilde{b}_{s''}(t)}$$

This is the standard HMM belief update, but applied to the 4-state joint process.

---

## 7. Understanding the Visualizations

### Tab 1: Direct Simplex Visualization

For MESS3, we plot the belief state directly on the 2-simplex (triangle). No information is lost.

### Tab 2: Marginal Projection

For the Golden Mean × Even process, the belief lives in a 3-simplex (tetrahedron)—a 3D object that's hard to visualize directly.

Instead, we project to **marginal beliefs**:

**X-axis:** $P(\text{Driver} = A \mid y_{1:t}) = b_0 + b_1$

This is the sum of probabilities for states (A,A) and (A,B), both of which have the driver in state A.

**Y-axis:** $P(\text{Transducer} = A \mid y_{1:t}) = b_0 + b_2$

This is the sum of probabilities for states (A,A) and (B,A), both of which have the transducer in state A.

### What Information Does the Projection Lose?

The marginals capture 2 degrees of freedom, but the full joint belief has 3 degrees of freedom. The missing information is the **correlation** between driver and transducer beliefs.

**Example:** These two beliefs have identical marginals but different joint structures:

| Belief | (A,A) | (A,B) | (B,A) | (B,B) | P(D=A) | P(T=A) |
|--------|-------|-------|-------|-------|--------|--------|
| $\mathbf{b}_1$ | 0.5 | 0 | 0 | 0.5 | 0.5 | 0.5 |
| $\mathbf{b}_2$ | 0 | 0.5 | 0.5 | 0 | 0.5 | 0.5 |

Both appear at point (0.5, 0.5) in the visualization, but:
- $\mathbf{b}_1$ believes driver and transducer states are **perfectly correlated** (both A or both B)
- $\mathbf{b}_2$ believes they are **perfectly anti-correlated** (one A, one B)

The joint belief bars below the plot show this full information.

### The Four Corners

The corners of the plot represent certainty about joint states:

| Point | P(D=A) | P(T=A) | Certain State |
|-------|--------|--------|---------------|
| (1, 1) | 1 | 1 | (A, A) |
| (1, 0) | 1 | 0 | (A, B) |
| (0, 1) | 0 | 1 | (B, A) |
| (0, 0) | 0 | 0 | (B, B) |

The **orange circle** indicating the true hidden state always appears at one of these corners.

---

## 8. Why Fractals Emerge

The fractal structure in the belief visualizations is not merely decorative—it reflects the **computational structure of optimal inference**.

### Iterated Function Systems (IFS)

Each observation $y$ induces a mapping on belief space:

$$f_y(\mathbf{b}) = \text{normalize}\Big( (\mathbf{b} \odot \mathbf{E}_{:,y}) \cdot \mathbf{T} \Big)$$

For MESS3 with 3 observations, we have 3 such maps: $f_0$, $f_1$, $f_2$.

Each map is a **contraction**—it pulls beliefs toward a specific region of the simplex (roughly toward the vertex corresponding to the state that likely emitted that observation).

### The Attractor

An **Iterated Function System** with contractive maps has a unique **attractor**—a set that is invariant under the maps. When we randomly apply these maps (according to the observation probabilities), the belief state traces out this attractor.

For MESS3, this attractor has a **Sierpiński-like** fractal structure.

### Why This Matters

The fractal geometry captures:
1. **Which beliefs are reachable** from a given starting point
2. **How beliefs cluster** under typical observation sequences
3. **The computational complexity** of the inference problem

Transformers trained on HMM sequences learn to **linearly encode** these belief states in their residual stream. The fractal geometry is literally embedded in the neural network's internal representations.

---

## Summary

| Aspect | MESS3 (Tab 1) | Golden Mean × Even (Tab 2) |
|--------|---------------|---------------------------|
| Hidden states | 3 | 4 (joint) |
| Observations | 3 | 2 |
| Belief dimension | 2 (triangle) | 3 (tetrahedron) |
| Visualization | Direct on simplex | Marginal projection to 2D |
| Information loss | None | Correlation structure |
| Fractal structure | Sierpiński-like | Complex, parameter-dependent |

Both visualizations demonstrate the same core concept: **Bayesian belief updating creates structured, often fractal, trajectories through probability space**.

---

## References

1. Rabiner, L. R. (1989). "A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition." *Proceedings of the IEEE*.

2. Crutchfield, J. P., & Young, K. (1989). "Inferring Statistical Complexity." *Physical Review Letters*.

3. Barnsley, M. F. (1988). *Fractals Everywhere*. Academic Press. (For IFS theory)

4. "Transformers Represent Belief State Geometry in Their Residual Stream" — The theoretical motivation for this visualization.
