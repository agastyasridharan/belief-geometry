# Belief Geometry

An interactive visualization of belief state dynamics for Hidden Markov Models, demonstrating how Bayesian inference creates fractal structures in belief space.

**Author:** Agastya Sridharan
**License:** MIT

## Overview

This visualization explores belief state dynamics for various HMM structures:

1. **MESS3 Process** — A 3-state symmetric HMM with fractal belief attractor
2. **Golden Mean × Even Process** — A hierarchical process where driver output modulates transducer
3. **Hierarchical Unifilar Process** — A 4-state process with Krohn-Rhodes decomposition Z/2Z ≀ Z/2Z

## Getting Started

Simply open `index.html` in a modern web browser. No server required—the visualization is completely self-contained (uses CDN for KaTeX math rendering).

### Controls

- **Run/Stop** — Start or stop automatic stepping through observations
- **Step** — Advance one observation manually
- **Reset** — Return to initial state
- **Speed slider** — Adjust animation speed (100ms to 5000ms per step)
- **Parameter sliders** — Modify HMM parameters (triggers reset)

## Mathematical Background

### Belief State

The belief state **b**(t) is the posterior distribution over hidden states given observations:

```
b(t) = [P(S=0 | y₁...yₜ), P(S=1 | y₁...yₜ), ...]
```

### Bayesian Update

**Step 1: Bayes' Rule** (observation update)
```
P(Sₜ=i | yₜ) ∝ P(yₜ | Sₜ=i) · P(Sₜ=i)
```

**Step 2: Chapman-Kolmogorov** (transition propagation)
```
b(t+1)[j] = Σᵢ P(Sₜ=i | y₁:ₜ) · T[i,j]
```

### Why Fractals Emerge

For 3-state HMMs, beliefs live in a 2-simplex (triangle). The belief update applies a sequence of contractive affine transformations. Under certain conditions (as in MESS3), this creates an Iterated Function System (IFS) that generates fractal attractors resembling the Sierpiński triangle.

## Project Structure

```
belief-geometry/
├── index.html          # Main visualization (self-contained)
├── css/
│   └── styles.css      # Extracted stylesheet
├── js/
│   ├── config.js       # Customizable parameters
│   └── hmm.js          # HMM algorithms
├── docs/
│   └── EXPLAINER.md    # Mathematical details
├── README.md
└── LICENSE
```

## Customization

Edit `js/config.js` to adjust:

- **HMM Parameters** — Self-loop probabilities, emission accuracies, driver/transducer parameters
- **Colors** — Belief state, true state indicator, fractal points, observations
- **Animation** — Speeds, transition durations, trail lengths
- **Canvas** — Dot sizes, ring widths, margins

Example:
```javascript
BeliefGeometryConfig.mess3.selfLoopProbability = 0.85;
BeliefGeometryConfig.colors.beliefState = '#e63946';
```

## Processes

### Tab 1: MESS3

A 3-state symmetric HMM where each state has high self-loop probability and preferentially emits a corresponding observation.

**Transition Matrix:**
```
T = [[1-2x,  x,    x   ]
     [x,     1-2x, x   ]
     [x,     x,    1-2x]]

where x = (1-p)/2
```

### Tab 2: Golden Mean × Even

A hierarchical (driver-transducer) process:
- **Driver:** Golden Mean process (no consecutive 1s)
- **Transducer:** Even process modulated by driver output

The driver OUTPUT (not state) selects which Even process variant is active.

### Tab 3: Hierarchical Unifilar

A 4-state unifilar process with Krohn-Rhodes decomposition Z/2Z ≀ Z/2Z:
- **Driver:** Z/2Z (2 states: L, R)
- **Transducer:** Z/2Z (2 states: L, R), driven by driver output

Both components are unifilar: the next state is uniquely determined by (current state, output).

## Technical Details

- **HTML5 Canvas** for drawing belief simplex and trajectories
- **KaTeX** for LaTeX math rendering (loaded via CDN)
- **Pure JavaScript** with no build step required
- **EB Garamond** font for elegant typography

## References

1. Computational Mechanics — The theoretical framework connecting HMMs, belief states, and optimal prediction
2. Krohn-Rhodes Theorem — Decomposition of finite automata into simple components
3. Mixed-State Presentation — Mathematical foundations for belief state geometry
4. Iterated Function Systems — Theory behind fractal attractors in belief space

## License

MIT License. See [LICENSE](LICENSE) for details.
