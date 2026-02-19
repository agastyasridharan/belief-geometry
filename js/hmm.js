/**
 * Belief Geometry - Hidden Markov Model Implementation
 *
 * Core HMM algorithms for belief state computation and sampling.
 * Based on computational mechanics and Bayesian inference.
 *
 * @author Agastya Sridharan
 * @license MIT
 */

const HMM = (function() {
    'use strict';

    // =========================================================================
    // Matrix Constructors
    // =========================================================================

    /**
     * Constructs MESS3 (Messy-3) process matrices.
     *
     * The MESS3 process is a 3-state symmetric HMM where:
     * - Each state has high self-loop probability
     * - Each state preferentially emits a corresponding observation
     *
     * @param {number} x - Switching probability (1-p)/2 where p is self-loop prob
     * @param {number} alpha - Emission accuracy (prob state i emits obs i)
     * @returns {Object} { transitionMatrix, emissionMatrix }
     */
    function mess3Matrices(x = 0.05, alpha = 0.85) {
        const transitionMatrix = [
            [1 - 2 * x, x, x],
            [x, 1 - 2 * x, x],
            [x, x, 1 - 2 * x]
        ];

        const emissionMatrix = [
            [alpha, (1 - alpha) / 2, (1 - alpha) / 2],
            [(1 - alpha) / 2, alpha, (1 - alpha) / 2],
            [(1 - alpha) / 2, (1 - alpha) / 2, alpha]
        ];

        return { transitionMatrix, emissionMatrix };
    }

    /**
     * Constructs Golden Mean × Even process matrices.
     *
     * A hierarchical (driver-transducer) process where:
     * - Driver: Golden Mean process (no consecutive 1s)
     * - Transducer: Even process modulated by driver output
     *
     * Joint state space: {(A,A), (A,B), (B,A), (B,B)}
     *
     * @param {number} p1 - Driver probability of emitting 1 in state A
     * @param {number} q1 - Transducer parameter when driver emits 1
     * @param {number} q2 - Transducer parameter when driver emits 0
     * @returns {Object} { transitionEmissionTensor }
     */
    function goldenMeanEvenMatrices(p1 = 0.3, q1 = 0.8, q2 = 0.2) {
        // T[y][current_state][next_state] = P(y, next | current)
        // States: 0=(A,A), 1=(A,B), 2=(B,A), 3=(B,B)
        const T = [
            // y = 0
            [
                [(1 - p1) * (1 - q2), 0, p1 * (1 - q1), 0],     // from (A,A)
                [0, (1 - p1) * (1 - q2), 0, p1 * (1 - q1)],     // from (A,B)
                [(1 - q2), 0, 0, 0],                             // from (B,A)
                [0, (1 - q2), 0, 0]                              // from (B,B)
            ],
            // y = 1
            [
                [0, (1 - p1) * q2, 0, p1 * q1],                 // from (A,A)
                [(1 - p1) * q2, 0, p1 * q1, 0],                 // from (A,B)
                [0, q2, 0, 0],                                   // from (B,A)
                [q2, 0, 0, 0]                                    // from (B,B)
            ]
        ];

        return { transitionEmissionTensor: T };
    }

    /**
     * Constructs Hierarchical Unifilar process matrices.
     *
     * A 4-state unifilar process with Krohn-Rhodes decomposition Z/2Z ≀ Z/2Z.
     * Both driver and transducer are unifilar (output determines next state).
     *
     * @param {number} p - Driver L state: prob of emitting 0
     * @param {number} q - Driver R state: prob of emitting 0
     * @param {number} r1 - Transducer L, driver=0: prob of emitting 0
     * @param {number} r2 - Transducer L, driver=1: prob of emitting 0
     * @param {number} s1 - Transducer R, driver=0: prob of emitting 0
     * @param {number} s2 - Transducer R, driver=1: prob of emitting 0
     * @returns {Object} { fullObsMatrix, partialObsMatrix }
     */
    function hierarchicalUnifilarMatrices(p = 0.7, q = 0.7, r1 = 0.8, r2 = 0.3, s1 = 0.8, s2 = 0.3) {
        // States: 0=(L,L), 1=(L,R), 2=(R,L), 3=(R,R)
        // Full observations: A, B, C, D (4 symbols)
        // Partial observations: 0, 1 (2 symbols, A,C→0 and B,D→1)

        // T[emission][current][next] for full observation
        const fullObsMatrix = [
            // A = (x=0, y=0): both stay
            [
                [p * r1, 0, 0, 0],           // from (L,L)
                [0, p * s1, 0, 0],           // from (L,R)
                [0, 0, q * r1, 0],           // from (R,L)
                [0, 0, 0, q * s1]            // from (R,R)
            ],
            // B = (x=0, y=1): driver stays, trans switches
            [
                [0, p * (1 - r1), 0, 0],     // from (L,L)
                [p * (1 - s1), 0, 0, 0],     // from (L,R)
                [0, 0, 0, q * (1 - r1)],     // from (R,L)
                [0, 0, q * (1 - s1), 0]      // from (R,R)
            ],
            // C = (x=1, y=0): driver switches, trans stays
            [
                [0, 0, (1 - p) * r2, 0],     // from (L,L)
                [0, 0, 0, (1 - p) * s2],     // from (L,R)
                [(1 - q) * r2, 0, 0, 0],     // from (R,L)
                [0, (1 - q) * s2, 0, 0]      // from (R,R)
            ],
            // D = (x=1, y=1): both switch
            [
                [0, 0, 0, (1 - p) * (1 - r2)],   // from (L,L)
                [0, 0, (1 - p) * (1 - s2), 0],   // from (L,R)
                [0, (1 - q) * (1 - r2), 0, 0],   // from (R,L)
                [(1 - q) * (1 - s2), 0, 0, 0]    // from (R,R)
            ]
        ];

        // Partial observation: collapse A,C → 0 and B,D → 1
        const partialObsMatrix = [
            // y = 0 (from A and C)
            fullObsMatrix[0].map((row, i) =>
                row.map((val, j) => val + fullObsMatrix[2][i][j])
            ),
            // y = 1 (from B and D)
            fullObsMatrix[1].map((row, i) =>
                row.map((val, j) => val + fullObsMatrix[3][i][j])
            )
        ];

        return { fullObsMatrix, partialObsMatrix };
    }

    // =========================================================================
    // Probability Utilities
    // =========================================================================

    /**
     * Returns uniform stationary distribution.
     *
     * @param {number} n - Number of states
     * @returns {number[]} Uniform distribution
     */
    function getStationaryDistribution(n = 3) {
        return Array(n).fill(1 / n);
    }

    /**
     * Samples from a categorical distribution.
     *
     * @param {number[]} probs - Probability vector (must sum to 1)
     * @returns {number} Sampled index
     */
    function sampleCategorical(probs) {
        const u = Math.random();
        let cumsum = 0;
        for (let i = 0; i < probs.length; i++) {
            cumsum += probs[i];
            if (u <= cumsum) return i;
        }
        return probs.length - 1;
    }

    /**
     * Normalizes a vector to sum to 1.
     *
     * @param {number[]} vec - Input vector
     * @returns {number[]} Normalized probability vector
     */
    function normalize(vec) {
        const sum = vec.reduce((a, b) => a + b, 0);
        if (sum === 0) return vec.map(() => 1 / vec.length);
        return vec.map(v => v / sum);
    }

    // =========================================================================
    // Belief State Updates
    // =========================================================================

    /**
     * Updates belief state using Bayes' rule given an observation.
     *
     * b_new[i] ∝ b[i] * P(obs | state=i)
     *
     * @param {number[]} prior - Current belief state
     * @param {number} observation - Observed emission
     * @param {number[][]} emissionMatrix - E[state][obs] = P(obs | state)
     * @returns {number[]} Updated belief state
     */
    function updatePosterior(prior, observation, emissionMatrix) {
        const likelihood = emissionMatrix.map(row => row[observation]);
        const unnormalized = prior.map((p, i) => p * likelihood[i]);
        return normalize(unnormalized);
    }

    /**
     * Propagates belief state through transition matrix.
     *
     * b_new[j] = Σ_i b[i] * T[i][j]
     *
     * @param {number[]} belief - Current belief state
     * @param {number[][]} transitionMatrix - T[i][j] = P(next=j | current=i)
     * @returns {number[]} Propagated belief state
     */
    function propagateValues(belief, transitionMatrix) {
        const n = belief.length;
        const result = Array(n).fill(0);

        for (let j = 0; j < n; j++) {
            for (let i = 0; i < n; i++) {
                result[j] += belief[i] * transitionMatrix[i][j];
            }
        }

        return normalize(result);
    }

    /**
     * Combined belief update for transition-emission tensors.
     *
     * Used for processes where transition and emission are coupled.
     * b_new[j] ∝ Σ_i b[i] * T[obs][i][j]
     *
     * @param {number[]} belief - Current belief state
     * @param {number} observation - Observed emission
     * @param {number[][][]} tensor - T[obs][current][next]
     * @returns {number[]} Updated belief state
     */
    function updateBeliefWithTensor(belief, observation, tensor) {
        const n = belief.length;
        const obsMatrix = tensor[observation];
        const result = Array(n).fill(0);

        for (let j = 0; j < n; j++) {
            for (let i = 0; i < n; i++) {
                result[j] += belief[i] * obsMatrix[i][j];
            }
        }

        return normalize(result);
    }

    // =========================================================================
    // Coordinate Transformations
    // =========================================================================

    /**
     * Converts 3-state belief to 2D Cartesian coordinates.
     *
     * Maps belief simplex to equilateral triangle for visualization.
     *
     * @param {number[]} belief - 3-element belief vector
     * @param {number} size - Canvas size in pixels
     * @returns {Object} { x, y } canvas coordinates
     */
    function beliefToCartesian3(belief, size) {
        const sqrt3 = Math.sqrt(3);

        // Vertices of equilateral triangle (top, bottom-left, bottom-right)
        const vertices = [
            { x: 0, y: sqrt3 / 2 },
            { x: -0.5, y: 0 },
            { x: 0.5, y: 0 }
        ];

        // Barycentric to Cartesian
        const x = belief[0] * vertices[0].x +
                  belief[1] * vertices[1].x +
                  belief[2] * vertices[2].x;
        const y = belief[0] * vertices[0].y +
                  belief[1] * vertices[1].y +
                  belief[2] * vertices[2].y;

        // Scale and translate to canvas
        return {
            x: size / 2 + x * size * 0.75,
            y: size * 0.82 - y * size * 0.75
        };
    }

    /**
     * Converts 4-state belief to 2D Cartesian coordinates.
     *
     * Uses marginal probabilities for projection:
     * - x-axis: P(transducer state)
     * - y-axis: P(driver state)
     *
     * @param {number[]} belief - 4-element belief vector
     * @param {number} size - Canvas size in pixels
     * @returns {Object} { x, y } canvas coordinates
     */
    function beliefToCartesian4(belief, size) {
        const margin = 60;
        const innerSize = size - 2 * margin;

        // States: 0=(L,L), 1=(L,R), 2=(R,L), 3=(R,R)
        // x = P(transducer=R) = b[1] + b[3]
        // y = P(driver=R) = b[2] + b[3]
        const x = belief[1] + belief[3];
        const y = belief[2] + belief[3];

        return {
            x: margin + x * innerSize,
            y: margin + (1 - y) * innerSize  // Flip y for canvas coords
        };
    }

    // =========================================================================
    // Fractal Computation
    // =========================================================================

    /**
     * Computes fractal attractor points via random sampling.
     *
     * Simulates many trajectories to reveal the belief attractor structure.
     *
     * @param {number[][]} transitionMatrix
     * @param {number[][]} emissionMatrix
     * @param {number} numTrajectories - Number of independent trajectories
     * @param {number} seqLength - Length of each trajectory
     * @returns {number[][]} Array of belief states [b0, b1, b2]
     */
    function computeFractalPoints(transitionMatrix, emissionMatrix, numTrajectories = 100, seqLength = 300) {
        const points = [];
        const n = transitionMatrix.length;

        for (let t = 0; t < numTrajectories; t++) {
            let state = sampleCategorical(getStationaryDistribution(n));
            let belief = getStationaryDistribution(n);

            for (let i = 0; i < seqLength; i++) {
                // Generate observation from current state
                const obs = sampleCategorical(emissionMatrix[state]);

                // Update belief
                belief = updatePosterior(belief, obs, emissionMatrix);

                // Store point (skip burn-in period)
                if (i > 50) {
                    points.push([...belief]);
                }

                // Propagate belief
                belief = propagateValues(belief, transitionMatrix);

                // Transition true state
                state = sampleCategorical(transitionMatrix[state]);
            }
        }

        return points;
    }

    // =========================================================================
    // Public API
    // =========================================================================

    return {
        // Matrix constructors
        mess3Matrices,
        goldenMeanEvenMatrices,
        hierarchicalUnifilarMatrices,

        // Probability utilities
        getStationaryDistribution,
        sampleCategorical,
        normalize,

        // Belief updates
        updatePosterior,
        propagateValues,
        updateBeliefWithTensor,

        // Coordinate transforms
        beliefToCartesian3,
        beliefToCartesian4,

        // Fractal computation
        computeFractalPoints
    };
})();

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = HMM;
}
