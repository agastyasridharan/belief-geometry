/**
 * Belief Geometry - Configuration
 *
 * Customizable settings for the visualization.
 * Modify these values to adjust HMM parameters, colors, and behavior.
 *
 * @author Agastya Sridharan
 * @license MIT
 */

const BeliefGeometryConfig = {
    // =========================================================================
    // MESS3 Process Parameters (Tab 1)
    // =========================================================================
    mess3: {
        // Self-loop probability: probability of staying in current state
        // Higher values = more persistent states, slower mixing
        selfLoopProbability: 0.90,

        // Emission accuracy: probability that state i emits observation i
        // Higher values = observations are more informative about state
        emissionAccuracy: 0.85,

        // Simulation settings
        simulation: {
            initialSpeed: 1000,      // milliseconds per step
            minSpeed: 100,           // fastest speed
            maxSpeed: 5000,          // slowest speed
            fractalTrajectories: 100, // number of trajectories for attractor
            fractalSeqLength: 300    // length of each trajectory
        }
    },

    // =========================================================================
    // Golden Mean × Even Process Parameters (Tab 2)
    // =========================================================================
    goldenMeanEven: {
        // Driver (Golden Mean) parameters
        driver: {
            // p1: probability of emitting 1 when in state A
            p1: 0.3,
        },

        // Transducer (Even Process) parameters
        transducer: {
            // When driver emits x=1
            p1: 0.8,
            // When driver emits x=0
            p2: 0.2
        },

        // Simulation settings
        simulation: {
            initialSpeed: 1000,
            minSpeed: 100,
            maxSpeed: 5000
        }
    },

    // =========================================================================
    // Hierarchical Unifilar Process Parameters (Tab 3)
    // =========================================================================
    hierarchicalUnifilar: {
        // Driver parameters
        driver: {
            p: 0.70,  // L state: probability of emitting 0 (stay)
            q: 0.70   // R state: probability of emitting 0 (stay)
        },

        // Transducer parameters (depends on driver output x)
        transducer: {
            r1: 0.80,  // L state, driver emits 0: prob of emitting 0
            r2: 0.30,  // L state, driver emits 1: prob of emitting 0
            s1: 0.80,  // R state, driver emits 0: prob of emitting 0
            s2: 0.30   // R state, driver emits 1: prob of emitting 0
        },

        // Simulation settings
        simulation: {
            initialSpeed: 1000,
            minSpeed: 100,
            maxSpeed: 5000
        }
    },

    // =========================================================================
    // Visual Settings
    // =========================================================================
    colors: {
        // Belief state visualization
        beliefState: '#2a9d8f',
        beliefTrail: 'rgba(42, 157, 143, 0.15)',

        // True hidden state indicator
        trueState: '#d97706',
        trueStateRing: 'rgba(217, 119, 6, 0.3)',

        // Fractal attractor points
        fractal: 'rgba(184, 168, 200, 0.4)',

        // Observation colors (for 3-state systems)
        observations: ['#c25450', '#4a7c59', '#4a6fa5'],

        // State colors for belief bars
        states: ['#c25450', '#4a7c59', '#4a6fa5', '#7a6890'],

        // Canvas background
        canvasBackground: '#fffdf9',

        // Simplex triangle
        simplexStroke: '#d4c8b8',
        simplexFill: 'rgba(248, 244, 237, 0.5)',

        // Vertex labels
        vertexLabel: '#6b5d4d'
    },

    // =========================================================================
    // Canvas Settings
    // =========================================================================
    canvas: {
        // Belief dot size
        beliefDotRadius: 8,

        // True state ring
        trueStateRingRadius: 14,
        trueStateRingWidth: 3,

        // Fractal point size
        fractalPointRadius: 1.5,

        // Trail settings
        trailLength: 50,
        trailMinOpacity: 0.1,
        trailMaxOpacity: 0.6
    },

    // =========================================================================
    // Animation Settings
    // =========================================================================
    animation: {
        // Transition durations (milliseconds)
        beliefTransition: 150,
        tabTransition: 300,
        revealTransition: 400
    }
};

// Make configuration globally available
if (typeof window !== 'undefined') {
    window.BeliefGeometryConfig = BeliefGeometryConfig;
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = BeliefGeometryConfig;
}
