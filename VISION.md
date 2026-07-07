# vision.md

## Project Title: Project Cora-Infans (Infant-Inspired Auditory & Cognitive Architecture)

## 1. Executive Summary & Core Objective
The objective of this project is to build an audio-centric AI architecture that replicates the biological paradigm of infant language development and hierarchical cognitive abstraction. Traditional machine learning models process uniform frames of text or audio symmetrically using unweighted, global backpropagation, demanding massive computational footprints. In contrast, **Project Infans** implements a biologically realistic, low-latency, and highly energy-efficient alternative. 

By coupling a **4-microphone hardware array** with an **Acoustic Saliency Reward Loop**, a **Predictive Coding Hierarchy**, and a **Stateful Event Workspace**, this system will simulate how an infant dynamically shifts from a "citizen of the world" to an agent that learns to isolate, track, understand, and ultimately simulate the motor plans required to "speak" native speech primitives.

---

## 2. Biological Foundations & Cognitive Philosophy

The architecture directly mimics the neural interplay that governs early human development:

* **Sensory Gating & Parentese (0–6 Months):** Subcortical structures (like the inferior colliculus and locus coeruleus) act as hardcoded deterministic filters that trigger neurochemical bursts (acetylcholine/norepinephrine) only when crossing specific physical thresholds of infant-directed speech ("parentese"). This anchors the prototypical formant boundaries ($F_1/F_2$) in the primary auditory cortex ($A1$).
* **Predictive Coding Hierarchy:** The network replaces global backpropagation with local prediction error minimization. Higher-level layers continuously generate top-down predictions. Lower layers compute the residual error and pass it upward, shifting the network's energy landscape into stable "attractor states" (semantic concepts).
* **Neural Oscillations & Global Workspace:** To solve the capacity constraints of a real-time working memory without data cross-talk, the runtime utilizes **Theta-Gamma Multiplexing**. Slow Theta cycles ($4\text{--}8\text{ Hz}$) define volatile workspace windows, while nested fast Gamma spikes ($30\text{--}80\text{ Hz}$) dynamically bind active target nodes in phase-locked harmony.
* **Analysis by Synthesis:** Speech is not treated as a passive listening task. Following the biological DIVA framework, the system bridges perception and production by mapping acoustic formant targets directly back into internal motor coordinate trajectories.

---

## 3. Hardware Implementation: ReSpeaker XMOS XVF3800

To bypass high-computation wave phase analysis in software, low-level Digital Signal Processing (DSP) is offloaded entirely to the **Seeed Studio ReSpeaker XMOS XVF3800** development board.