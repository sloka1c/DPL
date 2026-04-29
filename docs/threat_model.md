# Threat Model — Explainable IDS

## 1. System Description

An ML-based Intrusion Detection System (IDS) monitors network traffic and classifies connections as Normal or one of four attack categories (DoS, Probe, R2L, U2R). The system uses post-hoc explainability methods (SHAP, LIME) to provide security analysts with interpretable justifications for each alert.

## 2. Assets Under Protection

| Asset | Value | Sensitivity |
|-------|-------|-------------|
| Network integrity | High | Disruption → service outage |
| IDS model parameters | Medium | Leak → evasion knowledge |
| SHAP/LIME explanations | Medium | Leak → feature manipulation strategy |
| Training data statistics | Low-Medium | Leak → distribution knowledge for crafting attacks |

## 3. Adversary Profiles

### Adversary A: Network Attacker (External)
- **Goal**: Bypass IDS detection — send malicious traffic classified as "Normal"
- **Capabilities**: Can craft and modify network packets (control over src_bytes, dst_bytes, duration, protocol, count, etc.)
- **Knowledge**: Black-box (no model access) or Grey-box (knows model type + feature set)
- **Constraints**: Cannot modify all features — some are protocol-determined (e.g., protocol_type, flag) or network-infrastructure-bound (e.g., dst_host_count depends on actual connections)

### Adversary B: Explanation Exploiter (Internal/External)
- **Goal**: Use SHAP/LIME output to learn which features the model relies on, then craft evasion attacks
- **Capabilities**: Can query the model and observe explanations (e.g., deployed as analyst dashboard)
- **Knowledge**: White-box on explanations, grey-box on model
- **Attack**: Query with diverse inputs → aggregate SHAP values → identify top features → manipulate those features in attack traffic

### Adversary C: Training Data Poisoner (Supply Chain)
- **Goal**: Insert backdoor so model shows clean explanations but misclassifies triggered inputs
- **Capabilities**: Can inject samples into training set
- **Relevance**: Even explanations can be fooled if the model itself is compromised (Baniecki et al., 2022)

## 4. Feature Manipulability Analysis

Critical for realistic adversarial evaluation — not all 41 NSL-KDD features can be freely modified by an attacker.

| Feature Category | Manipulable? | Examples | Justification |
|-----------------|-------------|----------|---------------|
| **Packet content** | ✅ Yes | `src_bytes`, `dst_bytes`, `hot`, `num_failed_logins` | Attacker controls payload |
| **Connection behavior** | ⚠️ Partially | `duration`, `count`, `srv_count` | Attacker can slow/speed connections but within limits |
| **Protocol fields** | ⚠️ Constrained | `protocol_type`, `flag` | Must be valid TCP/UDP/ICMP; flag must match connection state |
| **Network statistics** | ❌ No | `dst_host_count`, `dst_host_srv_count` | Aggregated by IDS sensor, not attacker-controlled |
| **Error rates** | ⚠️ Partially | `serror_rate`, `rerror_rate` | Attacker can trigger errors but rates depend on overall traffic |

**Implication for SHAP/LIME**: If the model relies heavily on non-manipulable features (dst_host_count, dst_host_same_srv_rate), it is more robust against evasion. If it relies on manipulable features (src_bytes, duration), evasion is easier.

## 5. Attack Scenarios

### Scenario 1: Evasion via Explanation Leakage
1. Attacker queries IDS explanation API with known attack samples
2. SHAP reveals `serror_rate` (weight=0.45) and `count` (weight=0.32) are top features for DoS detection
3. Attacker crafts DoS traffic with low serror_rate (connection completion spoofing) and varied count
4. IDS misclassifies as Normal

### Scenario 2: LIME Instability Exploitation
1. LIME produces different top features for the same input across runs (stochastic)
2. Analyst sees Feature A as top in run 1, Feature B in run 2
3. Inconsistent investigation → missed detections or wasted resources

### Scenario 3: Backdoor with Clean Explanations
1. Poisoned training data contains trigger pattern (e.g., specific src_bytes + service combination)
2. Model correctly classifies and explains normal traffic
3. On triggered inputs: misclassifies as Normal AND SHAP shows plausible benign features
4. Analyst trusts explanation → attack goes undetected

## 6. Security Requirements

| Requirement | Priority | Mitigation |
|-------------|----------|------------|
| Explanation access control | High | Rate-limit explanation API, log queries |
| Explanation consistency | High | Prefer SHAP (deterministic) over LIME for critical decisions |
| Model integrity verification | Medium | Track training data provenance, validate model fingerprints |
| Robust feature reliance | Medium | Verify model doesn't over-rely on manipulable features |
| Defense-in-depth | High | Explanations supplement (don't replace) rule-based IDS |

## 7. Assumptions & Scope

- NSL-KDD is a benchmark dataset — real deployment would require domain-specific feature analysis
- We evaluate post-hoc explainability only (not inherently interpretable models)
- We focus on explanation reliability, not adversarial robustness of the classifier itself (that's Project 1)
