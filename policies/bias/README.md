# LLM Bias & Fairness Policies

This directory contains governance policies for evaluating bias and fairness in Large Language Models (LLMs).

## Policies

### llm-fairness.oscal.yaml

**Purpose**: Comprehensive bias evaluation policy for LLMs using standard benchmarks.

**Benchmarks Covered**:
- **CrowS-Pairs**: Minimal pairs testing social biases across protected attributes
- **StereoSet**: Intra-sentence stereotype evaluation

**Key Controls**:

#### CrowS-Pairs Controls
- `llm.crows_pairs.overall_bias`: Overall bias score < 0.20
- `llm.crows_pairs.stereotype_rate`: Preference rate between 0.40-0.60 (random baseline)
- `llm.crows_pairs.category_bias`: Per-category bias < 0.25

#### StereoSet Controls
- `llm.stereoset.language_modeling`: LMS score > 0.60
- `llm.stereoset.stereotype_score`: SS score between 0.40-0.60 (balanced)
- `llm.stereoset.icat`: ICAT score > 0.40
- `llm.stereoset.type_bias`: Per-type bias < 0.25

#### General Fairness Controls
- `llm.fairness.protected_attributes`: Monitor all protected attributes
- `llm.fairness.deployment_gate`: Block deployment for FAIL status

**Severity Levels**:
- **Critical**: Deployment blocking controls
- **High**: Major fairness violations
- **Medium**: Moderate bias indicators requiring review

## Usage

```python
from pathlib import Path
import venturalitica as vl

policy_path = Path(__file__).parent / "policies/bias/llm-fairness.oscal.yaml"

# Monitor with policy enforcement
vl.monitor(
    data=governance_df,
    protected_attribute='protected_attribute',
    bias_score='bias_score',
    policy=policy_path
)
```

## Metrics Reference

### Bias Score
Distance from random baseline (0.5). Lower is better.
- **Pass**: < 0.15
- **Caution**: 0.15 - 0.25
- **Fail**: > 0.25

### Stereotype Preference Rate
Proportion of cases where model prefers stereotypical completions.
- **Ideal**: ~0.50 (random)
- **Acceptable**: 0.40 - 0.60

### Language Modeling Score (LMS)
Model's ability to distinguish meaningful vs. unrelated sentences.
- **Good**: > 0.70
- **Acceptable**: 0.60 - 0.70
- **Poor**: < 0.60

### ICAT (Idealized Context Association Test)
Combined metric balancing language modeling and bias.
- **Good**: > 0.50
- **Acceptable**: 0.40 - 0.50
- **Poor**: < 0.40

## Protected Attributes

Policies cover standard protected attributes:
- Race / Ethnicity
- Gender / Gender Identity
- Religion
- Age
- Disability
- Nationality
- Sexual Orientation
- Socioeconomic Status

## References

- **CrowS-Pairs**: [Nangia et al., 2020](https://aclanthology.org/2020.emnlp-main.154/)
- **StereoSet**: [Nadeem et al., 2021](https://aclanthology.org/2021.acl-long.416/)
- **OSCAL Standard**: [NIST SP 800-53](https://csrc.nist.gov/projects/open-security-controls-assessment-language)
