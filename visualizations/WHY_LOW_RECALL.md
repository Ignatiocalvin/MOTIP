# Understanding Why 7-Concept Model Has Lower Recall

## Quick Answer

The 7-concept model has **27.59% recall** (vs 68.29% for 2-concept) because of **gradient competition** in multi-task learning. When training on 7 concepts instead of 2, the detection task receives less gradient signal, leading to worse detection performance.

## The Math Behind It

### Loss Weighting
Both models use `CONCEPT_LOSS_COEF = 0.5` for each concept:

**2-Concept Model:**
```
Total Concept Supervision = 2 concepts × 0.5 = 1.0
Detection gets 50% of gradient signal
```

**7-Concept Model:**
```
Total Concept Supervision = 7 concepts × 0.5 = 3.5
Detection gets only 22% of gradient signal
```

This **2.3× reduction** in detection gradient causes:
- **+32.4% worse bounding box loss**
- **+26.6% worse GIoU loss**
- **-40.7% lower recall** (27.59% vs 68.29%)

## The Cascade Effect

```
Less detection gradient 
    ↓
Worse detection training
    ↓
Lower confidence scores
    ↓
Fewer predictions pass threshold (0.5)
    ↓
Missing 72% of people instead of 32%
    ↓
Can't track what you can't detect
    ↓
Poor tracking: 27% MOTA vs 66% MOTA
```

## Visual Evidence

See the generated visualizations:

1. **[gradient_competition_visualization.png](gradient_competition_visualization.png)**
   - Shows how concept loss dominates (190:1 ratio vs 54:1)
   - Illustrates gradient distribution (22% vs 50% for detection)
   - Displays performance degradation metrics

2. **[tracking_pipeline_visualization.png](tracking_pipeline_visualization.png)**
   - Shows detection as the bottleneck
   - Compares 100 people → 68 detected → 66 tracked (2-concept)
   - vs 100 people → 28 detected → 27 tracked (7-concept)

3. **[GRADIENT_COMPETITION_ANALYSIS.md](GRADIENT_COMPETITION_ANALYSIS.md)**
   - Detailed mathematical analysis
   - Proposed solutions
   - Complete explanation

## Key Takeaway

**This is NOT a bug—it's a fundamental trade-off in multi-task learning.**

When you add more auxiliary tasks (concepts), you must carefully balance the loss weights. The current implementation treats all 7 concepts as equally important as detection, which starves detection of gradient signal.

**Solution:** Reduce `CONCEPT_LOSS_COEF` for the 7-concept model from 0.5 to ~0.143 to equalize total concept supervision with the 2-concept model.

## Why High Precision (99.93%) But Low Recall?

The model becomes **conservative**:
- Only makes predictions when very confident
- Filters out false positives → high precision
- But also misses real objects → low recall

This is the classic precision-recall trade-off. The undertrained detector produces lower confidence scores, so fewer predictions pass the 0.5 threshold.

## Bottom Line

**7-concept model isn't broken—it's overloaded.**

The neural network has limited capacity. When you ask it to predict 7 concepts instead of 2, something has to give. In this case, detection performance degraded by 30%, which cascaded into 40% lower recall and ultimately poor tracking performance.

The 2-concept model achieves better overall performance by focusing on fewer, more important concepts while maintaining strong detection capabilities.
