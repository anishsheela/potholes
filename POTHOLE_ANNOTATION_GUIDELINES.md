# Pothole Annotation Guidelines
**Version 1.0 - For Consistent Object Detection Training**

## Core Principle
**If you wouldn't slow down or swerve to avoid it, don't label it.**

---

## What to Label as "Pothole"

### ✅ ALWAYS Label These:

1. **Clear Depressions/Holes**
   - Visible depth (shadow inside)
   - Would cause vehicle jolt
   - Size: Larger than 10cm diameter
   - Example: Classic circular/oval holes in asphalt

2. **Significant Road Damage**
   - Missing chunks of asphalt
   - Exposed underlying material
   - Would damage vehicle if hit at speed
   - Deep cracks with visible depth

3. **Water-Filled Depressions**
   - Puddles that indicate underlying hole
   - Only if you can see the hole edges
   - Not just surface water on flat road

### ❌ NEVER Label These:

1. **Surface Features (Not Potholes)**
   - Paint/lane markings
   - Manhole covers (even if slightly sunken)
   - Speed bumps / road humps
   - Shadows (no actual hole)
   - Road stains/discoloration
   - Smooth patches/repairs

2. **Minor Road Imperfections**
   - Hairline cracks (no depth)
   - Small chips < 10cm
   - Rough texture but no hole
   - Worn paint

3. **Unclear/Ambiguous Cases**
   - **When in doubt, don't label it**
   - If you can't tell from the image, skip it
   - Better to miss a few than add noise

---

## How to Draw Bounding Boxes

### Rule 1: Tight Boxes
```
❌ WRONG (too loose):
   ┌─────────────────┐
   │                 │
   │      ●●●        │  ← Extra space
   │                 │
   └─────────────────┘

✅ CORRECT (tight):
   ┌────────┐
   │  ●●●   │  ← Minimal padding
   └────────┘
```

**How to do it:**
- Box edges should be ~5-10 pixels from pothole edge
- Include the entire damaged area
- Don't include extra road surface

### Rule 2: Multiple Potholes in Close Proximity

**If potholes are SEPARATE (>30cm apart):**
```
  ┌───┐      ┌───┐
  │ ● │      │ ● │  ← Two boxes
  └───┘      └───┘
```

**If potholes are CLUSTERED (touching or <30cm apart):**
```
  ┌─────────┐
  │ ●  ● ●  │  ← One box around cluster
  └─────────┘
```

**Why?** Object detection works better with discrete objects. Clusters are treated as single damaged area.

### Rule 3: Partially Visible Potholes

**If >50% visible:**
```
┌──────┐│
│ ●●●  ││  ← Label it
└──────┘│
  Road edge
```

**If <50% visible:**
```
     ┌─┐│
     │●││  ← Skip it
     └─┘│
  Road edge
```

**Why?** Partial potholes add noise. Model needs to see most of the object to learn.

### Rule 4: Fuzzy/Unclear Boundaries

**For irregular potholes:**
1. Identify the darkest/deepest part (core)
2. Look for color/texture change from good road
3. Draw box around the damaged area
4. When edges are unclear, err on the side of **smaller boxes**

```
Good road: ░░░░░░░
Damaged:   ░░███░░  ← Dark area = pothole
Good road: ░░░░░░░

Box placement:
           ┌───┐
        ░░░│███│░░
           └───┘
```

---

## Edge Cases & FAQ

### Q: Long crack vs pothole?
**A:** If it has **depth** (visible shadow) and is **wider than 5cm**, label as pothole. Hairline cracks = skip.

### Q: Patched/repaired area that's still bumpy?
**A:** Skip it. We're detecting potholes, not repairs.

### Q: Dark spot - shadow or hole?
**A:** Look for:
- Irregular edges (hole) vs straight edges (shadow from object)
- Texture change (hole) vs same texture (shadow)
- When unsure → **skip it**

### Q: Wet road with puddle - is there a hole?
**A:** Only label if you can see the hole edges. Random puddles = skip.

### Q: Multiple small potholes in a line (like 5-10 of them)?
**A:** 
- If each is clearly separate: Individual boxes
- If they form a continuous damaged strip: One long box
- If unclear: One box around the whole cluster

### Q: Pothole at edge of frame, mostly cut off?
**A:** Skip if <50% visible. Include if >50% visible.

---

## Quality Control Checklist

Before saving your annotations, ask:

- [ ] Would I swerve/slow down for this in real life?
- [ ] Is the box tight (minimal extra space)?
- [ ] Is it clearly a hole/depression (not just discoloration)?
- [ ] Am I being consistent with previous annotations?
- [ ] When in doubt, did I skip it?

---

## Examples with Images

### Example 1: Clear Pothole ✅
```
Description: Circular depression, visible depth, size ~20cm
Action: Label with tight box
Confidence: High
```

### Example 2: Road Stain ❌
```
Description: Dark patch on road, no depth, smooth surface
Action: Skip - not a pothole
Reason: Just discoloration
```

### Example 3: Cluster of Small Potholes ✅
```
Description: 3-4 small holes within 20cm of each other
Action: One box around the entire cluster
Confidence: Medium-High
```

### Example 4: Hairline Crack ❌
```
Description: Thin crack, no visible depth
Action: Skip
Reason: Too minor, not a pothole
```

### Example 5: Unclear Shadow ❌
```
Description: Dark area, can't tell if hole or shadow
Action: Skip when in doubt
Reason: Ambiguous = noise for model
```

---

## Training Data Quality Tips

### DO:
✅ Take breaks every 30 minutes (fatigue → inconsistency)
✅ Annotate similar lighting conditions together
✅ Review your work periodically
✅ Keep a "skip" count - aim for 60-70% skip rate (most frames have no potholes)
✅ When tired, stop and resume later

### DON'T:
❌ Rush through annotations
❌ Change your criteria mid-session
❌ Label when unsure
❌ Try to find potholes where there aren't any
❌ Make boxes too large "just in case"

---

## Minimum Size Guidelines

**Size threshold: 10cm diameter minimum**

Why? Smaller defects:
- Hard to see in dashcam footage
- Not safety-critical
- Add noise to training
- Inconsistent to annotate

**How to estimate 10cm:**
- Compare to lane width (~3.5m)
- Compare to visible road features
- When unsure, use the "would I swerve?" test

---

## Regional Considerations (Kerala Roads)

**You mentioned roads are relatively good - this is NORMAL!**

Expected annotation rates:
- Urban roads: 5-15% of frames have potholes
- Highway: 2-5% of frames have potholes
- Rural roads: 10-25% of frames have potholes

**It's OK to have mostly empty images!** The model needs to learn:
1. What IS a pothole (positive examples)
2. What IS NOT a pothole (negative examples = clean roads)

Clean road images are just as important as pothole images!

---

## Annotation Workflow

### Step 1: Quick Scan (5 seconds per image)
- Does this frame clearly show a pothole?
- **YES** → Proceed to Step 2
- **NO or UNSURE** → Mark as background, move on

### Step 2: Careful Labeling (20-30 seconds per image)
- Draw tight boxes around clear potholes
- Apply all rules above
- Double-check box placement

### Step 3: Quality Check (every 50 images)
- Review last 10 labeled images
- Ensure consistency
- Adjust if you notice drift in criteria

---

## Common Mistakes to Avoid

| Mistake | Why It's Bad | How to Fix |
|---------|-------------|------------|
| Boxing shadows | Model learns wrong features | Skip ambiguous dark areas |
| Huge loose boxes | Model can't learn precise location | Keep boxes tight (5-10px padding) |
| Labeling tiny chips | Adds noise, inconsistent | 10cm minimum size rule |
| Changing criteria mid-session | Inconsistent training data | Stick to these guidelines |
| Boxing manhole covers | Wrong object class | Only holes in road surface |
| Including repairs | Not a pothole | Only active damage |

---

## Re-Annotation Strategy

If you've already labeled data inconsistently:

### Option 1: Fresh Start (Recommended if <500 images)
- Start over with these clear guidelines
- 500 high-quality images > 2000 inconsistent images

### Option 2: Audit & Fix (If >500 images)
1. Sample 50 random images from your dataset
2. Re-annotate them following these guidelines
3. Compare with original annotations
4. If >30% different → start over
5. If <30% different → fix obvious mistakes

### Option 3: Two-Pass Approach
1. **Pass 1 (Quick):** Flag images with potholes vs without (binary classification)
2. **Pass 2 (Careful):** Only annotate flagged images with boxes

---

## Tools & Shortcuts

### Keyboard Shortcuts (Label Studio / CVAT)
- **Skip ambiguous images quickly** - don't waste time on unclear cases
- **Copy previous box size** - if labeling similar potholes
- **Zoom in** - for precise box placement

### Efficiency Tips
- Annotate chronologically (similar lighting/roads together)
- Batch similar road types
- Set realistic goals: 50-100 quality annotations per hour

---

## Success Metrics

You're doing well if:
- ✅ 60-70% of frames have NO potholes (background)
- ✅ Can explain why you labeled each pothole
- ✅ Boxes are consistent in size/placement
- ✅ You skip ambiguous cases without hesitation
- ✅ Model validation recall improves with more data

You need to revise if:
- ❌ >40% of frames have potholes (probably over-labeling)
- ❌ Validation metrics get worse with more data
- ❌ You're labeling things you wouldn't swerve for
- ❌ Boxes are very inconsistent in size

---

## Final Checklist Before Training

- [ ] Read these guidelines fully
- [ ] Understand "when in doubt, skip it" principle
- [ ] Applied 10cm minimum size rule
- [ ] Kept boxes tight (5-10px padding)
- [ ] Grouped clustered potholes
- [ ] Skipped all ambiguous cases
- [ ] ~60-70% of images are backgrounds (no labels)
- [ ] Reviewed sample of own work for consistency

**Remember: Quality >> Quantity**  
500 perfect annotations >> 2000 inconsistent annotations

---

## Questions or Unclear Cases?

When you encounter a difficult case:
1. Take a screenshot
2. Note why it's unclear
3. Create a reference library
4. Make a consistent decision for similar future cases

**Building your personal reference library of edge cases helps maintain consistency!**
