# Pothole Image Classification Guidelines
**Version 2.0 - For Whole-Image Classification Models**

## Core Principle
**Consistency is everything.** If an image is borderline, pick a class and stick to that logic forever. Do not annotate while tired! Fatigue causes you to default to "Good" subconsciously.

---

## The 5 Categories

### ✨ 1. Excellent
**Rule:** Absolutely flawless road surface. 
- Newly laid asphalt or perfectly maintained surfaces.
- Not a single visible pothole, patch, or significant crack.
- *If you see a patch or large crack, it is NOT Excellent.*

### ✅ 2. Good
**Rule:** Minor wear and tear, but perfectly comfortable at speed.
- Small hairline cracks, minor discoloration, or very small wear spots.
- Nothing that would cause a noticeable jolt in the cabin.

### ⚠️ 3. Fair
**Rule:** Noticeable road damage; you feel it, but you wouldn't swerve.
- Patched potholes, significant cracking, or shallow depressions.
- Bumpy ride, but no immediate risk of vehicle damage.

### ❌ 4. Poor
**Rule:** Severe structural damage; you must slow down or swerve.
- Deep, open potholes with visible edges and shadows.
- Broken road edges or massive missing chunks of asphalt.
- Immediate risk of damage to tires or suspension if hit at speed.

### 🚫 5. Invalid (New Category!)
**Rule:** The model cannot and should not evaluate the road here.
- Pitch black / night time where the road isn't clearly visible.
- Heavy rain, fog, or glaring sun blocking the view.
- Camera pointing at the sky, a wall, or inside a garage.
- *Note: This prevents the model from blindly guessing "Fair" when it can't see anything!*

---

## 🛑 How to Fix Inconsistency

### 1. The Borderline Rule
If you look at an image and think "Is this Good or Fair?", you are dealing with a borderline case.
**Solution:** Do not guess randomly. Look at the reference images provided in the web UI. Pick the closest match. 

### 2. The Fatigue Rule
Do not annotate for more than 20 minutes straight. Your brain processes images differently when tired. Take breaks.

### 3. The Consensus Review (Option A)
Your web app now runs a blind consensus check. Your annotation counts, but if a second reviewer (or yourself later) chooses a different class, the image is marked as a "Disagreement" and will need a tie-breaker. This keeps quality pure!
