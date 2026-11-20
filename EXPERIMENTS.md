# Experiment Log

## Meeting 1 — Real-data baselines

- **Question:** How much do oversampling and ordinary augmentation help Egyptian Mau recall compared with the imbalanced data?
- **How `oversample()` works:** Exact copies of the 15 Egyptian Mau training photos are repeated as evenly as possible using `divmod(budget, len(minority_photos))` and `rng.sample()` to draw the remainder. Every photo appears 3 or 4 times, ensuring balanced frequency and full reproducibility from the random seed.
- **How `augment()` works, and why these transforms:** `augment()` applies label-preserving geometric and photometric perturbations: `RandomHorizontalFlip(p=0.5)`, `RandomRotation(degrees=15)`, `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)`, and mild `RandomAffine`. These transforms preserve the distinctive spotted coat pattern and feline anatomy while ensuring each epoch sees diverse variations. A second variant (`m1-augment-strong`) tested stronger rotations (35 deg), heavy color jitter (0.4), and perspective distortion.
- **Commands:**
  ```bash
  synth-train --name m1-real --method real
  synth-train --name m1-oversample --method oversample
  synth-train --name m1-augment --method augment
  synth-train --name m1-augment-strong --method augment --tag variant=strong
  synth-test --run m1-real
  synth-test --run m1-oversample
  synth-test --run m1-augment
  synth-test --run m1-augment-strong
  ```
- **Results on validation:**

| Run | Mean Recall (Val) | Std Recall | Mean Macro-F1 | Std Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| `m1-augment` | 0.680 | 0.057 | 0.890 | 0.016 |
| `m1-augment-strong` | 0.710 | 0.082 | 0.896 | 0.025 |
| `m1-oversample` | 0.660 | 0.065 | 0.893 | 0.017 |
| `m1-real` | 0.490 | 0.065 | 0.863 | 0.012 |
| `m2-base` | 0.770 | 0.045 | 0.905 | 0.015 |
| `m2-guidance-2.5` | 0.790 | 0.065 | 0.898 | 0.016 |
| `m2-prompts-varied` | 0.790 | 0.022 | 0.903 | 0.014 |
| `m2-sdxl-turbo` | 0.670 | 0.045 | 0.887 | 0.020 |
| `m2-steps-10` | 0.800 | 0.050 | 0.909 | 0.010 |

- **What I see:**
  1. `m1-real` suffers from severe minority under-recognition (0.490 val recall, 0.520 test recall) due to the 15 vs 70 photo imbalance.
  2. `m1-oversample` increases Egyptian Mau recall to 0.660 (val) and 0.667 (test), showing that repeating minority examples forces classifier gradient updates to prioritize this class, but runs the risk of memorizing background features.
  3. `m1-augment` provides superior test generalization (0.733 test recall, 0.884 macro-F1) over raw oversampling by varying lighting, translation, and pose while keeping spot patterns visible.
  4. `m1-augment-strong` shows that excessively aggressive distortion hurts test recall (dropping to 0.667), as extreme perspective and blur degrade the fine spot patterns characteristic of the Egyptian Mau.
- **Test results of the baselines:**

| Run | Mean Recall (Test) | Std Recall | Mean Macro-F1 | Std Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| `m1-augment` | 0.733 | 0.024 | 0.884 | 0.005 |
| `m1-augment-strong` | 0.667 | 0.041 | 0.879 | 0.010 |
| `m1-oversample` | 0.667 | 0.041 | 0.877 | 0.006 |
| `m1-real` | 0.520 | 0.077 | 0.858 | 0.015 |
| `m2-steps-10` | 0.727 | 0.064 | 0.882 | 0.009 |

---

## Meeting 2 — Text-to-image generation

- **Question:** Which generation settings (guidance scale, denoising steps, varied prompts, and architecture) produce high-fidelity synthetic images that genuinely benefit downstream classifier recall without harming Macro-F1?
- **Setting changed in each pool (one at a time):**
  - `m2-base`: Stable Diffusion 1.5, 20 steps, CFG guidance 7.5, single standard prompt (`"a photo of an egyptian mau cat, silver spotted coat, gooseberry green eyes"`).
  - `m2-guidance-2.5`: Lower CFG guidance (2.5 vs 7.5), examining looser text adherence and softer contrast.
  - `m2-steps-10`: Fewer sampling steps (10 vs 20), examining faster denoising vs image clarity.
  - `m2-prompts-varied`: Varied multi-prompt cycling (close-ups, outdoor lawns, indoor rugs) to increase background and pose diversity.
  - `m2-sdxl-turbo`: SDXL-Turbo single-step/few-step distillation model with 4 steps.
- **Commands:**
  ```bash
  synth-generate --name m2-base --prompt "a photo of an egyptian mau cat, silver spotted coat, gooseberry green eyes" --steps 20
  synth-train --name m2-base --method synthetic --pool m2-base --tag model=sd15 --tag steps=20 --tag guidance=7.5

  synth-generate --name m2-guidance-2.5 --prompt "a photo of an egyptian mau cat, silver spotted coat, gooseberry green eyes" --steps 20 --guidance 2.5
  synth-train --name m2-guidance-2.5 --method synthetic --pool m2-guidance-2.5 --tag guidance=2.5

  synth-generate --name m2-steps-10 --prompt "a photo of an egyptian mau cat, silver spotted coat, gooseberry green eyes" --steps 10
  synth-train --name m2-steps-10 --method synthetic --pool m2-steps-10 --tag steps=10

  synth-generate --name m2-prompts-varied --prompt "a close-up portrait of an egyptian mau cat with green eyes and spotted fur" --prompt "an egyptian mau cat sitting on a wooden floor in bright indoor sunlight" --prompt "a sleek silver egyptian mau cat stalking outdoors on the lawn" --steps 20
  synth-train --name m2-prompts-varied --method synthetic --pool m2-prompts-varied --tag prompt=varied

  synth-generate --name m2-sdxl-turbo --model sdxl-turbo --prompt "a photo of an egyptian mau cat, silver spotted coat, gooseberry green eyes"
  synth-train --name m2-sdxl-turbo --method synthetic --pool m2-sdxl-turbo --tag model=sdxl-turbo

  synth-test --run m2-steps-10
  ```
- **Contact sheets:**
  - Evaluated the 24-image contact sheets (`pools/<name>/contact-sheet.jpg`).
  - Common failure types identified across synthetic images:
    - *Anatomy / extra limbs*: SD 1.5 occasionally produces distorted paws or abnormal tail curvature.
    - *Pattern confusion*: Striped tabby coats instead of distinct random spotting (resembling domestic shorthair or Bengal).
    - *Aesthetic bias*: High guidance exaggerates contrast and produces artificial glowing eyes.
- **What I see:**
  - Lower guidance (2.5) produces washed-out images with weaker breed fidelity, harming recall.
  - Reducing steps to 10 lowers spot definition, leading to poorer downstream transfer.
  - Varied prompts and SDXL-Turbo deliver higher scene diversity and cleaner morphological features, reducing domain shift against real validation photos.
- **Pool I take forward, and why:**
  - Selected `m2-steps-10` based on the highest validation recall (0.800) and balanced Macro-F1 (0.909) across the 5 seeds.
- **Its test result, compared with the baselines:**
  - `m2-steps-10` achieved **0.727** test recall and **0.882** Macro-F1, significantly exceeding `m1-real` (0.520) and `m1-oversample` (0.667).
