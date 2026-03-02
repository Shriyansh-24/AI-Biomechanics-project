# Non-Invasive Biological Movement Phenotyping System
### Quantifying Human Movement Biomarkers for Injury Risk Assessment

![Python](https://img.shields.io/badge/Python-3.12-blue)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-green)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8-orange)
![Status](https://img.shields.io/badge/Status-Active%20Research-yellow)
![Field](https://img.shields.io/badge/Field-Biotechnology%20%7C%20Biomechanics-purple)

---

What This Program Actually Does
You open your laptop, run motion_capture.py, and stand in front of your webcam. The program opens a live camera window showing your body with a skeleton drawn over it. As you perform squats, it calculates the exact angle of your knee joint in real time using 3D mathematics and displays that number on screen next to your knee. It watches for dangerous movement patterns — such as your knee collapsing inward, your two legs moving unevenly, or your form breaking down as you tire — and displays colour-coded warnings on screen the moment it detects them. Whenever the overall injury risk crosses a threshold, it silently saves an annotated photograph of that exact moment to a folder on your computer, with a plain-English caption explaining what went wrong and how to fix it. Every session also generates a CSV file of frame-by-frame joint data and a visual dashboard summarising your movement quality across the session. The system automatically detects whether you are facing the camera or standing sideways and adjusts which checks it runs accordingly — for example, it only checks for inward knee collapse when you are facing the camera, where that measurement is geometrically meaningful. No markers, no wires, no lab equipment — just a webcam and Python.

## Overview

In biotechnology, a **phenotype** is any observable, measurable characteristic of a biological organism. Human movement is a phenotype — and like all phenotypes, it encodes meaningful biological information about neuromuscular health, fatigue state, structural asymmetries, and injury susceptibility.

This system uses computer vision and biomechanical analysis to **non-invasively capture and quantify movement phenotypes** in real time, using only a standard webcam. It generates structured, repeatable biological data that can be used to investigate how movement patterns relate to injury risk — without physical markers, laboratory infrastructure, or clinical intervention.

Built as part of an NSRI School Ambassador research project exploring the intersection of **biotechnology, biomechanics, and data science**.

---

## The Biological Question

> *Can non-invasive, markerless movement analysis generate clinically meaningful biological data about injury susceptibility — and can that data reveal patterns that traditional observation cannot?*

This project is an attempt to answer that question through:

- Quantitative measurement of knee joint kinematics (3D angle data)
- Detection of neuromuscular compensation patterns (valgus, varus, asymmetry)
- Tracking fatigue-related phenotypic changes across repeated movements
- Generating structured datasets suitable for biological pattern analysis

---

## Why This Connects to Biotechnology

Traditional injury risk assessment requires either:
- Expensive laboratory motion capture
- Invasive biological testing (blood lactate, EMG electrodes)
- Subjective clinical observation

This system proposes a third approach: **phenotypic screening through computer vision**. Just as high-throughput biological screening identifies at-risk molecular phenotypes, this system identifies at-risk movement phenotypes — quickly, non-invasively, and at scale.

This has direct relevance to:

| Biotechnology Application | Connection to This Project |
|---|---|
| Rehabilitation medicine | Objective tracking of recovery phenotypes post-ACL surgery |
| Genetic risk research | Movement phenotypes correlate with connective tissue gene variants (COL5A1) |
| Clinical biomarker development | Fatigue score as a proxy for neuromuscular biological state |
| Population health screening | Scalable injury risk assessment without laboratory infrastructure |

---

## What the System Measures

### Primary Biological Measurements

**Knee Flexion Angle (3D)**
The interior angle at the knee joint calculated using the dot product of vectors from the hip and ankle. This is the primary kinematic phenotype — it quantifies joint range of motion and squat depth in three dimensions.

```
θ = arccos( v₁·v₂ / |v₁||v₂| )
```

Where v₁ = Hip→Knee vector and v₂ = Ankle→Knee vector, using x, y, z coordinates.

**Knee Valgus (Medial Deviation)**
Inward collapse of the knee relative to the hip-ankle alignment axis. A quantitative marker for ACL injury risk. The most common mechanism of non-contact ACL rupture in sport.

**Knee Varus (Lateral Deviation)**
Outward deviation of the knee beyond the hip-ankle axis. Associated with LCL stress, IT band syndrome, and long-term lateral compartment cartilage degradation.

**Left-Right Asymmetry**
Angular difference between the two knees at the same moment. A marker for neuromuscular imbalance and compensatory loading patterns — a known predictor of overuse injury.

**Fatigue-Related Phenotypic Shift**
Reduction in squat depth across successive repetitions. Biologically, this reflects neuromuscular fatigue — decreased motor unit recruitment, ATP depletion, and altered movement strategy under metabolic stress.

### Composite Output

**Injury Risk Score (0–100)**
A weighted composite of all detected risk factors per frame, providing a single quantitative summary of biological movement quality.

| Score | Biological Interpretation |
|---|---|
| 0–20 | Low risk — healthy movement phenotype |
| 20–50 | Moderate risk — compensatory patterns emerging |
| 50–100 | High risk — clinically significant deviation detected |

---

## Orientation-Aware Detection

The system automatically detects whether the subject is facing the camera (frontal plane view) or standing sideways (sagittal plane view) by measuring the horizontal distance between shoulder landmarks.

This is biologically important because different movement phenotypes are only visible from specific anatomical planes:

| Measurement | Frontal Plane (Front View) | Sagittal Plane (Side View) |
|---|---|---|
| Valgus / Varus | ✅ Meaningful | ❌ Geometrically invalid |
| Left-Right Asymmetry | ✅ Meaningful | ❌ Only one leg visible |
| Squat Depth (flexion angle) | ✅ Approximate | ✅ Most accurate |
| Fatigue Detection | ✅ Both views | ✅ Both views |

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/Shriyansh-24/AI-Biomechanics-project.git
cd AI-Biomechanics-project
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the AI model
Download `pose_landmarker_heavy.task` and place it in the project folder:
```
https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task
```

### 4. Run live capture
```bash
python motion_capture.py
```

### 5. Run post-session analysis
```bash
python analyze.py
```

---

## Output Data

Every session automatically generates:

**Structured CSV dataset** — frame-by-frame biological measurements:

| Column | Biological Meaning |
|---|---|
| `knee_angle_deg` | Primary kinematic phenotype — joint flexion angle |
| `knee_valgus` | Medial deviation marker — ACL stress indicator |
| `knee_varus` | Lateral deviation marker — LCL/IT band stress indicator |
| `asymmetry_deg` | Neuromuscular imbalance quantification |
| `fatigue_detected` | Phenotypic fatigue state — proxy for neuromuscular exhaustion |
| `risk_score_0_to_100` | Composite injury susceptibility score |

**Annotated risk screenshots** — images of every high-risk movement moment with plain-English biological explanation of what is wrong and how to correct it.

**Session analysis dashboard** — visual summary of all phenotypic measurements across the session.

---

## Research Limitations

Documenting limitations is a core part of scientific rigour:

| Limitation | Biological Impact | Notes |
|---|---|---|
| Single monocular camera | z-depth estimated, not measured | Multi-camera setup would improve 3D accuracy |
| No ground truth validation | Cannot confirm against lab-grade system | Key limitation for formal publication |
| Fixed thresholds | May not generalise across body types | Taller subjects may require threshold adjustment |
| No direct biological markers | Cannot measure lactate, EMG, or hormonal fatigue markers | Future work: correlate with wearable biosensors |
| 2D valgus proxy | True valgus requires 3D measurement | Single camera provides frontal plane approximation only |

---

## Future Directions

- [ ] Correlation with heart rate / HRV as biological fatigue proxy
- [ ] Hip flexion angle — forward trunk lean as a spinal load marker
- [ ] Ankle dorsiflexion — mobility phenotype linked to injury risk
- [ ] Longitudinal tracking — phenotypic changes across weeks of training
- [ ] Multi-subject dataset — population-level movement phenotype analysis
- [ ] Integration with wearable biosensors for multi-modal biological data

---

## Project Structure

```
├── motion_capture.py          # Live capture, phenotype detection, risk scoring
├── analyze.py                 # Post-session dashboard and statistical summary
├── requirements.txt           # Python dependencies
├── docs/
│   ├── TESTING_LOG.md         # Documented bugs, root causes, and fixes
│   └── RESEARCH_NOTES.md      # Biological context and scientific references
└── sample_data/
    └── sample_output.csv      # Example session dataset
```

---

## Interdisciplinary Framework

This project sits at the intersection of four disciplines:

```
        BIOLOGY
    (what to measure —
    injury mechanisms,          MATHEMATICS
    phenotype definition)    (how to measure —
            \                dot product, vector
             \               calculus, statistics)
              \             /
               \           /
            THIS PROJECT
               /           \
              /             \
    ARTIFICIAL INTELLIGENCE   COMPUTER SCIENCE
    (how to see —             (how to pipeline —
    BlazePose neural          real-time processing,
    network, landmark         data export, state
    detection)                management)
```

---

## Scientific References

- Hewett, T.E. et al. (2005). Biomechanical measures of neuromuscular control and valgus loading predict ACL injury risk. *American Journal of Sports Medicine*, 33(4), 492–501.
- Powers, C.M. (2010). The influence of abnormal hip mechanics on knee injury. *Journal of Orthopaedic & Sports Physical Therapy*, 40(2), 42–51.
- Bazrgari, B. et al. — MediaPipe BlazePose: https://arxiv.org/abs/2006.10204
- September, A.V. et al. (2009). Variants within the COL5A1 gene and ACL ruptures. *British Journal of Sports Medicine*, 43(15), 1222–1228.

---

## Author

**Shriyansh** — Grade 12 PCB, NSRI School Ambassador
*Research focus: Non-invasive biological monitoring at the intersection of computer vision and phenotypic data science*

---

## License

MIT License — free to use and build upon with attribution.
