#!/usr/bin/env python3
# ============================================================
# Biomechanics Insight Generator
# Reads your injury_prevention_data.csv and produces:
#   1. A printed report in the terminal
#   2. A visual dashboard saved as a PNG image
# ============================================================
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')
import glob

# Search for any CSV that could be a session file
csv_files = (
    sorted(glob.glob("session_*_data.csv")) +   # new timestamped format
    sorted(glob.glob("injury_prevention_data.csv"))  # old format
)

if not csv_files:
    # Last resort — find ANY csv in the folder
    csv_files = sorted(glob.glob("*.csv"))

if not csv_files:
    print("ERROR: No CSV data found in this folder.")
    print(f"Looking in: {os.path.abspath('.')}")
    print("Make sure you run analyze.py from the same folder as motion_capture.py")
    exit()

CSV_FILE   = csv_files[-1]
print(f"[LOADING] Using most recent session: {CSV_FILE}")

# Save dashboard image with same name as the CSV but as a .png
OUTPUT_IMG = CSV_FILE.replace(".csv", "_dashboard.png")
# ============================================================
# SECTION 1: LOAD AND CLEAN DATA
# ============================================================

def load_data(filepath):
    df = pd.read_csv(filepath)

    # Convert columns to correct types
    df['knee_angle_deg']       = pd.to_numeric(df['knee_angle_deg'],       errors='coerce')
    df['risk_score_0_to_100']  = pd.to_numeric(df['risk_score_0_to_100'],  errors='coerce')
    df['asymmetry_deg']        = pd.to_numeric(df['asymmetry_deg'],        errors='coerce')
    df['timestamp_s']          = pd.to_numeric(df['timestamp_s'],          errors='coerce')
    df['rep_number']           = pd.to_numeric(df['rep_number'],           errors='coerce')

    # Convert TRUE/FALSE strings to booleans
    df['knee_valgus']       = df['knee_valgus'].astype(str).str.upper() == 'TRUE'
    df['fatigue_detected']  = df['fatigue_detected'].astype(str).str.upper() == 'TRUE'

    # Split into right and left
    right = df[df['side'].str.contains('RIGHT')].copy()
    left  = df[df['side'].str.contains('LEFT')].copy()

    return df, right, left


# ============================================================
# SECTION 2: COMPUTE INSIGHTS
# ============================================================

def compute_insights(df, right, left):
    insights = {}

    # --- Basic session info ---
    insights['total_frames']    = len(df)
    insights['session_duration'] = round(df['timestamp_s'].max() - df['timestamp_s'].min(), 1)
    insights['total_reps']      = int(df['rep_number'].max()) if not df['rep_number'].isna().all() else 0

    # --- Knee angle stats ---
    for side, data, key in [("Right", right, "right"), ("Left", left, "left")]:
        angles = data['knee_angle_deg'].dropna()
        if len(angles) > 0:
            insights[f'{key}_min_angle']  = round(angles.min(), 1)   # deepest squat
            insights[f'{key}_max_angle']  = round(angles.max(), 1)   # most extended
            insights[f'{key}_mean_angle'] = round(angles.mean(), 1)  # average
            insights[f'{key}_std_angle']  = round(angles.std(), 1)   # consistency
        else:
            insights[f'{key}_min_angle']  = None
            insights[f'{key}_max_angle']  = None
            insights[f'{key}_mean_angle'] = None
            insights[f'{key}_std_angle']  = None

    # --- Valgus analysis ---
    if len(right) > 0:
        insights['right_valgus_pct'] = round(right['knee_valgus'].mean() * 100, 1)
    if len(left) > 0:
        insights['left_valgus_pct']  = round(left['knee_valgus'].mean() * 100, 1)

    # --- Asymmetry analysis ---
    asym = df['asymmetry_deg'].dropna()
    if len(asym) > 0:
        insights['mean_asymmetry'] = round(asym.mean(), 1)
        insights['max_asymmetry']  = round(asym.max(), 1)
        insights['dangerous_asym_pct'] = round((asym > 15).mean() * 100, 1)
    else:
        insights['mean_asymmetry']      = None
        insights['max_asymmetry']       = None
        insights['dangerous_asym_pct']  = None

    # --- Risk score analysis ---
    risk = df['risk_score_0_to_100'].dropna()
    if len(risk) > 0:
        insights['mean_risk']       = round(risk.mean(), 1)
        insights['max_risk']        = round(risk.max(), 1)
        insights['high_risk_pct']   = round((risk > 50).mean() * 100, 1)
        insights['low_risk_pct']    = round((risk < 20).mean() * 100, 1)
    else:
        insights['mean_risk']     = None
        insights['max_risk']      = None
        insights['high_risk_pct'] = None
        insights['low_risk_pct']  = None

    # --- Fatigue analysis ---
    fatigue = df['fatigue_detected']
    insights['fatigue_pct'] = round(fatigue.mean() * 100, 1)

    # --- Squat depth classification ---
    if insights['right_min_angle'] is not None:
        min_a = insights['right_min_angle']
        if min_a < 70:
            insights['squat_depth_grade'] = "Excellent (Deep Squat)"
            insights['squat_depth_color'] = 'green'
        elif min_a < 100:
            insights['squat_depth_grade'] = "Good (Parallel Squat)"
            insights['squat_depth_color'] = 'limegreen'
        elif min_a < 130:
            insights['squat_depth_grade'] = "Shallow (Partial Squat)"
            insights['squat_depth_color'] = 'orange'
        else:
            insights['squat_depth_grade'] = "Too Shallow"
            insights['squat_depth_color'] = 'red'
    else:
        insights['squat_depth_grade'] = "No data"
        insights['squat_depth_color'] = 'gray'

    return insights


# ============================================================
# SECTION 3: GENERATE PLAIN ENGLISH RECOMMENDATIONS
# ============================================================

def generate_recommendations(insights):
    recs = []

    # Valgus
    right_valgus = insights.get('right_valgus_pct', 0) or 0
    left_valgus  = insights.get('left_valgus_pct',  0) or 0
    if right_valgus > 20 or left_valgus > 20:
        recs.append({
            "priority": "HIGH",
            "issue":    "Knee Valgus (Knee Caving Inward)",
            "detail":   f"Right knee: {right_valgus}% of frames | Left knee: {left_valgus}% of frames",
            "fix":      "Strengthen glutes and hip abductors. Consciously push knees outward "
                        "over your little toe during squats. Consider box squats to practice alignment.",
            "color":    "red"
        })
    elif right_valgus > 5 or left_valgus > 5:
        recs.append({
            "priority": "MEDIUM",
            "issue":    "Mild Knee Valgus Tendency",
            "detail":   f"Right: {right_valgus}% | Left: {left_valgus}% of frames",
            "fix":      "Add resistance band squats to build hip abductor awareness.",
            "color":    "orange"
        })

    # Asymmetry
    mean_asym = insights.get('mean_asymmetry') or 0
    if mean_asym > 20:
        recs.append({
            "priority": "HIGH",
            "issue":    "Significant Left-Right Asymmetry",
            "detail":   f"Average asymmetry: {mean_asym}° (danger threshold: 15°)",
            "fix":      "Perform single-leg exercises (Bulgarian split squats, single-leg press) "
                        "to address the weaker side. Never train asymmetry away with bilateral movements.",
            "color":    "red"
        })
    elif mean_asym > 10:
        recs.append({
            "priority": "MEDIUM",
            "issue":    "Moderate Left-Right Asymmetry",
            "detail":   f"Average asymmetry: {mean_asym}°",
            "fix":      "Include unilateral (single-leg) training 2x per week.",
            "color":    "orange"
        })

    # Squat depth
    grade = insights.get('squat_depth_grade', '')
    if "Shallow" in grade or "Too Shallow" in grade:
        min_a = insights.get('right_min_angle', 'N/A')
        recs.append({
            "priority": "MEDIUM",
            "issue":    "Insufficient Squat Depth",
            "detail":   f"Deepest angle recorded: {min_a}° (target: below 100°)",
            "fix":      "Work on ankle and hip mobility. Try box squats sitting to a low box. "
                        "Shallow squats reduce training effectiveness and shift load incorrectly.",
            "color":    "orange"
        })

    # Fatigue
    fatigue_pct = insights.get('fatigue_pct', 0) or 0
    if fatigue_pct > 30:
        recs.append({
            "priority": "MEDIUM",
            "issue":    "High Fatigue-Related Form Breakdown",
            "detail":   f"Fatigue detected in {fatigue_pct}% of session",
            "fix":      "Reduce rep count or add longer rest periods between sets. "
                        "Training while fatigued increases injury risk by up to 40%.",
            "color":    "orange"
        })

    # Overall risk
    high_risk_pct = insights.get('high_risk_pct', 0) or 0
    if high_risk_pct > 25:
        recs.append({
            "priority": "HIGH",
            "issue":    "Consistently High Injury Risk Score",
            "detail":   f"{high_risk_pct}% of frames had risk score above 50/100",
            "fix":      "Review all other recommendations above. Consider working with a "
                        "physiotherapist or strength coach to correct movement patterns.",
            "color":    "red"
        })

    # Good form — positive feedback
    if not recs:
        recs.append({
            "priority": "GOOD",
            "issue":    "Excellent Movement Quality",
            "detail":   "No significant injury risk factors detected",
            "fix":      "Maintain current form. Consider progressive overload.",
            "color":    "green"
        })

    return recs


# ============================================================
# SECTION 4: PRINT TERMINAL REPORT
# ============================================================

def print_report(insights, recommendations):
    print("\n" + "=" * 60)
    print("       BIOMECHANICS INJURY PREVENTION REPORT")
    print("=" * 60)

    print(f"\n SESSION SUMMARY")
    print(f"  Duration      : {insights['session_duration']} seconds")
    print(f"  Total Frames  : {insights['total_frames']}")
    print(f"  Reps Detected : {insights['total_reps']}")

    print(f"\n KNEE ANGLE ANALYSIS")
    for side, key in [("Right", "right"), ("Left", "left")]:
        mn = insights.get(f'{key}_min_angle')
        mx = insights.get(f'{key}_max_angle')
        av = insights.get(f'{key}_mean_angle')
        sd = insights.get(f'{key}_std_angle')
        if mn:
            print(f"  {side} Knee:")
            print(f"    Deepest squat  : {mn}°  ← most important number")
            print(f"    Most extended  : {mx}°")
            print(f"    Average angle  : {av}°")
            print(f"    Consistency    : ±{sd}° (lower = more consistent)")

    print(f"\n SQUAT DEPTH GRADE  :  {insights['squat_depth_grade']}")

    print(f"\n INJURY RISK FACTORS")
    print(f"  Right valgus       : {insights.get('right_valgus_pct', 'N/A')}% of frames")
    print(f"  Left valgus        : {insights.get('left_valgus_pct',  'N/A')}% of frames")
    print(f"  Mean asymmetry     : {insights.get('mean_asymmetry',   'N/A')}°")
    print(f"  Fatigue detected   : {insights.get('fatigue_pct',      'N/A')}% of session")
    print(f"  Mean risk score    : {insights.get('mean_risk',        'N/A')}/100")
    print(f"  High risk frames   : {insights.get('high_risk_pct',    'N/A')}%")

    print(f"\n RECOMMENDATIONS")
    for i, rec in enumerate(recommendations, 1):
        print(f"\n  [{rec['priority']}] {rec['issue']}")
        print(f"  Detail : {rec['detail']}")
        print(f"  Fix    : {rec['fix']}")

    print("\n" + "=" * 60)


# ============================================================
# SECTION 5: VISUAL DASHBOARD
# ============================================================

def generate_dashboard(df, right, left, insights, recommendations):
    fig = plt.figure(figsize=(18, 12), facecolor='#0d1117')
    fig.suptitle('Biomechanics Injury Prevention Dashboard',
                 fontsize=20, color='white', fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(3, 3, figure=fig,
                           hspace=0.45, wspace=0.35,
                           left=0.06, right=0.97,
                           top=0.93, bottom=0.05)

    DARK   = '#0d1117'
    PANEL  = '#161b22'
    GREEN  = '#3fb950'
    ORANGE = '#d29922'
    RED    = '#f85149'
    BLUE   = '#58a6ff'
    WHITE  = '#e6edf3'
    GRAY   = '#8b949e'

    def style_ax(ax, title):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=GRAY, labelsize=8)
        ax.spines['bottom'].set_color('#30363d')
        ax.spines['left'].set_color('#30363d')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title(title, color=WHITE, fontsize=10, fontweight='bold', pad=8)
        ax.yaxis.label.set_color(GRAY)
        ax.xaxis.label.set_color(GRAY)

    # ── Plot 1: Knee Angle Over Time ──────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    style_ax(ax1, "Knee Angle Over Time")

    if len(right) > 0:
        ax1.plot(right['timestamp_s'], right['knee_angle_deg'],
                 color=BLUE, linewidth=1.5, label='Right Knee', alpha=0.9)
    if len(left) > 0:
        ax1.plot(left['timestamp_s'], left['knee_angle_deg'],
                 color=GREEN, linewidth=1.5, label='Left Knee', alpha=0.9, linestyle='--')

    ax1.axhline(y=100, color=ORANGE, linewidth=1, linestyle=':', alpha=0.6, label='Parallel (100°)')
    ax1.axhline(y=70,  color=GREEN,  linewidth=1, linestyle=':', alpha=0.6, label='Deep (70°)')
    ax1.set_ylabel("Angle (degrees)")
    ax1.set_xlabel("Time (seconds)")
    ax1.legend(facecolor=PANEL, edgecolor='#30363d',
               labelcolor=WHITE, fontsize=8)
    ax1.invert_yaxis()   # Lower angle = deeper squat, visually intuitive

    # ── Plot 2: Risk Score Over Time ──────────────────────────
    ax2 = fig.add_subplot(gs[1, :2])
    style_ax(ax2, "Injury Risk Score Over Time (0 = Safe, 100 = Dangerous)")

    risk_data = df.groupby('timestamp_s')['risk_score_0_to_100'].max().reset_index()
    if len(risk_data) > 0:
        colors_risk = [RED if r > 50 else ORANGE if r > 20 else GREEN
                       for r in risk_data['risk_score_0_to_100']]
        ax2.bar(risk_data['timestamp_s'], risk_data['risk_score_0_to_100'],
                color=colors_risk, width=0.25, alpha=0.8)
        ax2.axhline(y=50, color=RED,    linewidth=1, linestyle='--', alpha=0.5, label='High Risk (50)')
        ax2.axhline(y=20, color=ORANGE, linewidth=1, linestyle='--', alpha=0.5, label='Medium Risk (20)')
        ax2.set_ylim(0, 105)
        ax2.set_ylabel("Risk Score")
        ax2.set_xlabel("Time (seconds)")
        ax2.legend(facecolor=PANEL, edgecolor='#30363d', labelcolor=WHITE, fontsize=8)

    # ── Plot 3: Squat Depth Grade (Gauge) ─────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    style_ax(ax3, "Squat Depth Grade")
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')

    min_angle = insights.get('right_min_angle') or insights.get('left_min_angle') or 180
    grade     = insights['squat_depth_grade']
    g_color   = insights['squat_depth_color']

    ax3.text(0.5, 0.72, f"{min_angle}°",
             ha='center', va='center', fontsize=42,
             color=g_color, fontweight='bold',
             transform=ax3.transAxes)
    ax3.text(0.5, 0.42, "Peak Flexion",
             ha='center', fontsize=10, color=GRAY, transform=ax3.transAxes)
    ax3.text(0.5, 0.28, grade,
             ha='center', fontsize=11, color=g_color,
             fontweight='bold', transform=ax3.transAxes)

    ranges = [("< 70°  Excellent", GREEN),
              ("70-100° Good",     'limegreen'),
              ("100-130° Shallow", ORANGE),
              ("> 130° Too Shallow", RED)]
    for i, (label, col) in enumerate(ranges):
        ax3.text(0.5, 0.16 - i * 0.065, label,
                 ha='center', fontsize=8, color=col, transform=ax3.transAxes)

    # ── Plot 4: Left vs Right Comparison ─────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    style_ax(ax4, "Left vs Right Knee Comparison")

    metrics      = ['Min Angle\n(Deeper=Better)', 'Mean Angle', 'Valgus %']
    right_vals   = [
        insights.get('right_min_angle')  or 0,
        insights.get('right_mean_angle') or 0,
        insights.get('right_valgus_pct') or 0
    ]
    left_vals = [
        insights.get('left_min_angle')  or 0,
        insights.get('left_mean_angle') or 0,
        insights.get('left_valgus_pct') or 0
    ]

    x      = np.arange(len(metrics))
    width  = 0.35
    bars_r = ax4.bar(x - width/2, right_vals, width, label='Right', color=BLUE,   alpha=0.8)
    bars_l = ax4.bar(x + width/2, left_vals,  width, label='Left',  color=GREEN,  alpha=0.8)
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics, fontsize=7)
    ax4.legend(facecolor=PANEL, edgecolor='#30363d', labelcolor=WHITE, fontsize=8)

    for bar in list(bars_r) + list(bars_l):
        h = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., h + 0.5,
                 f'{h:.1f}', ha='center', va='bottom', fontsize=7, color=WHITE)

    # ── Plot 5: Asymmetry Over Time ───────────────────────────
    ax5 = fig.add_subplot(gs[2, :2])
    style_ax(ax5, "Left-Right Asymmetry Over Time (> 15° = Injury Risk)")

    asym_data = df[['timestamp_s', 'asymmetry_deg']].dropna()
    if len(asym_data) > 0:
        asym_grouped = asym_data.groupby('timestamp_s')['asymmetry_deg'].mean().reset_index()
        ax5.fill_between(asym_grouped['timestamp_s'], asym_grouped['asymmetry_deg'],
                         alpha=0.3, color=ORANGE)
        ax5.plot(asym_grouped['timestamp_s'], asym_grouped['asymmetry_deg'],
                 color=ORANGE, linewidth=1.5)
        ax5.axhline(y=15, color=RED, linewidth=1.5, linestyle='--', label='Danger threshold (15°)')
        ax5.set_ylabel("Asymmetry (degrees)")
        ax5.set_xlabel("Time (seconds)")
        ax5.legend(facecolor=PANEL, edgecolor='#30363d', labelcolor=WHITE, fontsize=8)

    # ── Plot 6: Recommendations Panel ────────────────────────
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.set_facecolor(PANEL)
    ax6.axis('off')
    ax6.set_title("Key Recommendations", color=WHITE,
                  fontsize=10, fontweight='bold', pad=8)

    priority_colors = {'HIGH': RED, 'MEDIUM': ORANGE, 'GOOD': GREEN}
    y_pos = 0.95

    for rec in recommendations[:4]:   # show top 4
        col = priority_colors.get(rec['priority'], WHITE)
        ax6.text(0.0, y_pos, f"[{rec['priority']}]",
                 transform=ax6.transAxes, fontsize=8,
                 color=col, fontweight='bold', va='top')
        ax6.text(0.22, y_pos, rec['issue'],
                 transform=ax6.transAxes, fontsize=8,
                 color=WHITE, va='top', wrap=True)

        # Wrap fix text manually
        fix_text = rec['fix'][:60] + '...' if len(rec['fix']) > 60 else rec['fix']
        ax6.text(0.22, y_pos - 0.07, fix_text,
                 transform=ax6.transAxes, fontsize=7,
                 color=GRAY, va='top', style='italic')

        y_pos -= 0.22

    plt.savefig(OUTPUT_IMG, dpi=150, bbox_inches='tight',
                facecolor=DARK, edgecolor='none')
    print(f"\n[DASHBOARD] Saved as '{OUTPUT_IMG}'")
    plt.show()


# ============================================================
# SECTION 6: MAIN
# ============================================================

def main():
    print(f"\n[LOADING] Reading {CSV_FILE}...")

    try:
        df, right, left = load_data(CSV_FILE)
    except FileNotFoundError:
        print(f"ERROR: '{CSV_FILE}' not found.")
        print("Make sure you run the motion capture script first to generate data.")
        return
    except Exception as e:
        print(f"ERROR reading CSV: {e}")
        return

    print(f"[LOADED] {len(df)} rows of data found.")

    if len(df) == 0:
        print("ERROR: CSV file is empty. Run the motion capture script first.")
        return

    insights        = compute_insights(df, right, left)
    recommendations = generate_recommendations(insights)

    print_report(insights, recommendations)
    generate_dashboard(df, right, left, insights, recommendations)


if __name__ == "__main__":
    main()