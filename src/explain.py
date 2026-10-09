"""Plain-language risk explanation engine for CareTrail Phase 3.

Provides deterministic, transparent, non-LLM feature contribution explanations
derived directly from PyTorch Logistic Regression weights:
    logit = w^T * x + b
    probability = sigmoid(logit)

Identifies top-K contributing factors (|w_i * x_i|) with patient raw measurements,
standard medical units, direction (increases vs decreases risk), and relative scores.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import torch

from src.data_prep import FEATURE_COLUMNS
from src.model import LogisticRegressionModel

GUARDRAIL_TEXT: str = (
    "This is an educational demo for ForgeHacks and does not constitute medical advice or clinical diagnosis."
)

FEATURE_METADATA: Dict[str, Dict[str, Any]] = {
    "age": {
        "display_name": "Age",
        "unit": "years",
        "format": lambda v: f"{int(round(float(v)))} years",
    },
    "sex": {
        "display_name": "Biological Sex",
        "unit": "",
        "format": lambda v: "Male" if float(v) == 1.0 else "Female",
    },
    "cp": {
        "display_name": "Chest Pain Type",
        "unit": "",
        "format": lambda v: {
            1: "Typical Angina (1)",
            2: "Atypical Angina (2)",
            3: "Non-anginal Pain (3)",
            4: "Asymptomatic (4)",
        }.get(int(round(float(v))), f"Type {v}"),
    },
    "trestbps": {
        "display_name": "Resting Blood Pressure",
        "unit": "mm Hg",
        "format": lambda v: f"{int(round(float(v)))} mm Hg",
    },
    "chol": {
        "display_name": "Serum Cholesterol",
        "unit": "mg/dl",
        "format": lambda v: f"{int(round(float(v)))} mg/dl",
    },
    "fbs": {
        "display_name": "Fasting Blood Sugar > 120 mg/dl",
        "unit": "",
        "format": lambda v: "Elevated (>120 mg/dl)" if float(v) == 1.0 else "Normal (≤120 mg/dl)",
    },
    "restecg": {
        "display_name": "Resting ECG Results",
        "unit": "",
        "format": lambda v: {
            0: "Normal (0)",
            1: "ST-T Wave Abnormality (1)",
            2: "Left Ventricular Hypertrophy (2)",
        }.get(int(round(float(v))), str(v)),
    },
    "thalach": {
        "display_name": "Maximum Heart Rate Achieved",
        "unit": "bpm",
        "format": lambda v: f"{int(round(float(v)))} bpm",
    },
    "exang": {
        "display_name": "Exercise-Induced Angina",
        "unit": "",
        "format": lambda v: "Yes (Present)" if float(v) == 1.0 else "No (Absent)",
    },
    "oldpeak": {
        "display_name": "ST Depression (Exercise)",
        "unit": "mm",
        "format": lambda v: f"{float(v):.1f} mm",
    },
    "slope": {
        "display_name": "Slope of Peak Exercise ST",
        "unit": "",
        "format": lambda v: {
            1: "Upsloping (1)",
            2: "Flat (2)",
            3: "Downsloping (3)",
        }.get(int(round(float(v))), str(v)),
    },
    "ca": {
        "display_name": "Fluoroscopy Major Vessels",
        "unit": "vessels",
        "format": lambda v: "Not recorded" if pd.isna(v) else f"{int(round(float(v)))} colored vessels",
    },
    "thal": {
        "display_name": "Thallium Scintigraphy",
        "unit": "",
        "format": lambda v: {
            3: "Normal (3)",
            6: "Fixed Defect (6)",
            7: "Reversible Defect (7)",
        }.get(int(round(float(v))), str(v)),
    },
}


def explain_prediction(
    model: LogisticRegressionModel,
    scaled_features: Union[np.ndarray, torch.Tensor],
    raw_features: Union[Dict[str, Any], pd.Series],
    feature_names: Optional[List[str]] = None,
    top_k: int = 3,
) -> Dict[str, Any]:
    """Generate a deterministic, transparent plain-language explanation for a prediction.

    Args:
        model: Trained LogisticRegressionModel.
        scaled_features: Normalized 1D feature array of length 13.
        raw_features: Patient raw unscaled measurements (dict or Series).
        feature_names: Ordered list of feature names (defaults to FEATURE_COLUMNS).
        top_k: Number of top contributing factors to highlight (default 3).

    Returns:
        Structured explanation dictionary with:
        - logit: float
        - probability: float (0.0 to 1.0)
        - risk_category: "Low" | "Moderate" | "High"
        - top_factors: list of dicts with feature details and relative contribution
        - all_factors: complete list of ranked contributions
        - summary_text: plain-language narrative
        - guardrail: disclaimer string
    """
    if feature_names is None:
        feature_names = FEATURE_COLUMNS

    if isinstance(scaled_features, torch.Tensor):
        x = scaled_features.detach().cpu().numpy().flatten()
    else:
        x = np.asarray(scaled_features).flatten()

    # Extract model parameters
    weights = model.linear.weight.detach().cpu().squeeze().numpy()
    bias = float(model.linear.bias.detach().cpu().item())

    # Calculate logit and probability
    logit = float(np.dot(weights, x) + bias)
    probability = float(1.0 / (1.0 + np.exp(-logit)))

    # Contributions: c_i = w_i * x_i
    contributions = weights * x
    abs_contributions = np.abs(contributions)
    total_abs = float(np.sum(abs_contributions)) if np.sum(abs_contributions) > 0 else 1.0

    ranked_indices = np.argsort(abs_contributions)[::-1]

    factors: List[Dict[str, Any]] = []
    for idx in ranked_indices:
        fname = feature_names[idx]
        c_val = float(contributions[idx])
        abs_c = float(abs_contributions[idx])
        rel_score = float(abs_c / total_abs)

        meta = FEATURE_METADATA.get(fname, {})
        display_name = meta.get("display_name", fname)
        unit = meta.get("unit", "")
        formatter = meta.get("format", lambda v: str(v))

        raw_val = raw_features.get(fname, np.nan) if isinstance(raw_features, dict) else raw_features[fname]
        formatted_val = formatter(raw_val) if not pd.isna(raw_val) else "N/A"

        if c_val > 0.001:
            direction = "Increases Risk"
            direction_desc = "elevating predicted cardiovascular risk"
        elif c_val < -0.001:
            direction = "Decreases Risk"
            direction_desc = "lowering predicted cardiovascular risk"
        else:
            direction = "Neutral"
            direction_desc = "having minimal impact"

        factors.append({
            "feature": fname,
            "display_name": display_name,
            "raw_value": raw_val,
            "formatted_value": formatted_val,
            "unit": unit,
            "weight": float(weights[idx]),
            "scaled_value": float(x[idx]),
            "contribution": c_val,
            "abs_contribution": abs_c,
            "relative_score": round(rel_score, 4),
            "relative_percentage": round(rel_score * 100.0, 1),
            "direction": direction,
            "direction_description": direction_desc,
        })

    top_factors = factors[:top_k]

    # Risk categorization
    if probability < 0.35:
        risk_category = "Low Risk"
    elif probability < 0.65:
        risk_category = "Moderate Risk"
    else:
        risk_category = "High Risk"

    # Build narrative summary
    narrative_lines: List[str] = [
        f"The model estimates a **{risk_category}** ({probability * 100:.1f}%) probability of coronary artery disease."
    ]
    narrative_lines.append(f"Top {top_k} primary clinical drivers:")
    for i, f in enumerate(top_factors, start=1):
        narrative_lines.append(
            f"{i}. **{f['display_name']}** ({f['formatted_value']}): "
            f"**{f['direction']}** ({f['relative_percentage']}% relative contribution, {f['direction_description']})."
        )

    summary_text = "\n".join(narrative_lines)

    return {
        "logit": round(logit, 4),
        "probability": round(probability, 4),
        "probability_percent": round(probability * 100.0, 2),
        "risk_category": risk_category,
        "top_factors": top_factors,
        "all_factors": factors,
        "summary_text": summary_text,
        "guardrail": GUARDRAIL_TEXT,
    }
