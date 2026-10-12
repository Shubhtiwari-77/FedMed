from __future__ import annotations

import json
import math
import os
from http.server import BaseHTTPRequestHandler


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT_DIR, "outputs", "vercel_model.json")

DISEASES = [
  "Viral_URI", "Influenza", "Dengue", "Chikungunya", "Malaria", "Typhoid",
  "Bacterial_Pneumonia", "Autoimmune_Flare",
]
REGIONS = ["Tropical_Monsoon", "Tropical_Dry", "Temperate", "Cold_Subzero"]
SEASONS = ["Winter", "Summer", "Monsoon", "Post_Monsoon"]
BINARY_SYMPTOMS = [
  "joint_pain", "rash", "retro_orbital_pain", "headache", "cough",
  "sore_throat", "breathlessness", "vomiting", "abdominal_pain", "chills_rigor",
]
SCALE = {
  "age": (45.0, 45.0), "fever_days": (10.0, 10.0), "temp_max_c": (38.5, 2.5),
  "platelet_count_k": (250.0, 200.0), "wbc_count_k": (9.0, 8.0),
  "hemoglobin_gdl": (13.0, 5.0), "crp_mgl": (60.0, 60.0), "spo2_pct": (95.0, 8.0),
  "systolic_bp": (120.0, 40.0),
}
MONTH_TO_SEASON = {
  12: "Winter", 1: "Winter", 2: "Winter", 3: "Summer", 4: "Summer", 5: "Summer",
  6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
  10: "Post_Monsoon", 11: "Post_Monsoon",
}
MISSING_FLAGS = [f"{name}_was_missing" for name in list(SCALE)[:-1]] + ["systolic_bp_was_missing"]


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FedMed-FL CDSS</title>
  <style>
    :root { color-scheme: light; font-family: system-ui, sans-serif; }
    body { max-width: 52rem; margin: 3rem auto; padding: 0 1.25rem; color: #17212b; }
    h1 { margin-bottom: .35rem; }
    p { line-height: 1.5; }
    form { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1rem; margin-top: 2rem; }
    label { display: grid; gap: .35rem; font-weight: 600; }
    input, select, button { box-sizing: border-box; font: inherit; padding: .65rem; border: 1px solid #aab4bd; border-radius: .35rem; }
    button { grid-column: 1 / -1; cursor: pointer; background: #176b87; color: white; border: 0; }
    pre { white-space: pre-wrap; background: #f0f4f6; padding: 1rem; border-radius: .35rem; }
    .notice { padding: .8rem 1rem; background: #fff4d6; border-left: .25rem solid #d99a00; }
    @media (max-width: 36rem) { form { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <h1>FedMed-FL Clinical Decision Support</h1>
  <p>Context-aware differential diagnosis powered by the federated global model.</p>
  <p class="notice">Decision support only. Confirm every result with an appropriately qualified clinician and laboratory testing.</p>
  <form id="patient-form">
    <label>Region<select name="region"><option>Tropical_Monsoon</option><option>Tropical_Dry</option><option>Temperate</option><option>Cold_Subzero</option></select></label>
    <label>Month<input name="month" type="number" min="1" max="12" value="8"></label>
    <label>Age<input name="age" type="number" min="1" max="95" value="31"></label>
    <label>Fever days<input name="fever_days" type="number" min="0" max="30" value="12"></label>
    <label>Maximum temperature (C)<input name="temp_max_c" type="number" step="0.1" value="39.6"></label>
    <label>Platelets (x1000/uL)<input name="platelet_count_k" type="number" value="82"></label>
    <label>WBC (x1000/uL)<input name="wbc_count_k" type="number" step="0.1" value="3.2"></label>
    <label>SpO2 (%)<input name="spo2_pct" type="number" value="97"></label>
    <button type="submit">Rank differential diagnosis</button>
  </form>
  <pre id="result" hidden></pre>
  <script>
    const form = document.querySelector('#patient-form');
    const result = document.querySelector('#result');
    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      const patient = Object.fromEntries(new FormData(form));
      for (const key of ['month', 'age', 'fever_days', 'temp_max_c', 'platelet_count_k', 'wbc_count_k', 'spo2_pct']) {
        patient[key] = Number(patient[key]);
      }
      result.hidden = false;
      result.textContent = 'Calculating...';
      const response = await fetch('/api', { method: 'POST', headers: {'content-type': 'application/json'}, body: JSON.stringify(patient) });
      result.textContent = JSON.stringify(await response.json(), null, 2);
    });
  </script>
</body>
</html>"""


def _json(handler: BaseHTTPRequestHandler, status: int, payload: dict) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _patient_features(patient: dict) -> list[float]:
    defaults = {
        "age": 30.0, "gender": 1, "fever_days": 3.0, "temp_max_c": 38.5,
        "platelet_count_k": 250.0, "wbc_count_k": 8.0, "hemoglobin_gdl": 13.0,
        "crp_mgl": 20.0, "spo2_pct": 98.0, "systolic_bp": 120.0,
        "region": "Tropical_Monsoon", "month": 8,
    }
    values = defaults | patient
    features = []
    for name, (mid, half) in SCALE.items():
        features.append((float(values[name]) - mid) / half)
    features.extend(float(values.get(name, 0)) for name in BINARY_SYMPTOMS)
    features.append(float(values["gender"]))
    features.extend(float(values["region"] == region) for region in REGIONS)
    month = int(values["month"])
    season = MONTH_TO_SEASON[month]
    features.extend(float(season == value) for value in SEASONS)
    features.extend((math.sin(2 * math.pi * month / 12), math.cos(2 * math.pi * month / 12)))
    features.extend(0.0 for _ in MISSING_FLAGS)
    return features


def _linear(values: list[float], weights: list[list[float]], bias: list[float]) -> list[float]:
    return [sum(weight * value for weight, value in zip(row, values)) + offset
            for row, offset in zip(weights, bias)]


def _predict(patient: dict) -> list[dict]:
    with open(MODEL_PATH, encoding="utf-8") as file:
        state = json.load(file)["state_dict"]
    values = _patient_features(patient)
    for index in (0, 4, 8):
        values = _linear(values, state[f"net.{index}.weight"], state[f"net.{index}.bias"])
        if index != 8:
            norm_weight = state[f"net.{index + 1}.weight"]
            norm_bias = state[f"net.{index + 1}.bias"]
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            values = [
                ((value - mean) / math.sqrt(variance + 1e-5)) * scale + offset
                for value, scale, offset in zip(values, norm_weight, norm_bias)
            ]
            values = [max(value, 0.0) for value in values]
    maximum = max(values)
    probabilities = [math.exp(value - maximum) for value in values]
    total = sum(probabilities)
    probabilities = [value / total for value in probabilities]
    order = sorted(range(len(probabilities)), key=probabilities.__getitem__, reverse=True)[:5]
    return [
        {
            "rank": rank,
            "differential_diagnosis": DISEASES[index],
            "confidence": f"{probabilities[index] * 100:5.2f}%",
            "probability": round(probabilities[index], 4),
        }
        for rank, index in enumerate(order, 1)
    ]


class handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        body = HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            patient = json.loads(self.rfile.read(length) or b"{}")
            if not os.path.exists(MODEL_PATH):
                _json(self, 503, {
                "error": f"No Vercel model at {MODEL_PATH}.",
                "hint": "Run python scripts/export_vercel_model.py before deploying.",
                })
                return
            _json(self, 200, {"results": _predict(patient)})
        except FileNotFoundError as error:
            _json(self, 503, {"error": str(error), "hint": "Export the Vercel model before deploying."})
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            _json(self, 400, {"error": f"Invalid patient payload: {error}"})
        except Exception as error:
            _json(self, 500, {"error": str(error)})

    def log_message(self, format: str, *args: object) -> None:
        return