from pathlib import Path
import os, json, sqlite3, csv, io
from datetime import datetime
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for, Response, flash

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "model" / "liver_stage_pipeline.joblib"))
METRICS_PATH = BASE_DIR / "model" / "evaluation_metrics.json"
DATA_PATH = BASE_DIR / "data" / "liver_cirrhosis.csv"
DB_PATH = BASE_DIR / "liverai_history.db"
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "change-this-for-deployment")
model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

NUMERIC_FIELDS = ["N_Days", "Age", "Bilirubin", "Cholesterol", "Albumin", "Copper", "Alk_Phos", "SGOT", "Tryglicerides", "Platelets", "Prothrombin"]
CATEGORICAL_FIELDS = ["Drug", "Sex", "Ascites", "Hepatomegaly", "Spiders", "Edema"]
FEATURES = NUMERIC_FIELDS + CATEGORICAL_FIELDS
FIELD_LABELS = {"N_Days":"Follow-up days", "Age":"Age (years)", "Bilirubin":"Bilirubin", "Cholesterol":"Cholesterol", "Albumin":"Albumin", "Copper":"Copper", "Alk_Phos":"Alkaline phosphatase", "SGOT":"SGOT", "Tryglicerides":"Triglycerides", "Platelets":"Platelets", "Prothrombin":"Prothrombin time", "Drug":"Treatment group", "Sex":"Sex", "Ascites":"Ascites", "Hepatomegaly":"Hepatomegaly", "Spiders":"Spider angiomas", "Edema":"Edema"}


def connect_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with connect_db() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS predictions (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, patient_label TEXT NOT NULL, stage INTEGER NOT NULL, confidence REAL, probabilities TEXT NOT NULL, inputs TEXT NOT NULL)")
init_db()


def make_input(source, age_in_years=False):
    row = {}
    for field in NUMERIC_FIELDS:
        value = source.get(field, "")
        if value is None or str(value).strip() == "": raise ValueError(f"Please enter {FIELD_LABELS[field]}.")
        value = float(value)
        if field == "Age" and age_in_years: value *= 365.25
        if value < 0: raise ValueError(f"{FIELD_LABELS[field]} cannot be negative.")
        row[field] = value
    for field in CATEGORICAL_FIELDS:
        value = source.get(field, "")
        if value is None or str(value).strip() == "": raise ValueError(f"Please select {FIELD_LABELS[field]}.")
        row[field] = str(value).strip()
    return pd.DataFrame([row], columns=FEATURES)


def predict_one(X):
    prediction = int(model.predict(X)[0])
    probabilities = {}
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        classes = model.named_steps["model"].classes_
        probabilities = {f"Stage {int(c)}": round(float(p)*100, 2) for c,p in zip(classes, probs)}
    return {"stage": prediction, "probabilities": probabilities, "confidence": max(probabilities.values()) if probabilities else None}


def metrics_data():
    if METRICS_PATH.exists():
        try: return json.loads(METRICS_PATH.read_text(encoding='utf-8'))
        except (ValueError, OSError): pass
    return {}


def dataset_summary():
    try:
        df = pd.read_csv(DATA_PATH)
        counts = df['Stage'].value_counts().sort_index().to_dict()
        return {"rows":len(df), "stage_counts":{str(k):int(v) for k,v in counts.items()}, "features":len(FEATURES)}
    except Exception: return {"rows":0,"stage_counts":{},"features":len(FEATURES)}


def recent_predictions(limit=5):
    with connect_db() as conn:
        return conn.execute('SELECT * FROM predictions ORDER BY id DESC LIMIT ?', (limit,)).fetchall()

@app.context_processor
def inject_globals():
    return {"model_ready": model is not None, "nav_endpoint": request.endpoint or ""}

@app.route('/')
def dashboard():
    with connect_db() as conn:
        total = conn.execute('SELECT COUNT(*) FROM predictions').fetchone()[0]
        stage_counts = {str(r['stage']):r['n'] for r in conn.execute('SELECT stage, COUNT(*) n FROM predictions GROUP BY stage')}
    return render_template('dashboard.html', title='Dashboard', metrics=metrics_data(), data=dataset_summary(), total=total, stage_counts=stage_counts, recent=recent_predictions(6))

@app.route('/predict', methods=['GET','POST'])
def predict():
    result = error = None
    form_values = {}
    if request.method == 'POST':
        form_values = request.form.to_dict()
        if model is None: error = 'The model has not been trained yet. Run train_model.py first.'
        else:
            try:
                X = make_input(request.form, age_in_years=True)
                result = predict_one(X)
                label = (request.form.get('patient_label') or 'Anonymous case').strip()[:80]
                safe_inputs = request.form.to_dict()
                safe_inputs.pop('patient_label', None)
                with connect_db() as conn:
                    cur = conn.execute('INSERT INTO predictions(created_at,patient_label,stage,confidence,probabilities,inputs) VALUES(?,?,?,?,?,?)', (datetime.now().strftime('%Y-%m-%d %H:%M'), label, result['stage'], result['confidence'], json.dumps(result['probabilities']), json.dumps(safe_inputs)))
                    result['record_id'] = cur.lastrowid
            except (ValueError, TypeError) as exc: error = str(exc)
            except Exception as exc: error = f'Prediction failed: {exc}'
    return render_template('predict.html', title='New prediction', result=result, error=error, form_values=form_values)

@app.route('/history')
def history():
    with connect_db() as conn: rows = conn.execute('SELECT * FROM predictions ORDER BY id DESC').fetchall()
    return render_template('history.html', title='Prediction history', rows=rows)

@app.route('/history/<int:record_id>')
def history_detail(record_id):
    with connect_db() as conn: row = conn.execute('SELECT * FROM predictions WHERE id=?', (record_id,)).fetchone()
    if not row: return render_template('not_found.html', title='Record not found'), 404
    return render_template('detail.html', title='Prediction details', row=row, inputs=json.loads(row['inputs']), probabilities=json.loads(row['probabilities']))

@app.route('/history/export.csv')
def export_history():
    with connect_db() as conn: rows = conn.execute('SELECT id,created_at,patient_label,stage,confidence,probabilities FROM predictions ORDER BY id DESC').fetchall()
    output = io.StringIO(); writer = csv.writer(output)
    writer.writerow(['Record ID','Created at','Case label','Predicted stage','Model probability (%)','Class probabilities'])
    for r in rows: writer.writerow([r['id'],r['created_at'],r['patient_label'],r['stage'],r['confidence'],r['probabilities']])
    return Response(output.getvalue(), mimetype='text/csv', headers={'Content-Disposition':'attachment; filename=liverai_prediction_history.csv'})

@app.route('/insights')
def insights():
    return render_template('insights.html', title='Model insights', metrics=metrics_data(), data=dataset_summary())

@app.route('/about')
def about(): return render_template('about.html', title='About LiverAI')

@app.route('/api/health')
def health(): return jsonify({"status":"ok", "model_loaded":model is not None, "database":"ready"})

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if model is None: return jsonify({"success":False,"error":"Model not trained; run train_model.py."}),503
    payload=request.get_json(silent=True) or {}; missing=[f for f in FEATURES if f not in payload]
    if missing: return jsonify({"success":False,"error":"Missing fields","fields":missing}),400
    try: return jsonify({"success":True,"prediction":predict_one(make_input(payload,age_in_years=True))})
    except Exception as exc: return jsonify({"success":False,"error":str(exc)}),400

if __name__ == '__main__': app.run(debug=True)
