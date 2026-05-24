from flask import Flask, render_template, request, jsonify, Response
import camera_morse
import os
import next_word_predictor
from next_word_predictor import WordPredictor
import io
import base64

app = Flask(__name__, template_folder="templates", static_folder="static")

# ================= Predictor =================
predictor = WordPredictor()
predictor.load_or_train()

# ================= Camera =================
morse_cam = None

def get_morse_cam():
    global morse_cam
    if morse_cam is None:
        morse_cam = camera_morse.MorseCam(predictor)
    return morse_cam

# ================= Storage =================
CORPUS_DIR = "corpus"
SAVE_FILE = os.path.join(CORPUS_DIR, "pizza.txt")

os.makedirs(CORPUS_DIR, exist_ok=True)
os.makedirs("static/css", exist_ok=True)

last_saved_text = ""

# ================= Routes =================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    cam = get_morse_cam()
    return Response(
        cam.generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/api/sync_text")
def sync_text():
    cam = get_morse_cam()
    snapshot = cam.get_state_snapshot()
    return jsonify(snapshot)

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json() or {}
    preds = predictor.predict_next_word(data.get("context", ""))
    return jsonify([p.to_dict() for p in preds])

@app.route("/api/complete", methods=["POST"])
def complete():
    data = request.get_json() or {}
    preds = predictor.complete_word(
        data.get("partial", ""),
        data.get("context", "")
    )
    return jsonify([p.to_dict() for p in preds])

@app.route("/api/save", methods=["POST"])
def save():
    global last_saved_text

    text = (request.json.get("text") or "").strip()

    if len(text) < 5:
        return jsonify({"saved": False, "reason": "too_short"})

    if text == last_saved_text:
        return jsonify({"saved": False, "reason": "duplicate"})

    text = next_word_predictor.clean_text(text)

    try:
        with open(SAVE_FILE, "a", encoding="utf-8") as f:
            f.write(text + "\n")

        predictor.ngram.learn_sentence(text)
        predictor.schedule_lstm_retrain()

        last_saved_text = text
        return jsonify({"saved": True})

    except Exception as e:
        return jsonify({"saved": False, "error": str(e)}), 500

@app.route("/api/control", methods=["POST"])
def control():
    data = request.get_json() or {}
    action = data.get("action")
    if not action:
        return jsonify({"success": False, "error": "No action provided"}), 400
    
    cam = get_morse_cam()
    cam.manual_control(action)
    return jsonify({"success": True})

# ================= Training Graphs =================
@app.route("/training-graphs")
def training_graphs():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ultralytics import YOLO

    model = YOLO("best.pt")
    r = model.ckpt.get("train_results", {})

    epochs = r.get("epoch", [])

    def make_chart(title, ylabel, series):
        """series: list of (label, values, color) tuples"""
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor("#0f1117")
        ax.set_facecolor("#1a1d2e")
        for label, values, color in series:
            ax.plot(epochs, values, label=label, color=color, linewidth=2)
        ax.set_title(title, color="white", fontsize=13, pad=10)
        ax.set_xlabel("Epoch", color="#aaa")
        ax.set_ylabel(ylabel, color="#aaa")
        ax.tick_params(colors="#aaa")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
        ax.legend(facecolor="#1a1d2e", edgecolor="#333", labelcolor="white", fontsize=9)
        ax.grid(True, color="#2a2d3e", linewidth=0.5)
        buf = io.BytesIO()
        plt.tight_layout()
        fig.savefig(buf, format="png", dpi=130, facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()

    charts = []

    # Box loss
    charts.append({
        "title": "Box Loss",
        "img": make_chart("Box Loss (train vs val)", "Loss", [
            ("train/box_loss", r.get("train/box_loss", []), "#00b4d8"),
            ("val/box_loss",   r.get("val/box_loss", []),   "#f77f00"),
        ])
    })

    # Cls loss
    charts.append({
        "title": "Classification Loss",
        "img": make_chart("Classification Loss (train vs val)", "Loss", [
            ("train/cls_loss", r.get("train/cls_loss", []), "#06d6a0"),
            ("val/cls_loss",   r.get("val/cls_loss", []),   "#ef233c"),
        ])
    })

    # DFL loss
    charts.append({
        "title": "DFL Loss",
        "img": make_chart("DFL Loss (train vs val)", "Loss", [
            ("train/dfl_loss", r.get("train/dfl_loss", []), "#7209b7"),
            ("val/dfl_loss",   r.get("val/dfl_loss", []),   "#f4a261"),
        ])
    })

    # Metrics
    charts.append({
        "title": "Detection Metrics",
        "img": make_chart("Metrics", "Score", [
            ("Precision",  r.get("metrics/precision(B)", []), "#00ffcc"),
            ("Recall",     r.get("metrics/recall(B)", []),    "#ffbe0b"),
            ("mAP50",      r.get("metrics/mAP50(B)", []),     "#3a86ff"),
            ("mAP50-95",   r.get("metrics/mAP50-95(B)", []),  "#ff006e"),
        ])
    })

    # Learning rate
    charts.append({
        "title": "Learning Rate",
        "img": make_chart("Learning Rate Schedule", "LR", [
            ("lr/pg0", r.get("lr/pg0", []), "#8ecae6"),
        ])
    })

    # Final metrics summary
    tm = model.ckpt.get("train_metrics", {})
    summary = {
        "Precision":  f"{tm.get('metrics/precision(B)', 0):.4f}",
        "Recall":     f"{tm.get('metrics/recall(B)', 0):.4f}",
        "mAP50":      f"{tm.get('metrics/mAP50(B)', 0):.4f}",
        "mAP50-95":   f"{tm.get('metrics/mAP50-95(B)', 0):.4f}",
        "Val Box Loss": f"{tm.get('val/box_loss', 0):.5f}",
        "Val Cls Loss": f"{tm.get('val/cls_loss', 0):.5f}",
        "Val DFL Loss": f"{tm.get('val/dfl_loss', 0):.5f}",
        "Fitness":    f"{tm.get('fitness', 0):.5f}",
    }

    train_args = model.ckpt.get("train_args", {})
    info = {
        "Base Model": train_args.get("model", "yolov8n.pt"),
        "Epochs Configured": train_args.get("epochs", "—"),
        "Epochs Trained": len(epochs),
        "Batch Size": train_args.get("batch", "—"),
        "Image Size": train_args.get("imgsz", "—"),
        "Optimizer": train_args.get("optimizer", "—"),
        "LR0": train_args.get("lr0", "—"),
    }

    return render_template("training_graphs.html", charts=charts, summary=summary, info=info)

# ================= Main =================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
