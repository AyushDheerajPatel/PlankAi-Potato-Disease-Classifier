# PlankAi — Potato Disease Classifier

Small Flask web app that predicts potato leaf disease from an uploaded image using a Keras model.

## Features
- Upload a leaf image (PNG/JPG/JPEG) via the web UI.
- Model predicts: Early Blight, Late Blight, or Healthy and shows treatment guidance.

## Requirements
- Python 3.8+
- Packages: `flask`, `tensorflow`, `numpy`, `pillow`, `opencv-python`, `werkzeug`

You can install the main packages with:

```bash
python -m pip install flask tensorflow numpy pillow opencv-python werkzeug
```

## Files of interest
- `main.py` — Flask application and prediction logic.
- `potato_model.h5` — Trained Keras model (must be present in project root).
- `templates/fixed_index.html` — Frontend UI used by the app.
- `static/` — CSS and uploaded images (`fixed_style.css`, `style.css`).

## Setup & Run
1. Ensure `potato_model.h5` is in the project root (next to `main.py`).
2. (Optional) Create and activate a virtual environment:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
```

3. Install dependencies (see Requirements).
4. Run the app:

```bash
python main.py
```

5. Open your browser at `http://127.0.0.1:5000/` and upload an image.

## Usage Notes
- Allowed file types: PNG, JPG, JPEG.
- The app saves uploaded images to the `static/` folder.
- If the model fails to load, check console logs; `main.py` logs model load errors.

## Troubleshooting
- If you see errors related to TensorFlow model loading, ensure `potato_model.h5` is a compatible Keras model.
- For OpenCV / Pillow image issues, verify the uploaded image is not corrupted.

## Next Steps (suggested)
- Add a `requirements.txt` file (`pip freeze > requirements.txt`).
- Add `.gitignore` to exclude `venv/` and `__pycache__/` and uploaded images if desired.
- Add Dockerfile or deployment instructions.

## License
Add a license of your choice (e.g., MIT) if you plan to publish the code.
