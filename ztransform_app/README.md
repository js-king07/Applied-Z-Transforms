# Z-Transform Control Studio

Interactive Flask application for digital control-system analysis using Z-transform methods.

## Modules
- Module 1: Pole-zero analysis and discrete stability assessment
- Module 2: Continuous-to-discrete conversion using Zero-Order Hold (ZOH), with response graph comparison
- Module 3: Discrete PID simulation with adjustable gains
- Dashboard: Overview and illustrative response graph

## Local run
1. Install Python 3.10+.
2. In the project folder, create/activate a virtual environment (optional but recommended).
3. Install dependencies: `pip install -r requirements.txt`
4. Run: `python app.py`
5. Open `http://127.0.0.1:5000`.

## Render
Build command: `pip install -r requirements.txt`
Start command: `gunicorn app:app`
