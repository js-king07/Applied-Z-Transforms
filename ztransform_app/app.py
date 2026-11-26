from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

app = Flask(__name__)
BASE = os.path.dirname(os.path.abspath(__file__))
PLOT = os.path.join(BASE, 'static', 'plot.png')


def coeff(text):
    values = np.array([float(x.strip()) for x in text.split(',') if x.strip()], dtype=float)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError('Please enter valid finite coefficients.')
    return values


def normalize_coefficients(b, a):
    b = np.asarray(b, dtype=float).flatten()
    a = np.asarray(a, dtype=float).flatten()
    if not len(a) or a[0] == 0:
        raise ValueError('The first denominator coefficient must be non-zero.')
    return b / a[0], a / a[0]


def save_plot():
    os.makedirs(os.path.dirname(PLOT), exist_ok=True)
    plt.tight_layout()
    plt.savefig(PLOT, dpi=140)
    plt.close()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/module1', methods=['GET', 'POST'])
def m1():
    result = None
    if request.method == 'POST':
        try:
            b, a = normalize_coefficients(coeff(request.form['b']), coeff(request.form['a']))
            zeros = np.roots(b) if len(b) > 1 else np.array([])
            poles = np.roots(a) if len(a) > 1 else np.array([])
            stable = bool(np.all(np.abs(poles) < 1))

            fig, ax = plt.subplots(figsize=(6, 5))
            ang = np.linspace(0, 2*np.pi, 500)
            ax.plot(np.cos(ang), np.sin(ang), '--', label='Unit circle')
            ax.axhline(0, color='gray'); ax.axvline(0, color='gray')
            if zeros.size:
                ax.scatter(zeros.real, zeros.imag, marker='o', facecolors='none', edgecolors='blue', s=90, label='Zeros')
            if poles.size:
                ax.scatter(poles.real, poles.imag, marker='x', color='red', s=90, label='Poles')
            ax.set_xlabel('Real'); ax.set_ylabel('Imaginary'); ax.set_title('Pole-Zero Map')
            ax.grid(True); ax.axis('equal'); ax.legend(); save_plot()
            result = {'zeros': np.round(zeros, 4).tolist(), 'poles': np.round(poles, 4).tolist(), 'stable': stable, 'status': 'Stable' if stable else 'Unstable'}
        except Exception as e:
            result = {'error': str(e)}
    return render_template('module.html', module=1, result=result, title='Pole-Zero Analysis', fields=[('b', 'Numerator coefficients', '0.2,0.2'), ('a', 'Denominator coefficients', '1,-0.6')], description='Calculate zeros and poles and check discrete-time stability against the unit circle.')


@app.route('/module2', methods=['GET', 'POST'])
def m2():
    result = None
    if request.method == 'POST':
        try:
            num = coeff(request.form['num']).flatten()
            den = coeff(request.form['den']).flatten()
            T = float(request.form['T'])
            if not np.isfinite(T) or T <= 0:
                raise ValueError('Sampling period must be a finite positive number.')
            if den[0] == 0:
                raise ValueError('The first denominator coefficient must be non-zero.')

            system = signal.TransferFunction(num, den)
            bd, ad, _ = signal.cont2discrete((num, den), T, method='zoh')
            bd = np.asarray(bd, dtype=float).flatten()
            ad = np.asarray(ad, dtype=float).flatten()

            # Let SciPy create its continuous step-response time vector (avoids the previous T=t error).
            tout, yc = signal.step(system)
            tout = np.asarray(tout, dtype=float).flatten()
            yc = np.asarray(yc, dtype=float).flatten()

            n = np.arange(0, int(5 / T) + 1)
            discrete_system = signal.dlti(bd, ad, dt=T)
            _, yd = signal.dlsim(discrete_system, np.ones(len(n)))
            yd = np.asarray(yd, dtype=float).flatten()

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(tout, yc, label='Continuous response')
            ax.step(n * T, yd, where='post', label='Discrete ZOH')
            ax.set_xlabel('Time (s)'); ax.set_ylabel('Output')
            ax.set_title('Continuous vs Discrete Response'); ax.grid(True); ax.legend(); save_plot()
            result = {'b': np.round(bd, 6).tolist(), 'a': np.round(ad, 6).tolist(), 'T': T, 'fs': 1 / T}
        except Exception as e:
            result = {'error': str(e)}
    return render_template('module.html', module=2, result=result, title='Continuous Plant Discretization', fields=[('num', 'Continuous numerator', '1'), ('den', 'Continuous denominator', '1,1'), ('T', 'Sampling period (seconds)', '0.2')], description='Convert a continuous transfer function to discrete form using Zero-Order Hold (ZOH), then compare responses.')


@app.route('/module3', methods=['GET', 'POST'])
def m3():
    result = None
    if request.method == 'POST':
        try:
            b, a = normalize_coefficients(coeff(request.form['b']), coeff(request.form['a']))
            T = float(request.form['T']); kp = float(request.form['kp']); ki = float(request.form['ki']); kd = float(request.form['kd']); N = int(request.form['N'])
            if T <= 0 or not np.isfinite(T):
                raise ValueError('Sampling period must be positive and finite.')
            if N < 2 or N > 5000:
                raise ValueError('Simulation samples must be between 2 and 5000.')
            if not all(np.isfinite(v) for v in (kp, ki, kd)):
                raise ValueError('PID gains must be finite.')

            n = np.arange(N); r = np.ones(N); y = np.zeros(N); u = np.zeros(N); e = np.zeros(N)
            integral = 0.0; previous_error = 0.0
            for i in range(N):
                e[i] = r[i] - (y[i-1] if i > 0 else 0.0)
                integral += e[i] * T
                derivative = (e[i] - previous_error) / T if i > 0 else 0.0
                u[i] = kp * e[i] + ki * integral + kd * derivative
                previous_error = e[i]
                y[i] = sum(b[j] * u[i-j] for j in range(len(b)) if i-j >= 0) - sum(a[j] * y[i-j] for j in range(1, len(a)) if i-j >= 0)

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(n * T, r, label='Reference'); ax.plot(n * T, y, label='Controlled output')
            ax.set_xlabel('Time (s)'); ax.set_ylabel('Output'); ax.set_title('Discrete PID Closed-loop Simulation'); ax.grid(True); ax.legend(); save_plot()
            result = {'kp': kp, 'ki': ki, 'kd': kd, 'final': round(float(y[-1]), 4), 'samples': N, 'rise_note': 'Simulation uses the supplied gains and a discrete plant recurrence.'}
        except Exception as e:
            result = {'error': str(e)}
    return render_template('module.html', module=3, result=result, title='Digital PID Controller', fields=[('b', 'Plant numerator', '0.2,0.2'), ('a', 'Plant denominator', '1,-0.6'), ('T', 'Sampling period', '0.1'), ('kp', 'Kp', '1.2'), ('ki', 'Ki', '0.4'), ('kd', 'Kd', '0.05'), ('N', 'Simulation samples', '60')], description='Run a sampled PID control simulation with adjustable gains and plant coefficients.')


@app.route('/dashboard')
def dashboard():
    # Illustrative dashboard response (not a quantization comparison).
    n = np.arange(12); y = 0.2 * (0.6 ** n)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(n, y, marker='o', label='Sample response')
    ax.grid(True); ax.legend(); ax.set_title('Dashboard: illustrative response')
    ax.set_xlabel('Sample'); ax.set_ylabel('Output'); save_plot()
    return render_template('dashboard.html')


@app.route('/resetplot')
def resetplot():
    return redirect(url_for('home'))


@app.errorhandler(404)
def not_found(error):
    return 'Page not found', 404


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
