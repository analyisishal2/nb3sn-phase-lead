import numpy as np
import matplotlib.pyplot as plt

# --- Konstanten ---
c = 299792458.0   # Lichtgeschwindigkeit (m/s)
L = 5.0           # Strecke (m)

# --- Pulsparameter ---
f0 = 7.4e12
omega0 = 2 * np.pi * f0

# Zeitachse
t = np.linspace(-5e-12, 5e-12, 4096)
dt = t[1] - t[0]

# Gauß-Puls
tau = 0.5e-12
E_in = np.exp(-t**2 / (2 * tau**2)) * np.cos(omega0 * t)

# --- Fourier-Transformation ---
E_w = np.fft.fft(E_in)
freqs = np.fft.fftfreq(len(t), dt)
omega = 2 * np.pi * freqs

# --- Dispersives Modell ---
# einfacher frequenzabhängiger Brechungsindex
n = 1 + 0.05 * np.exp(-((omega - omega0)**2) / (2 * (0.2 * omega0)**2))

# Wellenzahl
k = n * omega / c

# --- Propagation ---
E_w_out = E_w * np.exp(1j * k * L)
E_out = np.fft.ifft(E_w_out)

# --- Phase-Lead berechnen (Peak-Verschiebung) ---
t_peak_in = t[np.argmax(E_in)]
t_peak_out = t[np.argmax(np.real(E_out))]
phase_lead_ps = (t_peak_in - t_peak_out) * 1e12

# --- Plot ---
plt.figure(figsize=(10,5))
plt.plot(t * 1e12, E_in, label="Input")
plt.plot(t * 1e12, np.real(E_out), label="Output")
plt.xlabel("Zeit (ps)")
plt.ylabel("Amplitude")
plt.title(f"Phase Lead: {phase_lead_ps:.2f} ps")
plt.legend()
plt.grid()
plt.tight_layout()

# Speichern für GitHub
plt.savefig("phase_shift.png", dpi=300)

plt.show()

# --- Ausgabe ---
print("-" * 40)
print("PHASE LEAD ANALYSIS")
print("-" * 40)
print(f"Peak Input:  {t_peak_in*1e12:.3f} ps")
print(f"Peak Output: {t_peak_out*1e12:.3f} ps")
print(f"Phase Lead:  {phase_lead_ps:.2f} ps")
print("-" * 40)
