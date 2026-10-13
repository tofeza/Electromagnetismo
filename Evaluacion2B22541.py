import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons
from matplotlib.animation import FuncAnimation

# Configuración inicial de la onda y dimensiones
E0_i = 1.0  
omega = 2 * np.pi  
c = 1.0  
d = 5.0  

# Diccionario con los índices de refracción de los materiales
materiales = {
    'Vacío (1.00)': 1.0,
    'Agua (1.33)': 1.33,
    'Vidrio (1.50)': 1.5,
    'Diamante (2.42)': 2.42
}
nombres = list(materiales.keys())

# Configuración de la figura y los ejes
fig, ax = plt.subplots(figsize=(12, 7))
plt.subplots_adjust(bottom=0.35)

# Definición del eje Z para los tres medios
z1 = np.linspace(-5, 0, 400)
z2 = np.linspace(0, d, 400)
z3 = np.linspace(d, d + 5, 400)

# Inicialización de las líneas para las ondas
line_i, = ax.plot(z1, np.zeros_like(z1), 'b', label='Incidente ($E_i$)', alpha=0.3)
line_r, = ax.plot(z1, np.zeros_like(z1), 'r', label='Reflejada ($E_r$)', alpha=0.3)
line_tot1, = ax.plot(z1, np.zeros_like(z1), 'k--', label='Total Medio 1')

line_A, = ax.plot(z2, np.zeros_like(z2), 'orange', label='Avanza Med 2 (A)', alpha=0.5)
line_B, = ax.plot(z2, np.zeros_like(z2), 'cyan', label='Refleja Med 2 (B)', alpha=0.5)
line_tot2, = ax.plot(z2, np.zeros_like(z2), 'purple', label='Total Medio 2', linestyle='--')

line_t3, = ax.plot(z3, np.zeros_like(z3), 'g', label='Transmitida ($E_{t3}$)')

# Marcadores de las interfaces
ax.axvline(0, color='black', linewidth=2, linestyle=':') 
ax.axvline(d, color='black', linewidth=2, linestyle=':')

ax.set_xlim(-5, d + 5)
ax.set_ylim(-2.5, 2.5)
ax.set_xlabel('Posición $z$')
ax.set_ylabel('Campo Eléctrico $E(z,t)$')
ax.set_title('Reflexión y Transmisión de Onda EM (Interferencia en Lámina)')
ax.legend(loc='upper right', fontsize='x-small', ncol=2)

# Caja de texto para mostrar las amplitudes y el balance de energía
text_energy = ax.text(-4.8, 1.3, '', fontsize=9, bbox=dict(facecolor='white', alpha=0.8))

# Creación de los ejes para los RadioButtons
axcolor = 'lightgoldenrodyellow'
ax_n1 = plt.axes([0.05, 0.05, 0.25, 0.18], facecolor=axcolor)
ax_n2 = plt.axes([0.38, 0.05, 0.25, 0.18], facecolor=axcolor)
ax_n3 = plt.axes([0.71, 0.05, 0.25, 0.18], facecolor=axcolor)

ax_n1.set_title('Material Medio 1')
ax_n2.set_title('Material Medio 2')
ax_n3.set_title('Material Medio 3')

# Inicialización de los RadioButtons
radio1 = RadioButtons(ax_n1, nombres, active=0) 
radio2 = RadioButtons(ax_n2, nombres, active=2) 
radio3 = RadioButtons(ax_n3, nombres, active=0) 

def update_wave(t_val):
    n1 = materiales[radio1.value_selected]
    n2 = materiales[radio2.value_selected]
    n3 = materiales[radio3.value_selected]
    
    k1 = n1 * omega / c
    k2 = n2 * omega / c
    k3 = n3 * omega / c
    
    r12 = (n1 - n2) / (n1 + n2)
    t12 = 2 * n1 / (n1 + n2)
    r23 = (n2 - n3) / (n2 + n3)
    t23 = 2 * n2 / (n2 + n3)
    
    delta = k2 * d
    
    denom = 1 + r12 * r23 * np.exp(2j * delta)
    r_tot = (r12 + r23 * np.exp(2j * delta)) / denom
    t_tot = (t12 * t23 * np.exp(1j * delta)) / denom
    
    E0_r = r_tot * E0_i
    E0_t3 = t_tot * E0_i
    
    A = (t12 / denom) * E0_i
    B = (t12 * r23 * np.exp(2j * delta) / denom) * E0_i
    
    # Unificación de las expresiones de campo a notación compleja pura
    E_i = np.real(E0_i * np.exp(1j * (k1 * z1 - omega * t_val)))
    E_r = np.real(E0_r * np.exp(1j * (-k1 * z1 - omega * t_val)))
    E_tot1 = E_i + E_r 
    
    E_A = np.real(A * np.exp(1j * (k2 * z2 - omega * t_val)))
    E_B = np.real(B * np.exp(1j * (-k2 * z2 - omega * t_val)))
    E_tot2 = E_A + E_B
    
    E_t3 = np.real(E0_t3 * np.exp(1j * (k3 * (z3 - d) - omega * t_val)))
    
    line_i.set_ydata(E_i)
    line_r.set_ydata(E_r)
    line_tot1.set_ydata(E_tot1)
    line_A.set_ydata(E_A)
    line_B.set_ydata(E_B)
    line_tot2.set_ydata(E_tot2)
    line_t3.set_ydata(E_t3)
    
    R = np.abs(r_tot)**2
    T_trans = (n3 / n1) * np.abs(t_tot)**2
    
    text_energy.set_text(f'Amplitudes Totales (Modulo):\n'
                         f'$|E_{{0i}}|$ = {E0_i:.2f} | $|E_{{0r}}|$ = {np.abs(E0_r):.2f} | $|E_{{0t3}}|$ = {np.abs(E0_t3):.2f}\n\n'
                         f'Balance Energético:\n'
                         f'Reflectancia (R) = {R:.6f}\n'
                         f'Transmitancia (T) = {T_trans:.6f}\n'
                         f'Conservación (R + T) = {R + T_trans:.6f}')
                         
    return line_i, line_r, line_tot1, line_A, line_B, line_tot2, line_t3, text_energy

def animate(i):
    t_val = i * 0.02
    return update_wave(t_val)

ani = FuncAnimation(fig, animate, frames=200, interval=20, blit=True)

plt.show()