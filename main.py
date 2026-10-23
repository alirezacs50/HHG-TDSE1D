import numpy as np
import matplotlib.pyplot as plt

##################################################

wavelength_nm: float = 800.0
intensity_wcm2: float = 3e14
a2 = 2.0  
n_cycles: int = 10
x_max: float = 200.0     # Size 
nx: int = 4096
dt: float = 0.05


##################################################
def Atomic_unit_parameetrs(wavelength_nm , intensity_wcm2 , n_cycles): 

    w_0 = 45.5634 / wavelength_nm
    E_0 = np.sqrt(intensity_wcm2 / 3.51e16)
    T = 2 * np.pi/w_0
    T_total = n_cycles * T
    return w_0, E_0, T, T_total
##################################################

def build_grids(x_max , nx , dt , T_total):
    dx = 2 * x_max/nx
    x = np.linspace(-x_max , x_max , nx , endpoint= False)
    k = 2 * np.pi * np.fft.fftfreq( nx , d = dx)
    nt = int(T_total / dt)
    t = np.arange(nt) * dt

    return x , dx , k , t

##################################################

# input laser field 
def laser_field(t, E_0, w_0 , T_total):
    envelope = np.sin(np.pi * t / T_total)**2
    E_t = E_0 * envelope * np.cos(w_0 * t)
    return np.where((t >= 0 ) & (t <= T_total), E_t , 0.0)


##################################################

def potential(x , a2):
    v_x = -1 / np.sqrt(x**2 + a2)

    dv_dx = x / (x**2 + a2)**1.5         # dv/dx

    return v_x , dv_dx
     

##################################################

def ground_state(x, dx, k, V, dtau=0.05, max_steps=20000, tol=1e-10):
    psi = np.exp(-x**2).astype(complex)
    psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx)

    expV = np.exp(-V * dtau / 2)
    expT = np.exp(-(k**2 / 2) * dtau)      # = exp(-k² dτ / 2)
    T_k = k**2 / 2

    E_old = 0.0
    for step in range(max_steps):
        psi = expV * psi
        psi = np.fft.ifft(expT * np.fft.fft(psi))
        psi = expV * psi

        psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx)

        Tpsi = np.fft.ifft(T_k * np.fft.fft(psi))
        E_kin = np.real(np.sum(np.conj(psi) * Tpsi) * dx)
        E_pot = np.sum(V * np.abs(psi)**2) * dx
        E = E_kin + E_pot

        if abs(E - E_old) < tol:
            break
        E_old = E

    return psi, E
##################################################

 # Elimination of the effects of electron wavefunction reflection at the boundary.

def absorber(x, x_max):
    x_abs = 0.8 * x_max
    arg = np.pi * (np.abs(x) - x_abs) / (2 * (x_max - x_abs))
    return np.where(np.abs(x) <= x_abs, 1.0, np.cos(np.clip(arg, 0, np.pi/2)) ** (1/8))
##################################################

def propagate(psi0, x, dx, k, V, dv_dx, E, M, dt):
    nt = len(E)
    psi = psi0.copy()
    acc = np.zeros(nt)

    expT = np.exp(-1j * (k**2 / 2) * dt)

    for n in range(nt):
        E_n = E[n]                                  
        expV = np.exp(-1j * (V + x * E_n) * dt / 2)
        psi = expV * psi                             
        psi = np.fft.ifft(expT * np.fft.fft(psi))    
        psi = expV * psi                             
        psi = M * psi                               

        acc[n] = dipole_acceleration(psi, dv_dx, dx, E_n)

    return acc


##################################################

def dipole_acceleration( psi , dv_dx , dx ,E_n ):
    density = np.abs(psi)**2
    return -np.sum(density *dv_dx)*dx - E_n

##################################################

def hhg_spectrum(dipole_acceleration, t, T_total, dt, w_0):

    w_t = np.sin(np.pi * t / T_total)**2
    s_w = dipole_acceleration * w_t
    nt = len(dipole_acceleration)
    spec = np.abs(np.fft.fft(s_w))**2
    w = 2 * np.pi * np.fft.fftfreq(nt , d= dt)

    # Positive Frequencies
    pos_spec = w > 0
    order = w[pos_spec]/w_0
    return order,spec[pos_spec]
##################################################

def cutoff(E_0 , W_0 , ip ):
    up = E_0**2 / (4 * W_0**2)
    E_cutoff = ip + 3.17 * up
    N_cutoff = E_cutoff / W_0
    return N_cutoff , E_cutoff


##################################################

def plot_results(t, E, acc, order, S, N_cut, save_path=None):
    fig, ax = plt.subplots(3, 1, figsize=(8, 10))

    ax[0].plot(t, E)
    ax[0].set_xlabel("t (a.u.)")
    ax[0].set_ylabel("E(t) (a.u.)")
    ax[0].set_title("Laser field")

    ax[1].plot(t, acc)
    ax[1].set_xlabel("t (a.u.)")
    ax[1].set_ylabel("a(t) (a.u.)")
    ax[1].set_title("Dipole acceleration")

    ax[2].semilogy(order, S)
    ax[2].axvline(N_cut, color="r", linestyle="--", label="cutoff")
    ax[2].set_xlim(0, 60)
    ax[2].set_xlabel("Harmonic order")
    ax[2].set_ylabel("Intensity (arb. units)")
    ax[2].set_title("HHG spectrum")
    ax[2].legend()

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200)
    plt.show()


##################################################

def main():

    # Constant parameters
    wavelength_nm: float = 800.0
    intensity_wcm2: float = 3e14
    a2 = 2.0  
    n_cycles: int = 10
    x_max: float = 200.0     # Size 
    nx: int = 4096
    dt: float = 0.05

    # Convert to atomic unit
    w_0, E_0, T, T_total = Atomic_unit_parameetrs(wavelength_nm, intensity_wcm2, n_cycles)

    # Lattices
    dx = 2 * x_max/nx
    x = np.linspace(-x_max , x_max , nx , endpoint= False)
    k = 2 * np.pi * np.fft.fftfreq( nx , d = dx)
    nt = int(T_total / dt)
    t = np.arange(nt) * dt

    # Absorbet , potential and Field
    v_x , dv_dx = potential(x , a2)
    M = absorber ( x , x_max)
    E = laser_field (t+dt / 2 , E_0 , w_0 , T_total)

    # Ground state 
    psi0 , E_gs = ground_state(x , dx ,k ,v_x)
    ip = abs(E_gs)
    print("Ground state energy : ",E_gs)

    # propagate , dipole acceleration
    acc = propagate(psi0, x, dx, k, v_x, dv_dx, E, M, dt)

    # Cutoff , Spetrum
    order, S_w = hhg_spectrum(acc, t, T_total, dt, w_0)
    N_cut,E_cut = cutoff(E_0 , w_0 , ip)
    print("Cutoff harmonic:", N_cut)

    # Save 
    np.save("hhg_spectrum.npy", np.vstack([order, S_w]))
    plot_results(t, E, acc, order, S_w, N_cut, save_path="hhg_result.png")


if __name__ == "__main__":
    main()