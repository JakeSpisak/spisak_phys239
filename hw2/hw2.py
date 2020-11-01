import numpy as np
import matplotlib.pyplot as plt

#Constants throughout the notebook
D = 100*3.086*10**18 #cm
n = 1 #cm^-3

#PROBLEM 1

print("Beginning problem #1")
cd = D*n
print(r"Column density is {} cm^-2".format(cd))

def compute_cross(tau, D, n):
    """Given a desired optical depth 'tau' and a depth 'D' (cm) and density 'n' (cm^-3),
    compute the cross section"""
    return float(tau)/(n*D)

sig1 = compute_cross(0.001, D, n)
sig2 = compute_cross(1, D, n)
sig3 = compute_cross(1000, D, n)

print(f"Cross section if tau=0.001 is {sig1} cm^2")
print(f"Cross section if tau=1 is {sig2} cm^2")
print(f"Cross section if tau=1000 is {sig3} cm^2")


#PROBLEM 2
print("Beginning problem #2")
def compute_I(I0, Sv, sigma, D, n):
    """Given the initial intensity, source function, cross section, distance, and number density (all cgs units),
    compute the final intensity using 1000 steps"""
    N = 10**4
    ds = D/N
    I_list = np.zeros(N)
    I_list[0] = I0

    #For each step, solve the radiative tranfer equation using values from the last step
    for j in range(N-1):
        dI = n*sigma*ds*(Sv-I_list[j])
        I_list[j+1] = I_list[j] + dI

    return I_list[-1]

#An example
print("Example calculation for I(0)=1, Sv=0.5, tau=0.5")  
I0 = 1
Sv = 0.5
sigma = compute_cross(0.5, D, n)

I_fin = compute_I(I0, Sv, sigma, D, n)
print(f"I(D)={I_fin}")


#PROBLEM 3
print("Beginning problem #3")

def cross_freq(freqs, amp, const=0):
    """ Given an input list of frequencies, a max amplitude, and a constant floor, return a 
    guassian profile of the cross section. 
    """
    mean = 150 #150 Ghz
    std = 10 #10 Ghz standard deviation
    cross = []
    for f in freqs:
        cross.append((amp-const)*np.exp(-0.5*(f-mean)**2/std**2) +const)
    return cross

freqs = np.arange(30, 300, 1)
cross1 = cross_freq(freqs, sig1)
cross2 = cross_freq(freqs, sig2)
cross3 = cross_freq(freqs, sig3)

print(f"Saving tau={sig1} cross section figure to ./figures")
plt.figure(figsize=(10,8))
plt.plot(freqs, cross1)
plt.xlabel("Frequency (Ghz)")
plt.ylabel(r"$\sigma_{\nu}$ ($cm^2$)")
plt.title("Cross section when tau = 0.001")
plt.savefig("figures/sig_nu_tau1.png")

print(f"Saving tau={sig2} cross section figure to ./figures")
plt.figure(figsize=(10,8))
plt.plot(freqs, cross2)
plt.xlabel("Frequency (Ghz)")
plt.ylabel(r"$\sigma_{\nu}$ ($cm^2$)")
plt.title("Cross section when tau = 1")
plt.savefig("figures/sig_nu_tau2.png")

print(f"Saving tau={sig3} cross section figure to ./figures")
plt.figure(figsize=(10,8))
plt.plot(freqs, cross3)
plt.xlabel("Frequency (Ghz)")
plt.ylabel(r"$\sigma_{\nu}$ ($cm^2$)")
plt.title("Cross section when tau = 1000")
plt.savefig("figures/sig_nu_tau3.png")

#PROBLEM 4

print("Beginning problem 4")
def plot_I(freqs, I0, tau_max, tau_min, Sv, D, n, label):
    """ Generate and plot I(D). Inputs: I(0), the source function, the minimum and maximum optical depth
    used to generate the guassian cross section, the depth, the density, and the figure label.  
    """
    amp = compute_cross(tau_max, D, n)
    const = compute_cross(tau_min, D, n)
    cross_list = cross_freq(freqs, amp, const)
    I_list = []
    for c in cross_list:
        I_list.append(compute_I(I0, Sv, c, D, n))
    
    print(f"Generating Fig. 4{label} using I(0)={I0}, S={Sv}, tau_max={tau_max}, tau_min={tau_min}")
    plt.figure(figsize=(8,8))
    plt.title(f"Problem 4.{label}: tau_max={tau_max}, tau_min={tau_min}")
    plt.xlabel("GHz")
    plt.ylabel(r"I(D) ($W cm^{-2} sr^{-1} Hz^{-1}$)")
    plt.plot(freqs, I0*np.ones(len(freqs)), 'r--', label='I0')
    plt.plot(freqs, Sv*np.ones(len(freqs)), 'g--', label='Sv')
    plt.plot(freqs, I_list, 'b', label="I(D)")
    plt.legend()
    plt.savefig(f"figures/4{label}")    
    
freqs = np.arange(100, 200, 2)
I0_list = [0, 1, 0.5, 1, 0.5, 1]
Sv_list = [1, 0.5, 1, 0.5, 1, 0.5]
tau_max_list = [0.5, 0.5, 0.5, 10, 5, 5]
tau_min_list = [0, 0, 0, 5, 0.5, 0.5]
label_list = ['a', 'b', 'c', 'd', 'e', 'f']
for i in range(len(I0_list)):
    I0 = I0_list[i]
    Sv = Sv_list[i]
    tau_max = tau_max_list[i]
    tau_min = tau_min_list[i]
    label = label_list[i]
    plot_I(freqs, I0, tau_max, tau_min, Sv, D, n, label)
              
  