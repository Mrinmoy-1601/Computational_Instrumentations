import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

# Set seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_data(filename='FEM4_data.csv'):
    df = pd.read_csv(filename)
    X = torch.tensor(df['X'].values, dtype=torch.float32, device=device)
    Y = torch.tensor(df['Y'].values, dtype=torch.float32, device=device)
    return X, Y

def create_triangulation(X, Y):
    triang = tri.Triangulation(X.cpu(), Y.cpu())
    plt.figure(figsize=(8, 6))
    plt.triplot(triang, lw=0.8)
    plt.title('Triangular Mesh')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig('Triangular_Mesh.png')

def calculate_D_e(E, nu, device):
    D_e = E / ((1 + nu) * (1 - 2 * nu)) * torch.tensor([
        [1 - nu, nu, 0],
        [nu, 1 - nu, 0],
        [0, 0, (1 - 2 * nu) / 2]
    ], device=device)
    return D_e

def calculate_principal_stresses(sigma_xx, sigma_yy, sigma_xy):
    sigma_avg = (sigma_xx + sigma_yy) / 2
    R = torch.sqrt(((sigma_xx - sigma_yy) / 2) ** 2 + sigma_xy ** 2)
    sigma_1 = sigma_avg + R  # Maximum principal stress
    sigma_3 = sigma_avg - R  # Minimum principal stress
    return sigma_1, sigma_3

def mohr_coulomb_yield_function(sigma_1, sigma_3, c, phi):
    sin_phi = torch.sin(torch.deg2rad(phi))
    cos_phi = torch.cos(torch.deg2rad(phi))
    return (sigma_1 - sigma_3) - (sigma_1 + sigma_3) * sin_phi - 2 * c * cos_phi

def calculate_elastic_plastic_status(sigma_1, sigma_3, c, phi):
    sin_phi = torch.sin(torch.deg2rad(phi))
    cos_phi = torch.cos(torch.deg2rad(phi))
    tau_mob = (sigma_1 - sigma_3) / 2
    tau_max = c * cos_phi - ((sigma_1 + sigma_3) / 2) * sin_phi
    tau_rel = torch.abs(tau_mob / tau_max)
    tolerance = 0.99
    status = torch.where(tau_rel > tolerance, torch.ones_like(tau_rel), torch.zeros_like(tau_rel))
    return status, tau_rel, tau_max, tau_mob

def compute_stresses_strains(X, Y, E, nu, c, phi):
    D_e = calculate_D_e(E, nu, device)
    U_x = torch.rand_like(X) * 0.01  # Placeholder
    U_y = torch.rand_like(X) * 0.01  # Placeholder
    V_x = torch.rand_like(X) * 0.01  # Placeholder
    V_y = torch.rand_like(X) * 0.01  # Placeholder
    epsilon_xx = U_x
    epsilon_yy = V_y
    epsilon_xy = 0.5 * (U_y + V_x)
    strain_vector = torch.stack([epsilon_xx, epsilon_yy, epsilon_xy], dim=-1)
    sigma_vector = torch.einsum('ij,bj->bi', D_e, strain_vector)
    sigma_xx, sigma_yy, sigma_xy = sigma_vector.unbind(dim=-1)
    sigma_1, sigma_3 = calculate_principal_stresses(sigma_xx, sigma_yy, sigma_xy)
    failure_criterion = mohr_coulomb_yield_function(sigma_1, sigma_3, c, phi)
    status, tau_rel, tau_max, tau_mob = calculate_elastic_plastic_status(sigma_1, sigma_3, c, phi)
    return sigma_xx, sigma_yy, sigma_xy, sigma_1, sigma_3, epsilon_xx, epsilon_yy, epsilon_xy, tau_rel, tau_max, tau_mob, status

def save_results(X, Y, sigma_xx, sigma_yy, sigma_xy, sigma_1, sigma_3, epsilon_xx, epsilon_yy, epsilon_xy, tau_rel, tau_max, tau_mob, status, filename='results.csv'):
    df_out = pd.DataFrame({
        'X': X.cpu().numpy(),
        'Y': Y.cpu().numpy(),
        'sigma_xx': sigma_xx.cpu().numpy(),
        'sigma_yy': sigma_yy.cpu().numpy(),
        'sigma_xy': sigma_xy.cpu().numpy(),
        'sigma_1': sigma_1.cpu().numpy(),
        'sigma_3': sigma_3.cpu().numpy(),
        'epsilon_xx': epsilon_xx.cpu().numpy(),
        'epsilon_yy': epsilon_yy.cpu().numpy(),
        'epsilon_xy': epsilon_xy.cpu().numpy(),
        'tau_mob': tau_mob.cpu().numpy(),
        'tau_max': tau_max.cpu().numpy(),
        'tau_rel': tau_rel.cpu().numpy(),
        'status': status.cpu().numpy()
    })
    df_out.to_csv(filename, index=False)
    return df_out

def plot_results(df_out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # First subplot with explicit label
    axes[0].scatter(df_out['X'], df_out['Y'], label='Original Points', marker='o', s=10)
    axes[0].set_title('Original X and Y')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].legend()
    
    # Second subplot with a scatter plot and colorbar
    scatter = axes[1].scatter(df_out['X'], df_out['Y'], 
                            c=df_out['tau_rel'], 
                            cmap='coolwarm', 
                            edgecolors='k',
                            label='Relative Shear Strength')
    axes[1].set_title('Calculated X and Y')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    # Add colorbar instead of legend for the second plot
    plt.colorbar(scatter, ax=axes[1], label='Relative Shear Strength')
    
    plt.tight_layout()
    plt.savefig('Comparison.png')
    
    # Separate figure for detailed view
    plt.figure(figsize=(10, 6))
    scatter_detailed = plt.scatter(df_out['X'], df_out['Y'], 
                                 c=df_out['tau_rel'], 
                                 cmap='coolwarm', 
                                 edgecolors='k')
    plt.colorbar(scatter_detailed, label="Relative Shear Strength")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("FEM Analysis - Shear Strength Distribution")
    plt.savefig('FEM_Results.png')

def main():
    X, Y = load_data()
    create_triangulation(X, Y)
    E = torch.tensor(5.0, device=device)
    nu = torch.tensor(0.3, device=device)
    c = torch.tensor(3.0, device=device)
    phi = torch.tensor(13.0, device=device)
    results = compute_stresses_strains(X, Y, E, nu, c, phi)
    df_out = save_results(X, Y, *results)
    plot_results(df_out)

if __name__ == "__main__":
    main()