from __future__ import print_function
import argparse
from collections import defaultdict
import os
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import random
from utils import save_vars
from metrics import compute_sparsity
from models.vae_rotated_mnist import flattened_rotMNIST

# Définir une graine fixe pour PyTorch, Numpy et random
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Forcer le déterminisme dans PyTorch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.benchmark = True

# Parser pour les arguments
parser = argparse.ArgumentParser(description='Analyse des résultats GDVAE')
parser.add_argument('--save-dir', type=str, metavar='N', help='Répertoire de sauvegarde des résultats')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Désactiver CUDA')
parser.add_argument('--disentanglement', action='store_true', default=False, help='Calculer le metric de désentanglement')
parser.add_argument('--sparsity', action='store_true', default=False, help='Calculer la métrique de parcimonie')
parser.add_argument('--logp', action='store_true', default=False, help='Estimation de la vraisemblance marginale')
parser.add_argument('--iwae-samples', type=int, default=1000, help='Nombre d’échantillons pour IWAE')
cmds = parser.parse_args()

# Configuration du chemin de sauvegarde
runPath = os.path.join("/home/epic_joliot/workdir/disentangling-disentanglement/experiments", cmds.save_dir)
if not os.path.exists(runPath):
    os.makedirs(runPath)

# Chargement des arguments
args_path = os.path.join(runPath, "args.rar")
args = torch.load(args_path)
print(f"[DEBUG] args.latent_dim : {args.latent_dim}")

# Configuration CUDA
args.cuda = not cmds.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

# Chargement du modèle
from models.vae_rotated_mnist import VAE_RotatedMNIST
model = VAE_RotatedMNIST(args)
if args.cuda:
    model.cuda()

state_dict_path = os.path.join(runPath, "model.rar")
state_dict = torch.load(state_dict_path)
model.load_state_dict(state_dict)

# Chargement des loaders
_, test_loader = flattened_rotMNIST(
    num_tasks=5, per_task_rotation=45, batch_size=args.batch_size,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(), 
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ]),
    root='./data'
)


# Fonction pour visualiser les magnitudes latentes moyennes des classes 0, 1 et 2
def plot_continuous_latent_magnitude(data, path):
    """
    Affiche les moyennes des magnitudes latentes pour 3 classes sous forme de barres continues.
    Args:
        data (ndarray): Tableau de taille (3, latent_dim), contenant les moyennes pour chaque classe.
        path (str): Chemin pour sauvegarder le graphique.
    """
    x = np.arange(data.shape[1])  # Dimensions latentes (0 à latent_dim)
    width = 0.25  # Largeur des barres pour chaque classe
    fig, ax = plt.subplots(figsize=(12, 6))

    # Afficher les barres pour chaque classe
    ax.bar(x - width, data[0], width=width, color='red', label="Class 0")
    ax.bar(x, data[1], width=width, color='blue', label="Class 1")
    ax.bar(x + width, data[2], width=width, color='green', label="Class 2")

    # Configuration des axes
    ax.set_xlabel("Latent Dimension")
    ax.set_ylabel("Avg Magnitude")
    ax.set_title("Average Latent Magnitude per Dimension (Classes 0, 1, 2)")
    ax.set_xticks(np.arange(0, data.shape[1], 5))  # Ticks de l'axe X toutes les 5 dimensions
    ax.set_xticklabels(np.arange(0, data.shape[1], 5))
    ax.legend()

    plt.tight_layout()
    plt.savefig(path + "_continuous_bar_chart.png")
    print(f"Graphique sauvegardé dans : {path}_continuous_bar_chart.png")
    plt.close()

def plot_multiple_sparsity_curves(alpha_values, results, path):
    """
    Fonction pour tracer plusieurs courbes Beta et Gamma en fonction des Alpha.
    Beta sera en rouge, Gamma en bleu, avec des styles distincts pour chaque configuration.

    Args:
        alpha_values (list): Liste des valeurs d'Alpha.
        results (dict): Résultats contenant les valeurs de Beta et Gamma pour chaque configuration.
        path (str): Chemin pour sauvegarder le graphique.
    """
    plt.figure(figsize=(10, 6))
    line_styles = ['-', '--', '-.', ':']
    colors = ['red', 'blue']  # Rouge pour Beta, Bleu pour Gamma

    for i, ((beta, gamma), sparsity_values) in enumerate(results.items()):
        label = f"Beta={beta}, Gamma={gamma}"
        color = colors[i % 2]  # Rouge pour Beta, Bleu pour Gamma
        style = line_styles[i // 2 % len(line_styles)]

        # Tracer les courbes en fonction des Alpha
        plt.plot(alpha_values, sparsity_values, label=label, color=color, linestyle=style)

    plt.xlabel("Alpha")
    plt.ylabel("Avg Normalised Sparsity")
    plt.title("Beta and Gamma vs Alpha (Multiple Configurations)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    #print(f"[INFO] Graphique sauvegardé dans : {path}")
    plt.close()


@torch.no_grad()
def compute_sparsity_vs_alpha(alpha_values, beta, gamma):
    sparsity_means = []
    sparsity_stds = []

    for alpha in alpha_values:
        #print(f"[INFO] Calcul pour Alpha={alpha}, Beta={beta}, Gamma={gamma}")
        zs_mean = torch.zeros(len(test_loader.dataset), args.latent_dim, device=device)

        for i, (data, _, _) in enumerate(test_loader):
            data = data.to(device)
            z_mu, _, _ = model(data)
            zs_mean[i * data.size(0): (i + 1) * data.size(0), :] = z_mu

        # Calcul de la sparsity
        sparsity = compute_sparsity(zs_mean, norm=True)
        sparsity_means.append(sparsity.item())
        sparsity_stds.append(torch.std(sparsity).item())

    return sparsity_means, sparsity_stds

@torch.no_grad()
def compute_beta_gamma(alpha_values, beta, gamma):
    sparsity_results = []

    for alpha in alpha_values:
        #print(f"[INFO] Calcul pour Alpha={alpha}, Beta={beta}, Gamma={gamma}")
        zs_mean = torch.zeros(len(test_loader.dataset), args.latent_dim, device=device)

        for i, (data, _, _) in enumerate(test_loader):
            data = data.to(device)
            z_mu, _, _ = model(data)
            zs_mean[i * data.size(0): (i + 1) * data.size(0), :] = z_mu

        # Calcul de la sparsity normalisée
        sparsity = compute_sparsity(zs_mean, norm=True)
        sparsity_results.append(sparsity.item())
    
    return sparsity_results


# Exécution principale
if __name__ == "__main__":
    # Valeurs d'Alpha pour l'analyse
    alpha_values = [0, 200, 400, 600, 800, 1000]
    
    # Configurations spécifiques de Beta et Gamma
    beta_gamma_configs = [
        (0.1, 0.0),  # 1er: gamma = 0, beta = 0.1
        (0.1, 0.8),  # 2ème: gamma = 0.8, beta = 0.1
        (1.0, 0.0),  # 3ème: gamma = 0, beta = 1
        (1.0, 0.8),  # 4ème: gamma = 0.8, beta = 1
        (5.0, 0.0),  # 5ème: gamma = 0, beta = 5
        (5.0, 0.8),  # 6ème: gamma = 0.8, beta = 5
    ]
    
    results = {}  # Pour stocker les résultats Beta et Gamma

    # Calculer Beta et Gamma pour chaque configuration
    for beta, gamma in beta_gamma_configs:
        #print(f"[INFO] Début du calcul pour Beta={beta} et Gamma={gamma}")
        results[(beta, gamma)] = compute_beta_gamma(alpha_values, beta, gamma)

    # Tracer les courbes Beta (rouge) et Gamma (bleu) avec différentes lignes
    graph_path = os.path.join(runPath, "beta_gamma_vs_alpha.png")
    plot_multiple_sparsity_curves(alpha_values, results, graph_path)

    # Visualisation des magnitudes latentes moyennes
    zs_mean = torch.zeros(len(test_loader.dataset), args.latent_dim, device=device)
    ys = torch.zeros(len(test_loader.dataset), dtype=torch.long, device=device)

    # Calculer les valeurs latentes moyennes
    for i, (data, labels, _) in enumerate(test_loader):
        data = data.to(device)
        z_mu, _, _ = model(data)
        zs_mean[i * data.size(0): (i+1) * data.size(0), :] = z_mu
        ys[i * data.size(0): (i+1) * data.size(0)] = labels.to(device)

    # Moyennes des magnitudes pour les classes 0, 1 et 2
    class_means = []
    for class_label in [0, 1, 2]:
        class_data = zs_mean[ys == class_label]
        class_mean = class_data.abs().mean(dim=0).detach().cpu().numpy()
        class_means.append(class_mean)

    class_means = np.stack(class_means)
    plot_continuous_latent_magnitude(class_means, path=os.path.join(runPath, 'plot_sparsity'))

    print("[INFO] Analyse terminée avec succès.")