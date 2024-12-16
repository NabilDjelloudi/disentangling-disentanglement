import sys
import os
import datetime
import json
import subprocess
import argparse
from tempfile import mkdtemp
from collections import defaultdict
from torchvision.utils import save_image
from vis import visualize_latent_space

import torch
import numpy as np
from utils import Logger, Timer, save_model, save_vars, probe_infnan
from distributions.sparse import Sparse
import objectives
import regularisers
from models.vae_rotated_mnist import VAE_RotatedMNIST, flattened_rotMNIST  # Import spécifique pour RotatedMNIST

# Initialisation
runId = datetime.datetime.now().isoformat().replace(':', '_')
torch.backends.cudnn.benchmark = True

# Argument parser
parser = argparse.ArgumentParser(description='Disentangling Disentanglement in VAEs',
                                 formatter_class=argparse.RawTextHelpFormatter)
# General
parser.add_argument('--model', type=str, default='rotated_mnist', metavar='M', help='model name (default: rotated_mnist)')
parser.add_argument('--name', type=str, default='.', help='experiment name (default: None)')
parser.add_argument('--save-freq', type=int, default=10, help='Frequency to save model and losses (default: 10)')
parser.add_argument('--skip-test', action='store_true', default=False, help='skip test dataset computations')

# Neural nets
parser.add_argument('--num-hidden-layers', type=int, default=1, metavar='H', help='number of hidden layers in enc and dec (default: 1)')
parser.add_argument('--hidden-dim', type=int, default=100, help='number of units in hidden layers in enc and dec (default: 100)')
parser.add_argument('--fBase', type=int, default=32, help='parameter for DCGAN networks')

# Optimisation
parser.add_argument('--epochs', type=int, default=30, metavar='E', help='number of epochs to train (default: 30)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='batch size for data (default: 64)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for optimiser (default: 1e-4)')

# Objective
parser.add_argument('--latent-dim', type=int, default=10, metavar='L', help='latent dimensionality (default: 10)')
parser.add_argument('--alpha', type=float, default=1.0, help='weight for regularization term (default: 1.0)')
parser.add_argument('--beta', type=float, default=1.0, help='weight for KL divergence term (default: 1.0)')
parser.add_argument('--obj', type=str, default='decomp', help='objective function to use (default: decomp)')

# - Algorithm
parser.add_argument('--beta1', type=float, default=0.9, help='first parameter of Adam (default: 0.9)')
parser.add_argument('--beta2', type=float, default=0.999, help='second parameter of Adam (default: 0.900)')

# Prior / posterior
parser.add_argument('--prior', type=str, default='Normal', help='prior distribution (default: Normal)')
parser.add_argument('--posterior', type=str, default='Normal', help='posterior distribution (default: Normal)')
parser.add_argument('--gamma', type=float, default=0.8, help='weight of the spike component of the sparse prior')
parser.add_argument('--df', type=float, default=2., help='degree of freedom of the Student-t')

# - weights
parser.add_argument('--prior-variance', type=str, default='iso', choices=['iso', 'pca'], help='value prior variances initialisation')
parser.add_argument('--prior-variance-scale', type=float, default=1., help='scale prior variance by this value (default:1.)')
parser.add_argument('--learn-prior-variance', action='store_true', default=False, help='learn model prior variances')
# Ajout de l'argument `--regulariser`
parser.add_argument('--regulariser', type=str, default=None, choices=['mmd', 'mmd_dim'],
                    help="Type de régularisation à appliquer. Choix possibles : 'mmd', 'mmd_dim'. (default: None)")


# Computation
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA use')
parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed')

args = parser.parse_args()
# [DEBUG] Vérification de la dimension latente configurée
#print(f"\033[92m[DEBUG] Latent dimension configurée : {args.latent_dim}\033[0m")
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

# Setting random seed for reproducibility
if args.seed == 0:
    args.seed = int(torch.randint(0, 2**32 - 1, (1,)).item())
print('seed', args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True

directory_name = '../experiments/{}'.format(args.name)
if args.name != '.':
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    runPath = mkdtemp(prefix=runId, dir=directory_name)
else:
    runPath = mkdtemp(prefix=runId, dir=directory_name)
sys.stdout = Logger('{}/run.log'.format(runPath))
print('RunID:', runId)

# Save args
with open('{}/args.json'.format(runPath), 'w') as fp:
    json.dump(args.__dict__, fp)
with open('{}/args.txt'.format(runPath), 'w') as fp:
    git_hash = subprocess.check_output(['git', 'rev-parse', '--verify', 'HEAD'])
    command = ' '.join(sys.argv[1:])
    fp.write(git_hash.decode('utf-8') + '\n' + command)
torch.save(args, '{}/args.rar'.format(runPath))

# Load data and model
if args.model == 'rotated_mnist':
    print("Entraînement sur RotatedMNIST")
    train_loader, test_loader = flattened_rotMNIST(
        num_tasks=5, per_task_rotation=45, batch_size=args.batch_size, root='./data'
    )
    model = VAE_RotatedMNIST(args).to(device)
else:
    raise ValueError("Le modèle '{}' n'est pas supporté.".format(args.model))

# [DEBUG] Vérification de la configuration du modèle
#print(f"\033[93m[DEBUG] Modèle initialisé avec latent_dim={args.latent_dim}\033[0m")

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=args.lr
)

# Dynamically load objective function
objective = getattr(objectives, args.obj + '_objective', None)
t_objective = getattr(objectives, 'iwae_objective', None)

if t_objective is None:
    raise ValueError("Objective function 'iwae_objective' not found in 'objectives.py'. Check your implementation.")

def compute_mmd_dim_loss(z):
    """
    Calcule la perte MMD (Maximum Mean Discrepancy) entre deux distributions.
    Par exemple, compare une distribution isotropique normale N(0, 1) avec les échantillons latents z.

    Args:
        z: échantillons latents (de dimension [batch_size, latent_dim]).

    Returns:
        Une valeur scalaire représentant la perte MMD.
    """
    # Distribution cible (normale isotropique)
    z_prior = torch.randn_like(z)  # Distribution isotropique N(0, 1)
    
    # Kernel Gaussian
    def gaussian_kernel(x, y, sigma=1.0):
        x = x.unsqueeze(1)  # [batch_size, 1, latent_dim]
        y = y.unsqueeze(0)  # [1, batch_size, latent_dim]
        pairwise_sq_dist = torch.sum((x - y) ** 2, dim=2)  # [batch_size, batch_size]
        return torch.exp(-pairwise_sq_dist / (2 * sigma ** 2))

    # Kernel entre les échantillons z et z_prior
    k_zz = gaussian_kernel(z, z)
    k_zpzp = gaussian_kernel(z_prior, z_prior)
    k_zzp = gaussian_kernel(z, z_prior)

    # MMD computation
    mmd_loss = k_zz.mean() + k_zpzp.mean() - 2 * k_zzp.mean()
    return mmd_loss

def loss_function(recon_x, x, mu, logvar, regulariser=None, gamma=0.8, alpha=1.0):
    recon_means = recon_x.mean if isinstance(recon_x, torch.distributions.Distribution) else recon_x
    recon_means = recon_means.view_as(x)
    recon_loss = torch.nn.functional.binary_cross_entropy(recon_means, x, reduction='sum')

    # Divergence KL pondérée par Alpha
    kld = alpha * -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Pénalité L1 pour la sparsity
    sparsity_penalty = torch.sum(torch.abs(mu))

    # Régularisation optionnelle
    if regulariser == 'mmd':
        mmd_loss = compute_mmd_loss(mu)
        total_loss = recon_loss + kld + gamma * mmd_loss + 0.1 * sparsity_penalty
    else:
        total_loss = recon_loss + kld + 0.1 * sparsity_penalty

    return total_loss


# Fonction d'entraînement
def train(epoch, agg, regulariser, gamma):
    model.train()
    b_loss = 0.0
    for i, (data, _, _) in enumerate(train_loader):
        data = data.view(data.size(0), -1).to(device)

        optimizer.zero_grad()
        mu, logvar, recon_batch = model(data)
        # [DEBUG] Vérification des dimensions dans `train`
        #print(f"\033[94m[DEBUG] mu shape: {mu.shape}, logvar shape: {logvar.shape}\033[0m")

        assert mu.shape[1] == args.latent_dim, f"Erreur : Dimension latente incorrecte, attendu {args.latent_dim} mais obtenu {mu.shape[1]}"

        loss = loss_function(recon_batch, data, mu, logvar, 
                     regulariser=args.regulariser, 
                     gamma=args.gamma, 
                     alpha=args.alpha)
        loss.backward()
        optimizer.step()
        b_loss += loss.item()

    agg['train_loss'].append(b_loss / len(train_loader.dataset))
    print(f"====> Epoch: {epoch:03d} Loss: {agg['train_loss'][-1]:.2f}")


# Fonction de test
@torch.no_grad()
def test(epoch, beta, alpha, agg, regulariser, gamma):
    model.eval()
    b_loss = 0.0
    for i, (data, labels, _) in enumerate(test_loader):
        data = data.view(data.size(0), -1).to(device)

        mu, logvar, recon_batch = model(data)
        #print(f"\033[95m[DEBUG] mu shape: {mu.shape}, logvar shape: {logvar.shape}\033[0m")
        
        assert mu.shape[1] == args.latent_dim, f"Erreur : Dimension latente incorrecte, attendu {args.latent_dim} mais obtenu {mu.shape[1]}"

        loss = loss_function(recon_batch, data, mu, logvar, 
                     regulariser=args.regulariser, 
                     gamma=args.gamma, 
                     alpha=args.alpha)
        b_loss += loss.item()

        if (args.save_freq == 0 or epoch % args.save_freq == 0) and i == 0:
            model.reconstruct(data, runPath, epoch)

    avg_loss = b_loss / len(test_loader.dataset)
    agg['test_loss'].append(avg_loss)
    print('====> Test:      Loss: {:.2f}'.format(avg_loss))


# Main training loop
if __name__ == '__main__':
    with Timer('ME-VAE') as t:
        agg = defaultdict(list)
        print('Starting training...')
        
        for epoch in range(1, args.epochs + 1):
            # Entraînement
            train(epoch, agg, args.regulariser, args.gamma)

            # Génération forcée à chaque époque
            model.generate(runPath, epoch)
            
            # Reconstruction forcée à partir d'un batch du train loader
            data, _, _ = next(iter(train_loader))  # Récupérer un batch du train loader
            data = data.view(data.size(0), -1).to(device)  # Mettez les données à plat
            model.reconstruct(data, runPath, epoch)

            # Sauvegarde des modèles et des variables
            save_model(model, runPath + '/model.rar')
            save_vars(agg, runPath + '/losses.rar')
            
            # Test
            if not args.skip_test:
                test(epoch, args.beta, args.alpha, agg, args.regulariser, args.gamma)
        
        # Affichage final
        print("p(z) params:")
        print(model.pz_params)
        
        # Visualisation après l'entraînement
        # [DEBUG] Avant la visualisation
        #print(f"\033[96m[DEBUG] Visualisation de l'espace latent avec latent_dim={args.latent_dim}\033[0m")
        print("Visualisation de l'espace latent...")
        visualize_latent_space(model, test_loader, method='TSNE', num_samples=2000, save_path=f"{runPath}/latent_space.png")