import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as F
import numpy as np
from torchvision.utils import save_image


# Ajoutez une activation Sigmoid à la sortie du décodeur
class VAE_RotatedMNIST(nn.Module):
    def __init__(self, args):
        super(VAE_RotatedMNIST, self).__init__()
        self.latent_dim = args.latent_dim  # Taille latente passée via les arguments
        # [DEBUG] Vérification de l'initialisation du modèle
        print(f"\033[93m[DEBUG] Initialisation du modèle avec latent_dim={self.latent_dim}\033[0m")

        # Paramètres pour le prior p(z)
        self.gamma = torch.tensor(0.8)  # Exemple, peut être ajusté
        self.prior_variance_scale = torch.tensor(1.0)  # Exemple, peut être ajusté
        self._pz_mu = nn.Parameter(torch.zeros(self.latent_dim))
        self._pz_logvar = nn.Parameter(torch.zeros(self.latent_dim))


        # Encodeur
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 400),
            nn.ReLU(),
            nn.Linear(400, 2 * self.latent_dim)  # Génère mu et logvar
        )

        # Décodeur
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 28 * 28)
        )


    def encode(self, x):
        h = self.encoder(x)
        z_mu, z_logvar = h[:, :self.latent_dim], h[:, self.latent_dim:]
        return z_mu, z_logvar


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """
        Decode latent variables z into reconstructed images.
        """
        h = self.decoder(z)
        decoded = torch.sigmoid(h)  # Map to [0, 1]
        # Commented out debugging print
        # print(f"Decoded shape after decode: {decoded.shape}")  # Debugging
        # This assertion can remain for safety during training
        assert decoded.shape[1] == 28 * 28, f"Decoded output has incorrect shape {decoded.shape}"
        return decoded


    
    def forward(self, x):
        """
        Passage avant pour le VAE.
        """
        B = x.size(0)  # Taille du batch
        x = x.view(B, -1)  # Aplatit les données d'entrée
        # print(f"Input data shape: {x.shape}")  # Ligne pour le débogage désactivée
        # Vérifie que la forme d'entrée est correcte
        assert x.shape[1] == 28 * 28, f"Problème de forme d'entrée : attendu 28*28, reçu {x.shape}"
    
        # Encodeur : génère la moyenne et la variance logarithmique de z
        z_mu, z_logvar = self.encode(x)
        std = torch.exp(0.5 * z_logvar)  # Écart-type
        eps = torch.randn_like(std)  # Échantillon gaussien
        zs = z_mu + eps * std  # Réparamétrisation pour générer les échantillons latents
    
        # Décodeur : reconstruit les données à partir des échantillons latents
        x_recon = self.decode(zs)
        # print(f"Decoded shape after decode: {x_recon.shape}")  # Ligne pour le débogage désactivée
    
        x_recon = x_recon.view(B, -1)  # Réorganise pour correspondre à la forme d'entrée d'origine
        # print(f"Decoded shape after reshape: {x_recon.shape}")  # Ligne pour le débogage désactivée
    
        x_recon = x_recon.clamp(0, 1)  # Limite les valeurs pour éviter les instabilités
        px_z = torch.distributions.Bernoulli(probs=x_recon)  # Distribution Bernoulli pour les données reconstruites
    
        return z_mu, z_logvar, px_z



    @property
    def pz_params(self):
        """
        Retourne les paramètres du prior p(z)
        """
        return (
            self.gamma,  # Poids de la composante spike
            self._pz_mu,  # Moyenne de p(z)
            torch.sqrt(self.prior_variance_scale * self.latent_dim * torch.softmax(self._pz_logvar, dim=0))  # Écart-type
        )

    def generate(self, runPath, epoch):
        #print(f"[DEBUG] Appel de generate pour l'époque {epoch}")
        #print(f"[DEBUG] Chemin de sauvegarde : {runPath}/gen_samples_{epoch:03d}.png")
        N, K = 64, 8  # N: Total d'images, K: Nombre par ligne dans la grille
        z = torch.randn(N, self.latent_dim).to(next(self.parameters()).device)
        samples = self.decode(z).view(-1, 1, 28, 28)
        save_image(samples.data.cpu(), f'{runPath}/gen_samples_{epoch:03d}.png', nrow=K)
        #print(f"[DEBUG] Images générées et sauvegardées avec succès.")
    

    def reconstruct(self, data, runPath, epoch):
        """
        Reconstruit les images du batch d'entrée et sauvegarde une seule grille.
        """
        #print(f"[DEBUG] Données d'entrée forme : {data.shape}")
        
        # Assurez-vous que les données sont mises à plat pour correspondre à l'entrée du modèle
        data = data.to(next(self.parameters()).device)
        #print(f"[DEBUG] Données mises à plat pour reconstruction : {data.shape}")
        
        # Reconstruire les données
        mu, logvar, recon_data = self(data)  # Passez les données via le modèle
        
        # Accéder aux probabilités si recon_data est une distribution
        if isinstance(recon_data, torch.distributions.Bernoulli):
            recon_data = recon_data.probs  # Accédez aux probabilités
        #print(f"[DEBUG] Taille de recon_data après probs : {recon_data.shape}")
        
        try:
            # Redimensionner les données reconstruites pour correspondre au format des images
            recon_data = recon_data.view(-1, 1, 28, 28)
            #print(f"[DEBUG] Taille de recon_data après view : {recon_data.shape}")
        except Exception as e:
            # En cas d'erreur, afficher un message de débogage et lever une exception
            #print(f"[DEBUG] Erreur lors du reshape de recon_data : {e}")
            raise e
    
        # Redimensionner les données d'entrée pour correspondre au format des images
        data = data.view(-1, 1, 28, 28)
    
        # Créer une grille avec les entrées et les reconstructions
        comp = torch.cat([data, recon_data])
        save_image(comp.data.cpu(), f'{runPath}/recon_{epoch:03d}.png', nrow=8)
        #print(f"[DEBUG] Images reconstruites sauvegardées dans {runPath}/recon_{epoch:03d}.png")




class RotatedMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, root='./data', train=True, transform=None, download=True, num_tasks=5, per_task_rotation=45):
        if not isinstance(root, str):
            raise ValueError(f"Le chemin root doit être une chaîne, mais {type(root)} reçu.")
        self.root = root
        self.dataset = torchvision.datasets.MNIST(root=self.root, train=train, transform=None, download=download)
        self.transform = transform
        self.rotation_angles = [float(task * per_task_rotation) for task in range(num_tasks)]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        angle = np.random.choice(self.rotation_angles)  # Choix aléatoire d'un angle de rotation
        rotated_image = F.rotate(image, angle, fill=(0,))  # Rotation de l'image

        # Convertir en Tensor et garantir que les données restent dans [0, 1]
        if self.transform:
            rotated_image = self.transform(rotated_image)
        rotated_image = rotated_image.clamp(0, 1)  # Clamp pour garantir [0, 1]

        return rotated_image, label, angle


def flattened_rotMNIST(num_tasks, per_task_rotation, batch_size, transform=None, root='./data'):
    '''
    Retourne les DataLoaders pour le dataset RotatedMNIST
    '''
    g = torch.Generator()
    g.manual_seed(0)  # Même seed pour la reproductibilité

    if transform is None:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),  # Convertit les pixels en [0, 1]
        ])


    train = RotatedMNISTDataset(root=root, train=True, download=True, transform=transform,
                                 num_tasks=num_tasks, per_task_rotation=per_task_rotation)
    test = RotatedMNISTDataset(root=root, train=False, download=True, transform=transform,
                                num_tasks=num_tasks, per_task_rotation=per_task_rotation)

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True,
                                               num_workers=0, pin_memory=True, generator=g)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False,
                                              num_workers=0, pin_memory=True, generator=g)

    return train_loader, test_loader