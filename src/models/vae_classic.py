# La création d'une classe VAE classique dédiée est inutile pour notre étude.
# En effet, un VAE classique peut être obtenu directement en réglant les paramètres suivants :
# - Beta = 1
# - Alpha = 0
# - Gamma = 0
# Cela suffit pour entraîner rotated_mnist et comparer les résultats avec le Alpha-Beta VAE présenté dans le papier.