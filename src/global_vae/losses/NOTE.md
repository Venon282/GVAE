# Deferred

Reconstruction-loss and KL-loss modules per modality/latent-space live
here once needed. `GlobalVae.computeKlLoss` currently covers the
single-latent KL term inline; per-modality reconstruction losses and
beta-schedule/annealing logic are deferred pending the open question
in spec §11 ("Precise loss weighting / β-VAE annealing schedule").
