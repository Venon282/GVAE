# Comprendre les VAE : guide complet et accessible

Ce document explique les VAE (Variational Autoencoders) en partant de zéro. Chaque terme est défini au moment où il apparaît, y compris ceux qui semblent "évidents" pour quelqu'un déjà familier du sujet : la littérature sur les VAE est souvent mal vulgarisée, avec une notation qui suppose déjà des années de statistiques. Ce document reprend tout dans l'ordre, avec des liens directs vers le code du projet (`global-vae-project-specification.md` et les fichiers de `src/global_vae/`) à chaque étape.

---

## 1. Le point de départ : l'autoencodeur classique

Avant le "V" de VAE, il y a l'autoencodeur tout court (AE). L'idée est simple :

- Un **encoder** (encodeur) prend une donnée d'entrée `x` et la compresse en un petit vecteur `z`, appelé le **code latent** (ou **latent space**, "espace latent", pour l'ensemble des `z` possibles).
- Un **decoder** (décodeur) prend ce `z` et essaie de reconstruire `x` à l'identique, donnant une reconstruction qu'on note `x̂` (x-chapeau).

Analogie : résumer un livre en une fiche de 10 lignes (`z`), puis charger quelqu'un d'autre de réécrire le livre entier à partir de cette seule fiche (`x̂`). Si le résumé est bon, le livre reconstruit ressemble à l'original.

**Le problème d'un AE classique** : rien n'oblige les `z` de deux entrées similaires à être proches l'un de l'autre dans l'espace latent, ni l'espace latent à être "rempli" de façon continue. Résultat : en piochant un point au hasard dans cet espace (pour *générer* une donnée nouvelle, jamais vue), il y a de fortes chances de tomber dans un "trou" que le décodeur n'a jamais appris à interpréter, produisant n'importe quoi. Un AE classique sait compresser et reconstruire, mais ne sait pas *générer*.

C'est exactement le problème que le V (Variational) vient résoudre.

---

## 2. L'idée du VAE : remplacer un point par un nuage

Un VAE (Variational Autoencoder, autoencodeur variationnel) change une seule chose, mais elle a des conséquences énormes : au lieu que l'encoder produise un point `z` unique et précis, il produit une **distribution de probabilité** : une sorte de "nuage" de valeurs possibles pour `z`, avec une région centrale plus probable et des bords moins probables.

Une **distribution de probabilité**, c'est une règle qui dit, pour chaque valeur possible, à quel point elle est probable. Le VAE utilise la distribution la plus simple et la mieux comprise en statistiques : la **loi normale** (ou **gaussienne**, du nom du mathématicien Carl Friedrich Gauss), la fameuse "courbe en cloche". Une gaussienne est entièrement décrite par deux nombres :

- la **moyenne** (`mu`, la lettre grecque μ) : le centre de la cloche, la valeur la plus probable.
- la **variance** (souvent notée σ², sigma carré) : à quel point la cloche est large ou étroite. Une variance petite = cloche étroite = on est presque sûr de la valeur. Une variance grande = cloche large = beaucoup d'incertitude. La racine carrée de la variance s'appelle l'**écart-type** (σ, sigma), dans la même unité que la donnée elle-même.

Concrètement, pour une entrée `x` donnée, l'encoder ne produit plus un seul `z`, mais deux vecteurs : `mu` (le centre de la distribution) et quelque chose lié à la variance (voir juste en dessous pourquoi ce n'est pas la variance directement). C'est visible tel quel dans `AbstractEncoder.forward` :

```python
def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Retourne (mu, logvar), chacun de forme (batch, latent_dim)."""
```

### Pourquoi `logvar` et pas la variance directement ?

La variance doit toujours être positive (une "largeur" négative n'a pas de sens). Mais un réseau de neurones, en sortie d'une simple couche linéaire, peut produire n'importe quel nombre, positif ou négatif. Si on demandait au réseau de produire directement la variance, rien ne l'empêcherait de sortir un nombre négatif à un moment de l'entraînement, ce qui casserait tout.

La solution : le réseau produit `logvar` (le logarithme de la variance), qui lui peut être n'importe quel nombre réel sans problème, puis on repasse en variance réelle avec `exp(logvar)` (l'exponentielle "annule" le logarithme et garantit un résultat toujours positif, quel que soit `logvar`). C'est ce que fait exactement `LatentSpace.reparameterize` (`src/global_vae/latent/base.py`) :

```python
std = torch.exp(0.5 * logvar)   # écart-type = exp(logvar / 2)
```

(Le `0.5 *` vient du fait qu'on veut l'écart-type, pas la variance : écart-type = racine carrée de la variance = exp(logvar)^0.5 = exp(0.5 * logvar).)

---

## 3. Le prior : le plan qu'on impose à l'espace latent

**Prior** signifie "antérieur", "d'avant" (du latin *a priori*, "avant l'expérience"). En statistiques bayésiennes, le prior est la croyance qu'on a sur une quantité *avant même d'avoir regardé la donnée*. Ici, c'est la forme qu'on décide, à l'avance, de vouloir donner à tout l'espace latent, indépendamment de n'importe quelle entrée `x` particulière.

Dans un VAE, on choisit presque toujours le même prior : la loi normale centrée réduite, notée `N(0, I)` :

- moyenne = 0 dans chaque dimension du latent
- variance = 1 dans chaque dimension
- les dimensions sont indépendantes entre elles (le `I`, pour "matrice identité", signifie qu'il n'y a pas de corrélation entre les dimensions)

C'est un choix, pas une nécessité mathématique absolue, mais un excellent choix pratique pour trois raisons :

1. **C'est trivial à échantillonner.** Tirer un vecteur aléatoire selon `N(0, I)` revient juste à tirer des nombres aléatoires "normaux" indépendants, une opération de base dans n'importe quelle librairie numérique (`torch.randn_like(...)`).
2. **C'est bien compris mathématiquement**, ce qui permet d'avoir une formule simple pour comparer une distribution à ce prior (section 8).
3. **Ça donne la génération "gratuitement"** : une fois le modèle entraîné, pour créer une donnée entièrement nouvelle, il suffit de tirer un `z` au hasard depuis ce même prior `N(0, I)`, puis de le donner au decoder. Pas besoin d'entrée `x` du tout.

Le prior, c'est donc littéralement le "moule" qu'on veut que l'espace latent adopte, pour qu'on puisse ensuite piocher dedans à l'aveugle et obtenir toujours quelque chose que le decoder sait interpréter.

---

## 4. Posterior : le plan que l'on a réellement de l'espace latent

Une fois qu'on a défini le prior (la distribution qu'on *souhaite*), il faut un mot pour désigner la distribution que l'encoder produit *réellement*, pour une entrée `x` précise. C'est le **posterior** ("postérieur", "d'après", c'est-à-dire après avoir observé la donnée).

- Prior `p(z)` : la croyance de départ sur `z`, avant d'observer quoi que ce soit. Toujours `N(0, I)`, fixe, ne dépend d'aucune entrée.
- Posterior `q(z|x)` : la croyance mise à jour sur `z`, une fois qu'on a observé une entrée `x` précise. C'est exactement `(mu, logvar)` que l'encoder calcule pour ce `x`-là. Le `q(z|x)` se lit "la distribution de `z` sachant `x`".

Chaque entrée `x` a donc son propre posterior (son propre `mu`, sa propre variance), mais un seul prior partagé par tout le modèle. Tout l'objectif de l'entraînement est de faire en sorte que ces posteriors, pris ensemble, ressemblent le plus possible au prior, tout en restant suffisamment distincts d'une entrée à l'autre pour que le decoder puisse encore les différencier. C'est cette tension qui est au cœur de tout VAE (section 9).

---

## 5. Le reparameterization trick : comment on peut quand même faire du gradient descent

Une fois qu'on a `(mu, logvar)`, il faut *tirer* un `z` concret dans cette distribution pour pouvoir le donner au decoder. Le problème : "tirer au hasard" est une opération non différentiable. Or, l'entraînement d'un réseau de neurones repose entièrement sur la **descente de gradient** (gradient descent) : on calcule à quel point chaque poids du réseau a contribué à l'erreur finale, en remontant les calculs à l'envers (la **rétropropagation**, *backpropagation*). Si un tirage aléatoire brut se trouve au milieu de la chaîne de calculs, on ne peut plus remonter le gradient à travers lui : le calcul est "cassé" à cet endroit.

L'astuce, le **reparameterization trick**, consiste à sortir la partie aléatoire du chemin qu'on doit dériver :

```
epsilon ~ N(0, 1)          # un tirage aléatoire "pur", sans rapport avec le réseau
z = mu + epsilon * std     # combinaison différentiable de mu, std (calculés par le réseau) et epsilon
```

`epsilon` est un simple bruit externe, sans paramètre à apprendre. Tout ce qui compte pour le gradient, c'est `mu` et `std`, qui eux sont bien calculés par le réseau et donc différentiables normalement. C'est très exactement le code de `LatentSpace.reparameterize` :

```python
def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)   # epsilon, tiré au hasard, hors du graphe de calcul appris
    return mu + eps * std
```

C'est le petit tour de passe-passe qui rend tout le reste possible : sans lui, on ne pourrait tout simplement pas entraîner un VAE avec de la descente de gradient classique.

---

## 6. KL : le régalurisateur

### D'où vient le nom

**KL** est simplement les initiales de deux noms de famille : **Solomon Kullback** et **Richard Leibler**, deux statisticiens américains qui ont formalisé cette mesure en 1951. La "divergence de Kullback-Leibler" (ou "KL divergence") mesure à quel point une distribution de probabilité diffère d'une autre.

Ce n'est pas une vraie "distance" au sens mathématique strict (elle n'est pas symétrique : la divergence de A vers B n'est pas forcément égale à celle de B vers A), mais elle se comporte intuitivement comme une distance :

- si les deux distributions sont identiques, la divergence KL vaut **0**
- plus elles diffèrent, plus la divergence est **grande**

### Ce qu'elle mesure dans un VAE

On l'utilise pour comparer le posterior `q(z|x)` (ce que l'encoder a produit pour cette entrée précise) au prior `p(z) = N(0, I)` (ce qu'on voudrait que ça ressemble, en général). On la note `KL(q(z|x) || p(z))`.

### La formule, terme par terme

Comme le prior et le posterior sont tous les deux des gaussiennes, il existe une formule exacte (pas besoin d'approximation) pour cette divergence. C'est celle codée dans `KlStandardNormalRegularizer` (`src/global_vae/losses/regularizers/kl_standard_normal.py`) :

```python
def forward(self, mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
```

Décortiquons-la avec des valeurs concrètes :

- **Si `mu = 0` et `logvar = 0`** (donc variance = `exp(0) = 1`) : c'est exactement le prior `N(0, 1)`. Le calcul donne `-0.5 * (1 + 0 - 0 - 1) = 0`. Logique : si le posterior est identique au prior, la divergence est nulle.
- **Si `mu` s'éloigne de 0** : le terme `mu²` grandit, ce qui rend le résultat entre parenthèses plus négatif, donc (à cause du `-0.5 *` devant) le KL final plus grand. Autrement dit : *s'éloigner du centre du prior coûte cher.*
- **Si `logvar` devient très négatif** (donc la variance devient minuscule, l'encoder devient "trop confiant" sur un `z` quasi ponctuel) : le terme `logvar` diminue et `exp(logvar)` tend vers 0, ce qui fait aussi grandir le KL. Autrement dit : *devenir trop précis ou trop étroit coûte cher aussi, pas seulement s'éloigner du centre.*

Cette formule pénalise donc deux choses à la fois : s'éloigner du centre du prior, et être trop confiant (variance trop petite). Elle "tire" chaque posterior vers `N(0, I)` dans les deux sens.

### Pourquoi on a besoin de cette contrainte

Sans terme KL du tout, rien n'empêcherait l'encoder de faire exactement ce que fait un autoencodeur classique (section 1) : placer chaque entrée `x` à un endroit totalement arbitraire de l'espace latent, avec une variance minuscule (quasi un point). Le modèle apprendrait alors une sorte d'annuaire "cette entrée précise va exactement ici", ce qui reconstruit très bien les entrées connues mais laisse l'espace latent truffé de trous non organisés : exactement le problème qu'on cherchait à éviter en section 1.

Le terme KL, c'est la "colle" qui force tous les posteriors à rester proches d'un seul et même moule partagé (`N(0, I)`), et donc l'espace latent à rester continu et exploitable, dans son ensemble.

---

## 7. La reconstruction : l'autre moitié de la perte

Le decoder reçoit `z` et doit produire quelque chose qui ressemble à `x`. Pour mesurer "à quel point ça ressemble", on utilise une fonction de perte (loss) de reconstruction, la plus simple et la plus courante étant la **MSE** (*Mean Squared Error*, erreur quadratique moyenne) : on calcule la différence entre chaque valeur de `x` et de `x̂`, on met au carré (pour que les erreurs positives et négatives ne s'annulent pas, et pour pénaliser plus fort les grosses erreurs), puis on fait la moyenne. C'est `losses/reconstruction.py` dans ce projet, avec `torch.nn.functional.mse_loss` comme choix par défaut.

---

## 8. La perte totale : reconstruction + KL, et la tension entre les deux

La perte totale d'un VAE est simplement l'addition des deux morceaux :

```
loss_totale = loss_reconstruction + beta * loss_KL
```

C'est exactement `StepLosses` dans `training/trainer.py` :

```python
total_loss = reconstruction_loss + regularization_loss
```

Ces deux termes tirent dans des directions opposées, et c'est précisément ce qui rend l'entraînement d'un VAE délicat :

- Minimiser **seulement** la reconstruction pousserait l'encoder à redevenir un autoencodeur classique (variance minuscule, mémorisation par cœur) : bonne reconstruction, mauvais espace latent.
- Minimiser **seulement** le KL pousserait l'encoder à ignorer complètement `x` et à toujours produire `mu = 0`, `logvar = 0` (le prior lui-même, qui a un KL nul par définition) : bon espace latent bien organisé, mais qui ne contient plus aucune information utile sur `x`. Le decoder, privé de toute information, ne peut alors produire qu'une reconstruction moyenne, floue, identique pour tout le monde.

Un bon VAE trouve un compromis entre les deux. Dans la littérature, ce compromis a un nom formel : l'**ELBO** (*Evidence Lower BOund*, "borne inférieure de l'évidence"). C'est le nom mathématique exact de cette quantité (reconstruction moins KL, à un signe près selon les conventions), qu'on cherche à maximiser. Sa dérivation complète n'est pas nécessaire pour utiliser le framework, seulement le fait que "loss = reconstruction + beta * KL" *est* littéralement une reformulation pratique de l'ELBO.

### Le rôle de `beta`

`beta` (β) est un simple coefficient multiplicatif qui pondère l'importance du terme KL par rapport à la reconstruction. C'est le "β" du terme **β-VAE**, une variante largement utilisée. Plus `beta` est grand, plus on force l'espace latent à ressembler au prior (au risque d'ignorer `x`) ; plus il est petit, plus on privilégie la fidélité de reconstruction (au risque de retomber sur un espace latent mal organisé).

Dans ce projet, `beta` peut même varier au cours de l'entraînement, via `training/beta_schedules/` :

- `ConstantBetaSchedule` : `beta` fixe du début à la fin.
- `LinearWarmupBetaSchedule` : `beta` démarre à 0 et grimpe progressivement.
- `CyclicalAnnealingBetaSchedule` : `beta` remonte et redescend plusieurs fois pendant l'entraînement.

---

## 9. Le piège classique : le "posterior collapse"

Si `beta` est trop élevé, ou monte trop vite, dès le tout début de l'entraînement (avant même que le decoder ait appris quoi que ce soit d'utile à partir de `z`), l'encoder trouve le chemin le plus facile pour faire baisser vite la perte : produire `mu = 0`, `logvar = 0` pour absolument toutes les entrées, c'est-à-dire recopier le prior exactement, quel que soit `x`. Le KL tombe à 0, l'encoder "a gagné" sa moitié de la partie, mais il a complètement cessé d'utiliser `z` pour transmettre la moindre information sur `x`. C'est le **posterior collapse** (l'effondrement du posterior sur le prior).

Le symptôme : la reconstruction reste médiocre indéfiniment, quelle que soit la capacité du modèle (plus de couches, plus de paramètres) qu'on ajoute, parce que le problème n'est pas un manque de capacité, mais une latente qui ne porte plus aucune information à exploiter.

C'est exactement ce que le projet documente dans `examples/01_signal_vae_pipeline.py` (section "On regularization" de son docstring) : une première version, avec le régulariseur `kl_standard_normal` par défaut et un warm-up trop court, restait bloquée à un R² d'environ 0.3 (un score de qualité de reconstruction proche de "à peine mieux que prédire la moyenne partout"), quelle que soit la capacité ajoutée au modèle : la signature typique d'un collapse.

Deux parades, déjà présentes dans ce framework :

- **Le beta warm-up** (`LinearWarmupBetaSchedule`) : laisser `beta` très bas au début, le temps que le decoder apprenne réellement à se servir de `z`, avant de monter progressivement la pression de régularisation.
- **Les "free bits"** (`FreeBitsKlRegularizer`, `losses/regularizers/free_bits_kl.py`) : donner à chaque dimension du latent un petit "budget" de KL qu'elle n'est jamais pénalisée d'utiliser. Concrètement, en dessous d'un certain seuil (`free_bits`, souvent autour de 0.5 à 1.0), le KL de cette dimension ne rapporte plus rien pour faire baisser la perte, donc l'encoder n'a plus aucune incitation à l'écraser complètement vers 0. C'est cette solution que `01_signal_vae_pipeline.py` utilise finalement, faisant remonter le R² d'environ 0.3 à environ 0.97.

---

## 10. Vue d'ensemble : le trajet complet dans `GlobalVae`

En reliant tout ce qu'on vient de voir, voici le trajet exact qu'un batch d'entrées suit dans le code (`GlobalVae.forward`, `src/global_vae/models/global_vae.py`) :

1. `x` (l'entrée brute) passe dans l'**encoder**, qui produit `(mu, logvar)`. (`AbstractEncoder.forward`)
2. `(mu, logvar)` passe dans **reparameterize**, qui produit un `z` concret, échantillonné. (`LatentSpace.reparameterize`)
3. `z` passe dans le **decoder**, qui produit `x̂`, la reconstruction. (`AbstractDecoder.forward`)
4. On calcule la **perte de reconstruction** entre `x` et `x̂` (`losses/reconstruction.py`), et la **perte KL** entre `(mu, logvar)` et le prior `N(0, I)` (`losses/regularizers/kl_standard_normal.py`, ou une autre stratégie enregistrée).
5. `loss_totale = reconstruction + beta * KL`, et on fait un pas de descente de gradient dessus (`training/trainer.py::computeLosses` puis `.backward()`).

À l'inférence (une fois le modèle entraîné), il y a deux usages distincts, tous les deux déjà présents dans ce code :

- **Reconstruire une entrée existante** : passer par les étapes 1 à 3 normalement (`evaluate()` utilise même `use_mean=True`, c'est-à-dire qu'on prend directement `z = mu`, sans tirage aléatoire, pour avoir un résultat déterministe et reproductible d'une exécution à l'autre).
- **Générer une donnée entièrement nouvelle** : sauter l'étape 1 et 2 entièrement. Tirer directement `z ~ N(0, I)` (le prior), et ne faire que l'étape 3 (`decoder(z)`). C'est là que tout l'effort mis dans le terme KL paie : parce que l'espace latent entier a été entraîné à ressembler à ce prior, n'importe quel `z` tiré au hasard dedans donne quelque chose de plausible au decoder, même si ce `z` précis n'a jamais été produit par un vrai `x` pendant l'entraînement.

---

## 11. Bonus : et avec plusieurs modalités (signal + image) ?

Un aperçu très rapide, pour le jour où la Fusion (`fusion/poe.py`) entre en jeu : quand plusieurs encoders (un par modalité) alimentent le **même** espace latent, chacun produit son propre `(mu, logvar)`, comme des "experts" indépendants sur la même variable `z`. La stratégie **Product-of-Experts (PoE)** combine ces experts en un seul posterior fusionné, *avant* l'échantillonnage, en donnant plus de poids aux experts les plus confiants (ceux dont la variance est la plus petite). Tout le reste (reparameterize, decoder, pertes) fonctionne ensuite exactement comme décrit plus haut, sur ce posterior déjà fusionné. C'est pour ça que la spec distingue bien la Fusion (combiner des distributions, avant échantillonnage) de l'Assembler (combiner des vecteurs déjà échantillonnés, après) : deux étapes différentes du pipeline.

---

## Petit glossaire, pour retrouver un mot vite

| Terme | Définition courte |
|---|---|
| **Autoencodeur (AE)** | Réseau qui compresse une entrée en un code, puis la reconstruit à partir de ce code. |
| **VAE** | Autoencodeur dont l'encoder produit une distribution (pas un point) et dont l'espace latent est régularisé pour ressembler à un prior choisi. |
| **Encoder** | Réseau qui transforme l'entrée `x` en `(mu, logvar)`. |
| **Decoder** | Réseau qui transforme un `z` en reconstruction `x̂`. |
| **Espace latent (latent space)** | L'ensemble des `z` possibles ; la représentation compressée. |
| **z** | Un vecteur latent concret, échantillonné à partir de `(mu, logvar)`. |
| **mu (μ)** | La moyenne de la distribution produite par l'encoder : le centre le plus probable. |
| **logvar** | Le logarithme de la variance ; permet au réseau de sortir n'importe quel nombre tout en garantissant une variance positive après `exp()`. |
| **Écart-type (σ)** | Racine carrée de la variance ; mesure la largeur de la distribution, dans la même unité que la donnée. |
| **Distribution de probabilité** | Une règle qui donne la probabilité de chaque valeur possible. |
| **Gaussienne / loi normale** | La distribution "en cloche", entièrement définie par sa moyenne et sa variance. |
| **Prior `p(z)`** | La distribution qu'on impose à l'avance à l'espace latent, fixe, `N(0, I)` par défaut. |
| **Posterior `q(z\|x)`** | La distribution que l'encoder produit réellement, pour une entrée `x` précise. |
| **Reparameterization trick** | `z = mu + epsilon * std`, la technique qui rend l'échantillonnage différentiable. |
| **KL divergence** | Mesure (due à Kullback et Leibler) de l'écart entre deux distributions ; 0 si identiques. |
| **ELBO** | *Evidence Lower BOund*, la quantité formelle que "reconstruction + KL" reformule en pratique. |
| **Beta (β-VAE)** | Coefficient qui pondère le poids du terme KL dans la perte totale. |
| **Posterior collapse** | Panne où l'encoder ignore `x` et produit toujours le prior ; `z` ne porte plus d'information. |
| **Free bits** | Un petit budget de KL, par dimension, jamais pénalisé, pour éviter le collapse. |
| **Reconstruction loss** | Mesure de l'écart entre `x` et `x̂` (souvent la MSE). |
| **Fusion (PoE, etc.)** | Combine plusieurs `(mu, logvar)` (un par modalité) en un seul posterior, avant échantillonnage. |
| **Assembler** | Combine plusieurs `z` déjà échantillonnés en un seul vecteur, pour le decoder. |
