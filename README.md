# RL — Taxi-v3 (Gymnasium) : SARSA, Q-Learning, Expected SARSA, Monte Carlo, Double Q-Learning

Ce projet compare plusieurs algorithmes classiques d’apprentissage par renforcement **tabulaires** (Q-table) sur l’environnement **Taxi-v3** de Gymnasium. L’objectif est d’apprendre une politique qui déplace le taxi, récupère le passager et le dépose à la bonne destination.

> Environnement de référence : Taxi-v3 (Toy Text) :contentReference[oaicite:0]{index=0}
<img width="550" height="350" alt="image" src="https://github.com/user-attachments/assets/657d40c8-c773-4b9e-a4f6-17a55ddf9c6c" />

## 1) Fonctionnement du projet

### Environnement Taxi-v3
- **États (observation space)** : `Discrete(500)` (position taxi 25 × position passager 5 × destination 4). :contentReference[oaicite:1]{index=1}  
- **Actions (action space)** : `Discrete(6)`  
  0=Sud, 1=Nord, 2=Est, 3=Ouest, 4=Pickup, 5=Dropoff. :contentReference[oaicite:2]{index=2}  
- **Récompenses** : `-1` par pas, `+20` si dépôt réussi, `-10` si pickup/dropoff illégal. :contentReference[oaicite:3]{index=3}  
- **Fin d’épisode** : terminaison au dropoff (réussi), troncature typique à **200 pas** via TimeLimit. :contentReference[oaicite:4]{index=4}

> Note : Taxi-v3 peut fournir un `action_mask` (dans `info`) pour éviter les actions invalides (pickup/dropoff illégaux), ce qui accélère souvent l’apprentissage. :contentReference[oaicite:5]{index=5}

### Pipeline commun à tous les algorithmes
1. **Initialisation** : Q-table `Q[s, a] = 0` de taille `(n_states, n_actions)` → `(500, 6)`.
2. **Politique d’exploration** : `epsilon_greedy(Q, state, epsilon)`  
   - avec proba ε : action aléatoire  
   - sinon : `argmax_a Q[s, a]`
3. **Décroissance d’epsilon** : `epsilon = max(epsilon_end, epsilon * epsilon_decay)` (exploration forte au début, plus faible à la fin).
4. **Entraînement épisodique** : boucle sur `episodes`, chaque épisode limité à `max_steps=200`.
5. **Suivi des performances** : `rewards_history` + **moyenne mobile** (moving average) pour lisser la courbe.
6. **Évaluation finale** (greedy) : on fixe `epsilon = 0` implicitement en prenant toujours `argmax`, on mesure :
   - score moyen (et écart-type)
   - taux de succès (dropoff réussi)
  
  ## 2) Algorithmes implémentés

### SARSA (on-policy)
SARSA apprend la valeur de la **politique suivie réellement** (incluant l’exploration ε-greedy).

**Mise à jour :**
\[
Q(s,a) \leftarrow Q(s,a) + \alpha \Big(r + \gamma Q(s',a') - Q(s,a)\Big)
\]
où `a'` est **l’action choisie par la politique (ε-greedy)** dans `s'`.

✅ Points clés :
- plus “prudent” car il tient compte de l’exploration dans la cible
- souvent stable
  
---

### Q-Learning (off-policy)
Q-Learning apprend directement la **politique optimale** indépendamment de la politique utilisée pour explorer.

**Mise à jour :**
\[
Q(s,a) \leftarrow Q(s,a) + \alpha \Big(r + \gamma \max_{a'} Q(s',a') - Q(s,a)\Big)
\]

✅ Points clés :
- vise la meilleure action possible au prochain état (max)
- peut apprendre vite, mais peut surestimer (voir Double Q)

---

### Expected SARSA (compromis)
Expected SARSA remplace `Q(s', a')` (SARSA) et `max Q(s',a')` (Q-learning) par une **espérance** sous la politique ε-greedy.

**Idée utilisée dans le code :**
\[
\mathbb{E}[Q(s',a')] \approx (1-\varepsilon)\max Q(s',\cdot) + \varepsilon \,\text{mean}(Q(s',\cdot))
\]
✅ Points clés :
- souvent moins de variance que SARSA
- plus stable que Q-learning dans certains cas

---

### Monte Carlo First-Visit (sans bootstrapping)
Monte Carlo met à jour **uniquement à la fin de l’épisode** à partir du **retour réel** \(G\).

**Mise à jour :**
\[
Q(s,a) \leftarrow Q(s,a) + \alpha (G - Q(s,a))
\]

✅ / ⚠️ Points clés :
- pas de bootstrapping → pas de biais TD, mais **variance élevée**
- sur Taxi-v3, les retours peuvent être très bruités (beaucoup de -1 et pénalités) et l’apprentissage peut être lent/instable si on n’ajoute pas des astuces (meilleure exploration, action masking, ou moyenne incrémentale plutôt qu’un α fixe).

---

### Double Q-Learning (réduit la surestimation)
Double Q-Learning maintient deux estimateurs `Q_A` et `Q_B` pour **décorréler sélection** et **évaluation** :
- si on met à jour `Q_A` : action choisie par `argmax Q_A`, évaluée via `Q_B`
- et inversement pour `Q_B`

✅ Points clés :
- réduit la surestimation due au `max`
- peut être un peu plus coûteux (2 tables), mais souvent plus fiable

---

## 3) Résultats (exemple de run fourni)
Paramètres typiques : `episodes=20000`, `alpha=0.1`, `gamma=0.99`, `epsilon: 1.0 → 0.05`, `max_steps=200`.

Évaluation greedy (100 épisodes) :
- **SARSA** : Score ≈ **7.63 ± 3.00**, succès **100%**
- **Q-Learning** : Score ≈ **7.67 ± 2.95**, succès **100%**
- **Expected SARSA** : Score ≈ **7.67 ± 2.95**, succès **100%**
- **Double Q-Learning** : Score ≈ **7.67 ± 2.95**, succès **100%**
- **Monte Carlo** : Score ≈ **-369.81 ± 608.00**, succès **22%**

(Interprétation rapide : les méthodes TD tabulaires convergent bien sur Taxi-v3, tandis que Monte Carlo “pur” peut galérer sans réglages/techniques supplémentaires.)

---

