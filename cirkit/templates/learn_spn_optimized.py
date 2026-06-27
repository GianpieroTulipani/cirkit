"""
learn_spn_optimized.py  —  versione ANNOTATA e RIVISTA di cirkit/templates/learn_spn.py

NON sostituisce il file originale: e' pensata per essere LETTA e CONFRONTATA
(diff) con `cirkit/templates/learn_spn.py`. Integra le ottimizzazioni discusse.

Mappa delle modifiche (cerca i marcatori `# === OPT n`):

  OPT 1  Parametrizzazione coerente con l'attivazione.
         L'originale salva SEMPRE `np.log(probs)` nel TensorParameter e poi ci
         applica l'attivazione. E' corretto SOLO per softmax (softmax(log p)=p);
         per 'positive-clamp' i log-prob (negativi) vengono schiacciati sul floor
         ~1e-19 -> init distrutta. Qui `_to_preactivation` mappa le probabilita'
         nello spazio PRE-attivazione giusto:
           - softmax / none / exp  -> log(p)            (softmax esatto)
           - softplus              -> log(expm1(p))     (inverso esatto di softplus)
           - positive-clamp        -> p   (lineare)     (clamp(p)=p, niente unita' morte)
         Inoltre, nel percorso NON stimato, 'normal' (centrata in 0) e' patologica
         per il clamp -> si passa a un init positivo ('uniform').

  OPT 3  Diversita' che scala con la larghezza K.
         L'originale replica UNA stima su tutte le unita' (tile) + rumore: le K
         unita' sono cloni, l'informazione non cresce con K. Qui ogni unita' riceve
         una stima DIVERSA e data-driven (bootstrap del sottoinsieme, di default;
         'subcluster' come hook), cosi' il rango effettivo dell'init cresce con K.

  OPT 4  Rumore piccolo, solo rompi-simmetria.
         noise_scale di default 2.0 -> 0.2. Il rumore non e' piu' la fonte di
         diversita' (che ora viene da OPT 3): serve solo a non avere unita' identiche.
         In spazio log e' additivo; per il clamp e' moltiplicativo-positivo (non
         spinge i pesi sotto il floor).

  OPT 6  Foglie stabili.
         (a) `leaf_pool='global'` stima la marginale di foglia su TUTTE le righe
             (la "vecchia scelta"): in un circuito iperparametrizzato/tensorizzato
             la foglia e' condivisa, quindi la marginale globale e' una base migliore;
             la specializzazione la fanno i pesi di somma a valle + OPT 3.
         (b) `adaptive_alpha=True`: la massa di prior totale resta `alpha` (alpha/C
             per categoria) invece di `alpha*C`, cosi' lo smoothing non appiattisce
             le foglie profonde verso l'uniforme.

  OPT 7  Niente overwrite dei mixing layer.
         L'originale sovrascrive QUALSIASI SumLayer con un tensore denso, distruggendo
         la struttura block-diagonale di MixingWeightParameter e aggiungendo parametri
         O(K^2) non informati (che crescono x4 da 256 a 512). Qui i mixing layer
         vengono RICONOSCIUTI (isinstance(layer.weight.output, MixingWeightParameter))
         e NON toccati: si lascia l'init strutturato del factory. Inoltre, essendo una
         mixture su decomposizioni ALTERNATIVE della stessa scope, le righe non vanno
         clusterizzate: si passano intere ai figli (come per i product layer).

  OPT 5  (fuori da questo file) early stopping / schedule LR / weight_decay vivono in
         mnist_dataset_eval.py + config_mnist.yml. Promemoria in fondo al file.
"""

import random
import functools
from collections import deque
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch import LongTensor, Tensor

from fast_pytorch_kmeans import KMeans

from cirkit.symbolic.circuit import Circuit
from cirkit.templates.miwae import ConvVAE
from cirkit.symbolic.layers import SumLayer, InputLayer
from cirkit.symbolic.parameters import (
    TensorParameter,
    Parameter,
    ParameterFactory,
    MixingWeightParameter,
    mixing_weight_factory,
)
from cirkit.symbolic.initializers import ConstantTensorInitializer
from cirkit.templates.utils import (
    Parameterization,
    name_to_input_layer_factory,
    parameterization_to_factory,
    name_to_parameter_activation,
)
from cirkit.templates.region_graph.algorithms import QuadGraph, QuadTree


class LearnSPN:
    def __init__(
        self,
        alpha: float = 0.5,
        image_shape: Tuple[int, int] = (1, 28, 28),
        seed: Optional[int] = 42,
        noise_scale: float = 0.2,            # === OPT 4: era 1e-1/2.0; ora piccolo rompi-simmetria
        use_miwae: bool = False,
        weight_dir: str = None,
        device: Optional[torch.device] = None,
        data_format: str = None,
        # --- nuovi knob ---
        diversify: str = "bootstrap",        # === OPT 3: 'bootstrap' | 'subcluster' | 'replicate'
        leaf_pool: str = "subset",           # === OPT 6a: 'subset' (originale) | 'global'
        adaptive_alpha: bool = True,         # === OPT 6b: massa di prior totale = alpha
    ):

        assert data_format in ('image', 'tabular'), "data_format should be either 'image' or 'tabular'"
        assert diversify in ('bootstrap', 'subcluster', 'replicate')
        assert leaf_pool in ('subset', 'global')

        self.alpha = alpha
        self.use_miwae = use_miwae
        self.noise_scale = noise_scale
        self.image_shape = image_shape
        self.data_format = data_format

        self.diversify = diversify
        self.leaf_pool = leaf_pool
        self.adaptive_alpha = adaptive_alpha
        self._all_rows: Optional[LongTensor] = None  # popolato in _estimate_parameters

        self.device = device if device is not None else (torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        if seed is not None:
            self._set_seed(seed)

        if data_format == 'image':
            assert len(image_shape) == 3, "image_shape should be (C, H, W)"

            if use_miwae:
                _, H, W = image_shape
                self.coords = {i: (i // W, i % W) for i in range(H * W)}
                self.miwae = ConvVAE(input_channel=1, latent_dim=50).to(self.device)
                if weight_dir is not None:
                    self.miwae.load_state_dict(torch.load(weight_dir, map_location=self.device))

    def _set_seed(self, seed: int):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # ------------------------------------------------------------------ #
    # Helper di parametrizzazione (OPT 1 / OPT 4)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _clamp_floor() -> float:
        # stesso valore dell'originale: sqrt(tiny) ~ 1.08e-19 per float32
        return float(np.sqrt(np.finfo(np.float32).tiny))

    def _activation_kwargs(self, activation: str) -> dict:
        if activation == 'positive-clamp':
            return {'vmin': self._clamp_floor()}
        return {}

    def _to_preactivation(self, probs: np.ndarray, activation: str) -> np.ndarray:
        """=== OPT 1: porta le probabilita' nello spazio pre-attivazione coerente.

        activation(_to_preactivation(p)) ~= p, per ogni attivazione supportata.
        """
        p = np.clip(np.asarray(probs, dtype=float), 1e-12, None)
        if activation == 'positive-clamp':
            # clamp(p, min=floor) = p, perche' p > 0  -> init preservata, nessuna unita' morta
            return p
        if activation == 'softplus':
            # inverso esatto: softplus(log(expm1(p))) = p
            return np.log(np.expm1(p))
        # 'softmax', 'none' (ed 'exp', se mai venisse aggiunta): log-prob
        return np.log(p)

    def _apply_symmetry_breaking(self, theta: np.ndarray, activation: str) -> np.ndarray:
        """=== OPT 4: rumore piccolo, applicato nello spazio giusto."""
        s = self.noise_scale
        if not s or s <= 0.0:
            return theta
        if activation == 'positive-clamp':
            # spazio lineare: rumore moltiplicativo positivo -> resta sopra il floor
            noisy = theta * np.exp(np.random.normal(loc=0.0, scale=s, size=theta.shape))
            return np.clip(noisy, self._clamp_floor(), None)
        # spazio log: rumore additivo
        return theta + np.random.normal(loc=0.0, scale=s, size=theta.shape)

    def _alpha_per_bin(self, num_bins: int) -> float:
        """=== OPT 6b: massa di prior totale = alpha (non alpha*num_bins)."""
        if self.adaptive_alpha:
            return self.alpha / max(num_bins, 1)
        return self.alpha

    # ------------------------------------------------------------------ #
    # API principale
    # ------------------------------------------------------------------ #

    def learn_spn(
        self,
        data: LongTensor,
        region_graph: str = 'quad-graph',
        input_layer: str = 'categorical',
        activation: str = 'softmax',
        weights_init: str = 'normal',
        sum_product_layer='cp',
        sum_weight_param: Optional[Parameterization] = None,
        num_input_units: int = 1,
        num_sum_units: int = 1,
        num_classes: int = 1,
        use_estimated_weights: bool = True,
        use_mixing_weights: bool = True,
    ) -> Circuit:

        # === OPT 1: 'normal' (media 0) e' patologica per il clamp nel percorso NON stimato
        #            (~meta' dei pesi finisce sul floor con gradiente nullo).
        if activation == 'positive-clamp' and weights_init == 'normal':
            weights_init = 'uniform'   # init positivo: clamp e' un no-op all'inizio

        assert weights_init in ('normal', 'uniform', 'dirichlet', 'None'), (
            "weights_init should be 'normal', 'uniform', 'dirichlet' or 'None'"
        )

        if region_graph == 'quad-graph':
            rg = QuadGraph(self.image_shape)
        elif region_graph == 'quad-tree-2':
            rg = QuadTree(self.image_shape, num_patch_splits=2)
        elif region_graph == 'quad-tree-4':
            rg = QuadTree(self.image_shape, num_patch_splits=4)
        else:
            raise ValueError(f"Unknown region graph called {region_graph}")

        activation_dict = {}
        nary_sum_weight_factory: ParameterFactory
        num_categories = int(data.max().item() + 1)
        input_factory = name_to_input_layer_factory(input_layer, num_categories=num_categories)

        if activation == 'positive-clamp':
            activation_dict['vmin'] = self._clamp_floor()

        if sum_weight_param is None:
            sum_weight_param = Parameterization(
                activation=activation,
                initialization=weights_init,
                activation_kwargs=activation_dict,
            )
        sum_weight_factory = parameterization_to_factory(sum_weight_param)

        if use_mixing_weights:
            nary_sum_weight_factory = functools.partial(
                mixing_weight_factory,
                param_factory=sum_weight_factory,
            )
        else:
            nary_sum_weight_factory = sum_weight_factory

        sc = rg.build_circuit(
            input_factory=input_factory,
            sum_product=sum_product_layer,
            sum_weight_factory=sum_weight_factory,
            nary_sum_weight_factory=nary_sum_weight_factory,
            num_input_units=num_input_units,
            num_sum_units=num_sum_units,
            num_classes=num_classes,
            factorize_multivariate=True,
        )

        if use_estimated_weights:
            sc = self._estimate_parameters(sc, data, activation=activation)

        return sc

    # ------------------------------------------------------------------ #
    # Stima dei parametri (BFS sul circuito)
    # ------------------------------------------------------------------ #

    def _estimate_parameters(
        self,
        sc: Circuit,
        data: LongTensor,
        activation: str,
    ) -> Circuit:

        visited = set()
        self._all_rows = torch.arange(data.size(0), device=self.device, dtype=torch.long)
        queue = deque([(out, self._all_rows) for out in sc.outputs])

        while queue:
            layer, rows_idx = queue.popleft()

            if layer in visited:
                continue
            visited.add(layer)

            layer_in = sc.layer_inputs(layer)
            layer_out = sc.layer_outputs(layer)

            if isinstance(layer, InputLayer):
                scope = list(layer.scope)

                param = self._make_input_param_estimated(
                    feat_idx=int(scope[0]),
                    instance_ids=rows_idx,
                    data=data,
                    num_input_units=layer.num_output_units,
                    num_categories=layer.num_categories,
                    activation=activation,
                )

                layer.probs = param

            elif isinstance(layer, SumLayer):
                # === OPT 7: i mixing layer NON vanno sovrascritti ne' clusterizzati.
                # Sono una mixture su decomposizioni alternative della stessa scope:
                # si lascia l'init strutturato (MixingWeightParameter) del factory e
                # si passano le righe intere ai figli.
                if isinstance(layer.weight.output, MixingWeightParameter):
                    for child in layer_in:
                        if child not in visited:
                            queue.append((child, rows_idx))
                    continue

                feat_ids = torch.tensor(list(sc._scopes[layer]), dtype=torch.long, device=data.device)
                clusters = self._cluster_instances(feat_ids, rows_idx, data, len(layer_in))

                param = self._make_sum_param_estimated(
                    clusters=clusters,
                    rows_idx=rows_idx,
                    num_input_units=layer.num_input_units,
                    num_sum_units=(1 if layer_out is None else layer.num_output_units),
                    activation=activation,
                )

                layer.weight = param

                for child, cluster_ids in zip(layer_in, clusters):
                    if child not in visited:
                        queue.append((child, cluster_ids))
            else:
                for child in layer_in:
                    if child not in visited:
                        queue.append((child, rows_idx))

        return sc

    # ------------------------------------------------------------------ #
    # Clustering (invariato rispetto all'originale)
    # ------------------------------------------------------------------ #

    def _cluster_instances(
        self,
        feat_ids: LongTensor,
        instance_ids: LongTensor,
        data: Tensor,
        n_clusters: int = 2,
        mode: str = "euclidean",
    ) -> List[LongTensor]:

        if instance_ids.numel() == 0:
            return [instance_ids.new_empty((0,), dtype=torch.long) for _ in range(n_clusters)]

        kmeans = KMeans(n_clusters=n_clusters, mode=mode, verbose=0)

        if self.use_miwae:
            C, H, W = self.image_shape
            N = instance_ids.numel()
            sub = data.index_select(0, instance_ids)
            imgs = torch.zeros((N, C, H, W), device=self.device, dtype=torch.float32)

            for feat_idx in feat_ids.tolist():
                y, x = self.coords[feat_idx]
                imgs[:, 0, y, x] = sub[:, feat_idx].float() / 255.0

            with torch.no_grad():
                mu, log_var, _, _ = self.miwae.encoder(imgs)
                embeddings = torch.cat([mu, log_var], dim=1).detach()

            labels = kmeans.fit_predict(embeddings)
        else:
            sub = data.index_select(0, instance_ids).index_select(1, feat_ids).float()
            labels = kmeans.fit_predict(sub)

        clusters: List[LongTensor] = []
        for c in range(n_clusters):
            mask = (labels == c)
            clusters.append(instance_ids[mask])

        return clusters

    # ------------------------------------------------------------------ #
    # Stima delle distribuzioni di input (foglie)
    # ------------------------------------------------------------------ #

    def _estimate_marginal(
        self,
        rows: LongTensor,
        data: LongTensor,
        feat_idx: int,
        num_categories: int,
    ) -> np.ndarray:
        """Marginale categorica smussata di un pixel su un insieme di righe."""
        if rows is None or len(rows) == 0:
            return np.full(num_categories, 1.0 / num_categories, dtype=float)
        col = data[rows, feat_idx]
        counts = torch.bincount(col, minlength=num_categories).float().cpu().numpy()
        counts = counts + self._alpha_per_bin(num_categories)   # === OPT 6b
        return counts / counts.sum()

    def _leaf_pool_rows(self, instance_ids: LongTensor) -> LongTensor:
        """=== OPT 6a: pool di righe per stimare la base della foglia."""
        if self.leaf_pool == 'global' and self._all_rows is not None:
            return self._all_rows
        return instance_ids

    def _per_unit_distributions(
        self,
        pool: LongTensor,
        data: LongTensor,
        feat_idx: int,
        num_units: int,
        num_categories: int,
    ) -> np.ndarray:
        """=== OPT 3: K marginali DIVERSE (una per unita'), invece di una replicata."""
        # caso scalare o replica esplicita: comportamento originale (tile)
        if num_units == 1 or self.diversify == 'replicate':
            base = self._estimate_marginal(pool, data, feat_idx, num_categories)
            return np.tile(base.reshape(1, -1), (num_units, 1))

        rows = pool.detach().cpu().numpy()
        out = np.empty((num_units, num_categories), dtype=float)

        if rows.size == 0:
            out[:] = 1.0 / num_categories
            return out

        for k in range(num_units):
            # 'bootstrap' (default) e 'subcluster' (hook) producono entrambi un
            # sottoinsieme diverso per ogni unita'; qui usiamo il bootstrap, che e'
            # robusto anche su una sola feature. Per 'subcluster' si potrebbe
            # clusterizzare il pool in `num_units` gruppi e dare a k il suo gruppo.
            samp = rows[np.random.randint(0, rows.size, size=rows.size)]
            sub = torch.as_tensor(samp, dtype=torch.long, device=data.device)
            out[k] = self._estimate_marginal(sub, data, feat_idx, num_categories)

        return out

    def _make_input_param_estimated(
        self,
        feat_idx: int,
        instance_ids: LongTensor,
        data: LongTensor,
        num_input_units: int,
        num_categories: int,
        activation: str,
    ) -> Parameter:

        pool = self._leaf_pool_rows(instance_ids)                                    # OPT 6a
        per_unit_probs = self._per_unit_distributions(                               # OPT 3
            pool=pool, data=data, feat_idx=feat_idx,
            num_units=num_input_units, num_categories=num_categories,
        )
        theta = self._to_preactivation(per_unit_probs, activation)                   # OPT 1
        theta = self._apply_symmetry_breaking(theta, activation)                     # OPT 4

        tp = TensorParameter(
            num_input_units,
            num_categories,
            initializer=ConstantTensorInitializer(theta),
            learnable=True,
        )
        unary_op_factory = name_to_parameter_activation(activation, **self._activation_kwargs(activation))
        return Parameter.from_unary(unary_op_factory((num_input_units, num_categories)), tp)

    # ------------------------------------------------------------------ #
    # Stima dei pesi di somma
    # ------------------------------------------------------------------ #

    def _cluster_mixture_weights(self, clusters: List[LongTensor]) -> np.ndarray:
        sizes = np.array([int(c.numel()) for c in clusters], dtype=float)
        sizes = sizes + self._alpha_per_bin(len(clusters))   # === OPT 6b
        total = sizes.sum()
        if total <= 0:
            return np.full(len(clusters), 1.0 / len(clusters), dtype=float)
        return sizes / total

    def _row_cluster_labels(self, clusters: List[LongTensor], rows_idx: LongTensor) -> np.ndarray:
        """Etichetta di cluster per ogni riga (serve al bootstrap per-unita')."""
        label_of = {}
        for c, ids in enumerate(clusters):
            for i in ids.tolist():
                label_of[i] = c
        return np.array([label_of[int(i)] for i in rows_idx.tolist()], dtype=int)

    def _per_unit_mixtures(
        self,
        base_mix: np.ndarray,
        labels: np.ndarray,
        num_sum_units: int,
        arity: int,
    ) -> np.ndarray:
        """=== OPT 3: ogni unita' di somma riceve una mixture DIVERSA (bootstrap)."""
        if num_sum_units == 1 or self.diversify == 'replicate' or labels.size == 0:
            return np.tile(base_mix.reshape(1, -1), (num_sum_units, 1))

        a = self._alpha_per_bin(arity)
        out = np.empty((num_sum_units, arity), dtype=float)
        for k in range(num_sum_units):
            samp = labels[np.random.randint(0, labels.size, size=labels.size)]
            counts = np.bincount(samp, minlength=arity).astype(float) + a
            out[k] = counts / counts.sum()
        return out

    def _make_sum_param_estimated(
        self,
        clusters: List[LongTensor],
        rows_idx: LongTensor,
        num_input_units: int,
        num_sum_units: int,
        activation: str,
    ) -> Parameter:

        arity = len(clusters)
        base_mix = self._cluster_mixture_weights(clusters)   # (arity,), somma 1

        # caso scalare: comportamento originale
        if num_sum_units == 1 and num_input_units == 1:
            theta = self._to_preactivation(base_mix.reshape(1, arity), activation)
            theta = self._apply_symmetry_breaking(theta, activation)
            tp = TensorParameter(1, arity, initializer=ConstantTensorInitializer(theta), learnable=True)
            unary_op_factory = name_to_parameter_activation(activation, **self._activation_kwargs(activation))
            return Parameter.from_unary(unary_op_factory((1, arity)), tp)

        labels = self._row_cluster_labels(clusters, rows_idx)
        per_unit_mix = self._per_unit_mixtures(base_mix, labels, num_sum_units, arity)  # (K_o, arity)  OPT 3

        # Espande ogni peso di cluster sulle sue `num_input_units` copie (struttura a blocchi):
        # ciascuna copia prende base_mix[c]/num_input_units cosi' la riga somma alle proporzioni.
        expanded = np.repeat(per_unit_mix[:, :, None] / num_input_units, num_input_units, axis=2)
        expanded = expanded.reshape(num_sum_units, arity * num_input_units)

        theta = self._to_preactivation(expanded, activation)                  # OPT 1
        theta = self._apply_symmetry_breaking(theta, activation)              # OPT 4

        tp = TensorParameter(
            num_sum_units,
            arity * num_input_units,
            initializer=ConstantTensorInitializer(theta),
            learnable=True,
        )
        unary_op_factory = name_to_parameter_activation(activation, **self._activation_kwargs(activation))
        return Parameter.from_unary(unary_op_factory((num_sum_units, num_input_units * arity)), tp)


# ---------------------------------------------------------------------------
# === OPT 5  (fuori da questo file — vive in mnist_dataset_eval.py / config_mnist.yml)
#
#   Per dare al modello largo lo spazio di sfruttare la capacita':
#     - patience: 5 -> ~15-20            (config_mnist.yml)
#     - scheduler T_0: 1 -> step_per_epoca (~223)   # T_0=1 fa oscillare la LR ogni step
#     - weight_decay: 0.0 -> ~1e-3..1e-2  (come PIC)
#   La normalizzazione via partition function e' gia' corretta nel training loop
#   (`circuit(batch) - circuit_partition_function()`), quindi clamp/softplus
#   non normalizzati vanno bene cosi' come sono.
# ---------------------------------------------------------------------------
