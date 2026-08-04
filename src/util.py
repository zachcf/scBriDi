import torch
import torch.nn as nn
import numpy as np
import random
import torch.nn.functional as F
import math
import scanpy as sc
import anndata as ad
from sklearn import metrics
from torch.utils.data import Dataset
import scipy
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import scipy.sparse as sp
from scipy.spatial import cKDTree


class MultiModalDataset(Dataset):
    def __init__(self, data_dict, data_dict2):
        self.data_dict = data_dict
        self.data_dict2 = data_dict2
        self.xtypes = list(data_dict.keys())
        lengths = [len(data_dict[key]) for key in self.xtypes]
        assert len(set(lengths)) == 1, f"Data lengths are inconsistent: {lengths}"
        self.length = lengths[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {type: (self.data_dict[type][idx], self.data_dict2[type][idx]) for type in self.xtypes}


class UnpairedMultiModalDataset(Dataset):
    def __init__(self, feats, scaled_feats, cell_type_key='cell_type', match_method='spatial_aware',
                 sampling_strategy='upsample', sup_info=None, n_cluster=20):
        self.feats = feats
        self.scaled_feats = scaled_feats
        self.modalities = list(feats.keys())
        self.cell_type_key = cell_type_key
        self.match_method = match_method
        self.sampling_strategy = sampling_strategy
        self.sup_info = sup_info or {}
        self.n_cluster = n_cluster
        self.spatial_modality = self._identify_spatial_modality()
        self.scrna_modality = [m for m in self.modalities if m != self.spatial_modality][0] if self.spatial_modality else None
        self.cell_type_indices = self._create_cell_type_indices()
        if self.spatial_modality and self.spatial_modality in self.sup_info and 'coords' in self.sup_info[self.spatial_modality]:
            self.spatial_regions = self._create_spatial_regions()
        else:
            self.spatial_regions = None
        self.matched_pairs = self._create_matched_pairs()

    def _identify_spatial_modality(self):
        for mod in self.modalities:
            if mod in self.sup_info and 'coords' in self.sup_info[mod]:
                return mod
            if mod == "spatial":
                return mod
        return None

    def _create_cell_type_indices(self):
        import scanpy as sc
        indices_map = {}
        for mod in self.modalities:
            indices_map[mod] = {}
            if mod in self.sup_info and 'labels' in self.sup_info[mod]:
                labels = self.sup_info[mod]['labels']
                for cell_type in np.unique(labels):
                    indices_map[mod][cell_type] = np.where(labels == cell_type)[0]
                print(f"From sup_info read '{mod}' labels, total {len(np.unique(labels))} types")
            else:
                features = self.feats[mod].numpy() if isinstance(self.feats[mod], torch.Tensor) else self.feats[mod]
                temp_adata = sc.AnnData(features)
                sc.pp.neighbors(temp_adata)
                sc.tl.leiden(temp_adata, resolution=0.4)
                cluster_labels = temp_adata.obs['leiden'].astype(int).values
                n_clusters = len(np.unique(cluster_labels))
                self.n_cluster = n_clusters
                print(f"{mod} clustering done, total {n_clusters} clusters")
                for i in range(n_clusters):
                    indices_map[mod][f"cluster_{i}"] = np.where(cluster_labels == i)[0]
        return indices_map

    def _create_spatial_regions(self):
        from sklearn.cluster import KMeans
        coords = self.sup_info[self.spatial_modality]['coords']
        kmeans = KMeans(n_clusters=self.n_cluster, random_state=42)
        region_labels = kmeans.fit_predict(coords)
        regions = {}
        for i in range(self.n_cluster):
            regions[i] = np.where(region_labels == i)[0]
        return {'labels': region_labels, 'regions': regions, 'centers': kmeans.cluster_centers_}

    def _create_matched_pairs(self):
        pairs = []
        if self.match_method == 'label_based':
            print("Use label-based matching")
            pairs = self._create_label_based_pairs()
        elif self.match_method == 'spatial_label_rna_cluster':
            print("Use spatial label-RNA cluster matching")
            pairs = self._create_spatial_label_rna_cluster_pairs()
        elif self.match_method == 'cluster_based':
            print("Use cluster-based matching")
            pairs = self._create_cluster_based_pairs()
        elif self.match_method == 'spatial_aware' and self.spatial_regions is not None:
            print("Use spatial-aware matching")
            pairs = self._create_spatial_aware_pairs()
        else:
            print("Use random matching")
            pairs = self._create_random_pairs()
        print(f"Created {len(pairs)} pairs")
        return pairs

    def _create_label_based_pairs(self):
        pairs = []
        common_cell_types = set.intersection(*[set(ct for ct in self.cell_type_indices[mod].keys()
                                                   if 'cluster_' not in ct)
                                               for mod in self.modalities])
        print("common_cell_types", common_cell_types)
        for cell_type in common_cell_types:
            mod_indices = {mod: self.cell_type_indices[mod][cell_type] for mod in self.modalities}
            n_samples = self._determine_sample_size(
                [len(indices) for indices in mod_indices.values()]
            )
            sampled_indices = {}
            for mod, indices in mod_indices.items():
                sampled_indices[mod] = np.random.choice(indices, n_samples,
                                                        replace=(len(indices) < n_samples))
            for i in range(n_samples):
                pair = {mod: sampled_indices[mod][i] for mod in self.modalities}
                pairs.append(pair)
        return pairs

    def _create_spatial_label_rna_cluster_pairs(self):
        if not self.spatial_modality or not self.scrna_modality:
            print("Cannot use spatial label-RNA cluster matching: missing spatial or RNA modality")
            return self._create_cluster_based_pairs()

        spatial_mod = self.spatial_modality
        scrna_mod = self.scrna_modality
        pairs = []

        if spatial_mod in self.sup_info and 'coords' in self.sup_info[spatial_mod]:
            spatial_coords = self.sup_info[spatial_mod]['coords']
        else:
            print("No spatial coordinates found, ignoring spatial position")
            spatial_coords = None

        spatial_labels = [k for k in self.cell_type_indices[spatial_mod].keys() if 'cluster_' not in k]
        spatial_centers = {}
        for label in spatial_labels:
            indices = self.cell_type_indices[spatial_mod][label]
            if isinstance(self.feats[spatial_mod], torch.Tensor):
                center = self.feats[spatial_mod][indices].mean(dim=0).numpy()
            else:
                center = self.feats[spatial_mod][indices].mean(axis=0)
            spatial_centers[label] = center

        spatial_locations = {}
        spatial_dispersion = {}
        for label in spatial_labels:
            indices = self.cell_type_indices[spatial_mod][label]
            if spatial_coords is not None:
                mean_pos = np.mean(spatial_coords[indices], axis=0)
                spatial_locations[label] = mean_pos
                cell_positions = spatial_coords[indices]
                dists_to_center = np.linalg.norm(cell_positions - mean_pos, axis=1)
                spatial_dispersion[label] = np.mean(dists_to_center)

        scrna_clusters = [k for k in self.cell_type_indices[scrna_mod].keys() if 'cluster_' in k]
        scrna_centers = {}
        for cluster in scrna_clusters:
            indices = self.cell_type_indices[scrna_mod][cluster]
            if isinstance(self.feats[scrna_mod], torch.Tensor):
                center = self.feats[scrna_mod][indices].mean(dim=0).numpy()
            else:
                center = self.feats[scrna_mod][indices].mean(axis=0)
            scrna_centers[cluster] = center

        n_spatial = len(spatial_labels)
        n_scrna = len(scrna_clusters)
        similarity_matrix = np.zeros((n_spatial, n_scrna))

        spatial_distance_matrix = None
        if spatial_coords is not None and spatial_locations:
            spatial_distance_matrix = np.zeros((n_spatial, n_spatial))
            for i, label1 in enumerate(spatial_labels):
                for j, label2 in enumerate(spatial_labels):
                    if i != j:
                        base_dist = np.linalg.norm(spatial_locations[label1] - spatial_locations[label2])
                        dispersion_factor = (spatial_dispersion[label1] + spatial_dispersion[label2]) / 2
                        adjusted_dist = max(0, base_dist - dispersion_factor * 0.5)
                        spatial_distance_matrix[i, j] = adjusted_dist

        for i, label in enumerate(spatial_labels):
            for j, cluster in enumerate(scrna_clusters):
                v1 = spatial_centers[label]
                v2 = scrna_centers[cluster]
                cosine_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                if len(v1) == len(v2):
                    corr_sim = np.corrcoef(v1, v2)[0, 1]
                    if np.isnan(corr_sim): corr_sim = cosine_sim
                else:
                    corr_sim = cosine_sim
                euclidean_dist = np.sqrt(np.sum((v1 - v2) ** 2))
                euclidean_sim = 1 / (1 + euclidean_dist)
                expr_similarity = 0.8 * cosine_sim + 0.1 * corr_sim + 0.1 * euclidean_sim
                final_similarity = expr_similarity
                if spatial_distance_matrix is not None:
                    spatial_consistency = 0
                    total_weight = 0
                    for k, other_label in enumerate(spatial_labels):
                        if k != i:
                            distance = spatial_distance_matrix[i, k]
                            distance_weight = 1 / (1 + distance)
                            total_weight += distance_weight
                            other_best_cluster_idx = np.argmax(similarity_matrix[k])
                            if other_best_cluster_idx == j:
                                neighbor_sim = similarity_matrix[k, j]
                                spatial_consistency += distance_weight * neighbor_sim
                    if total_weight > 0:
                        spatial_consistency /= total_weight
                    final_similarity = 0.8 * expr_similarity + 0.2 * spatial_consistency
                similarity_matrix[i, j] = final_similarity

        from scipy.optimize import linear_sum_assignment
        cost_matrix = 1 - similarity_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        sim_mean, sim_std = np.mean(similarity_matrix), np.std(similarity_matrix)
        adaptive_threshold = max(0, sim_mean - 0.8 * sim_std)
        print(f"Adaptive threshold: {adaptive_threshold:.4f}")

        label_cluster_mapping = {}
        for i, j in zip(row_ind, col_ind):
            spatial_label = spatial_labels[i]
            scrna_cluster = scrna_clusters[j]
            sim_score = similarity_matrix[i, j]
            if sim_score > adaptive_threshold:
                label_cluster_mapping[spatial_label] = scrna_cluster
                print(f"Match: {spatial_label} -> {scrna_cluster} (similarity: {sim_score:.4f})")

        for spatial_label, scrna_cluster in label_cluster_mapping.items():
            spatial_indices = self.cell_type_indices[spatial_mod][spatial_label]
            scrna_indices = self.cell_type_indices[scrna_mod][scrna_cluster]
            n_samples = self._determine_sample_size([len(spatial_indices), len(scrna_indices)])
            sampled_spatial = np.random.choice(spatial_indices, n_samples,
                                               replace=(len(spatial_indices) < n_samples))
            sampled_scrna = np.random.choice(scrna_indices, n_samples,
                                             replace=(len(scrna_indices) < n_samples))
            for i in range(n_samples):
                pair = {spatial_mod: sampled_spatial[i], scrna_mod: sampled_scrna[i]}
                pairs.append(pair)

        print(f"Spatial label-RNA cluster matching: created {len(pairs)} cell pairs")
        return pairs

    def _create_cluster_based_pairs(self):
        pairs = []
        cluster_centers = {}
        mod_clusters = {mod: list(self.cell_type_indices[mod].keys()) for mod in self.modalities}
        for mod in self.modalities:
            centers = {}
            for cluster_id, indices in self.cell_type_indices[mod].items():
                features = self.scaled_feats[mod][indices].mean(dim=0).numpy() if isinstance(self.scaled_feats[mod],
                                                                                              torch.Tensor) else \
                    self.scaled_feats[mod][indices].mean(axis=0)
                centers[cluster_id] = features
            cluster_centers[mod] = centers

        mod1, mod2 = self.modalities[0], self.modalities[1]
        similarity_matrix = np.zeros((len(mod_clusters[mod1]), len(mod_clusters[mod2])))
        for i, cluster1 in enumerate(mod_clusters[mod1]):
            for j, cluster2 in enumerate(mod_clusters[mod2]):
                v1 = cluster_centers[mod1][cluster1]
                v2 = cluster_centers[mod2][cluster2]
                cosine_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                corr_sim = np.corrcoef(v1, v2)[0, 1] if len(v1) == len(v2) else cosine_sim
                if np.isnan(corr_sim): corr_sim = cosine_sim
                euclidean_dist = np.sqrt(np.sum((v1 - v2) ** 2))
                euclidean_sim = 1 / (1 + euclidean_dist)
                similarity = 0.7 * cosine_sim + 0.2 * corr_sim + 0.1 * euclidean_sim
                similarity_matrix[i, j] = similarity

        from scipy.optimize import linear_sum_assignment
        cost_matrix = 1 - similarity_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        sim_mean, sim_std = np.mean(similarity_matrix), np.std(similarity_matrix)
        adaptive_threshold = max(0.15, sim_mean - 0.8 * sim_std)
        cluster_mapping = {}
        for i, j in zip(row_ind, col_ind):
            cluster1 = mod_clusters[mod1][i]
            cluster2 = mod_clusters[mod2][j]
            sim_score = similarity_matrix[i, j]
            if sim_score > adaptive_threshold:
                size1 = len(self.cell_type_indices[mod1][cluster1])
                size2 = len(self.cell_type_indices[mod2][cluster2])
                size_ratio = min(size1, size2) / max(size1, size2)
                if size_ratio > 0.15:
                    cluster_mapping[cluster1] = (cluster2, mod2)

        for cluster1, (cluster2, _) in cluster_mapping.items():
            indices1 = self.cell_type_indices[mod1][cluster1]
            indices2 = self.cell_type_indices[mod2][cluster2]
            n_samples = self._determine_sample_size([len(indices1), len(indices2)])
            sampled_indices1 = np.random.choice(indices1, n_samples, replace=(len(indices1) < n_samples))
            sampled_indices2 = np.random.choice(indices2, n_samples, replace=(len(indices2) < n_samples))
            for i in range(n_samples):
                pair = {mod1: sampled_indices1[i], mod2: sampled_indices2[i]}
                pairs.append(pair)
        return pairs

    def _create_spatial_aware_pairs(self):
        if not self.spatial_modality or not self.spatial_regions:
            print("Cannot use spatial-aware matching: missing spatial info")
            return self._create_cluster_based_pairs()

        pairs = []
        scrna_mod = self.scrna_modality
        spatial_mod = self.spatial_modality

        spatial_centers = []
        region_indices_list = []
        for region_id, region_indices in self.spatial_regions['regions'].items():
            if len(region_indices) == 0:
                continue
            region_indices_list.append(region_indices)
            if isinstance(self.feats[spatial_mod], torch.Tensor):
                region_expr = self.feats[spatial_mod][region_indices].mean(dim=0).numpy()
            else:
                region_expr = self.feats[spatial_mod][region_indices].mean(axis=0)
            spatial_centers.append(region_expr)
        spatial_centers = np.array(spatial_centers)

        scrna_centers = []
        scrna_cluster_indices = []
        for cluster_id, cluster_indices in self.cell_type_indices[scrna_mod].items():
            scrna_cluster_indices.append(cluster_indices)
            if isinstance(self.feats[scrna_mod], torch.Tensor):
                cluster_expr = self.feats[scrna_mod][cluster_indices].mean(dim=0).numpy()
            else:
                cluster_expr = self.feats[scrna_mod][cluster_indices].mean(axis=0)
            scrna_centers.append(cluster_expr)
        scrna_centers = np.array(scrna_centers)

        n_spatial = len(spatial_centers)
        n_scrna = len(scrna_centers)
        similarity_matrix = np.zeros((n_spatial, n_scrna))
        for i in range(n_spatial):
            for j in range(n_scrna):
                spatial_center = spatial_centers[i]
                scrna_center = scrna_centers[j]
                cosine_sim = np.dot(spatial_center, scrna_center) / (
                        np.linalg.norm(spatial_center) * np.linalg.norm(scrna_center))
                if len(spatial_center) == len(scrna_center):
                    corr_sim = np.corrcoef(spatial_center, scrna_center)[0, 1]
                    if np.isnan(corr_sim): corr_sim = cosine_sim
                else:
                    corr_sim = cosine_sim
                euclidean_dist = np.sqrt(np.sum((spatial_center - scrna_center) ** 2))
                euclidean_sim = 1 / (1 + euclidean_dist)
                similarity = 0.6 * cosine_sim + 0.3 * corr_sim + 0.1 * euclidean_sim
                similarity_matrix[i, j] = similarity

        from scipy.optimize import linear_sum_assignment
        cost_matrix = 1 - similarity_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        sim_mean, sim_std = np.mean(similarity_matrix), np.std(similarity_matrix)
        adaptive_threshold = max(0.15, sim_mean - 0.8 * sim_std)

        matched_count = 0
        valid_similarities = []
        for spatial_idx, scrna_idx in zip(row_ind, col_ind):
            region_indices = region_indices_list[spatial_idx]
            cluster_indices = scrna_cluster_indices[scrna_idx]
            similarity = similarity_matrix[spatial_idx, scrna_idx]
            if similarity < adaptive_threshold:
                continue
            size1 = len(region_indices)
            size2 = len(cluster_indices)
            size_ratio = min(size1, size2) / max(size1, size2)
            if size_ratio < 0.15:
                continue
            valid_similarities.append(similarity)
            matched_count += 1
            n_samples = self._determine_sample_size([len(region_indices), len(cluster_indices)])
            sampled_spatial = np.random.choice(region_indices, n_samples, replace=(len(region_indices) < n_samples))
            sampled_scrna = np.random.choice(cluster_indices, n_samples, replace=(len(cluster_indices) < n_samples))
            for i in range(n_samples):
                pair = {spatial_mod: sampled_spatial[i], scrna_mod: sampled_scrna[i]}
                pairs.append(pair)

        avg_similarity = np.mean(valid_similarities) if valid_similarities else 0
        print(f"Spatial region-RNA cluster matching done, total {len(row_ind)} matches, adaptive threshold: {adaptive_threshold:.4f}")
        print(f"After filtering, kept {matched_count} high-quality matches, avg similarity: {avg_similarity:.4f}")
        return pairs

    def _create_random_pairs(self):
        pairs = []
        mod_lengths = {mod: len(self.feats[mod]) for mod in self.modalities}
        if self.sampling_strategy == 'upsample':
            n_samples = max(mod_lengths.values())
        elif self.sampling_strategy == 'downsample':
            n_samples = min(mod_lengths.values())
        else:
            n_samples = int(np.mean(list(mod_lengths.values())))
        n_samples = min(n_samples, 10000)
        for _ in range(n_samples):
            pair = {}
            for mod in self.modalities:
                pair[mod] = np.random.randint(0, mod_lengths[mod])
            pairs.append(pair)
        return pairs

    def _determine_sample_size(self, sizes_list):
        if self.sampling_strategy == 'upsample':
            return max(sizes_list)
        elif self.sampling_strategy == 'downsample':
            return min(sizes_list)
        else:
            return int(np.mean(sizes_list))

    def __len__(self):
        return len(self.matched_pairs)

    def __getitem__(self, idx):
        pair = self.matched_pairs[idx]
        data = {}
        for mod in self.modalities:
            index = pair[mod]
            data[mod] = (self.feats[mod][index], self.scaled_feats[mod][index])
        return data


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def target_distribution(batch: torch.Tensor, epsilon=1e-8) -> torch.Tensor:
    weight = (batch ** 2) / (torch.sum(batch, 0) + epsilon)
    p = (weight.t() / (torch.sum(weight, 1) + epsilon)).t()
    return p / p.sum(dim=1, keepdim=True)


def dec_loss(q, p):
    return F.kl_div(p.log(), q, reduction='batchmean')


def alignment_loss(q1, q2):
    return (F.kl_div(q1.log(), q2, reduction='batchmean') +
            F.kl_div(q2.log(), q1, reduction='batchmean')) / 2


def five_fold_split_dataset(
        RNA_data,
        seed=19193
):
    if not seed is None:
        setup_seed(seed)
    temp = [i for i in range(len(RNA_data.obs_names))]
    random.shuffle(temp)
    id_list = []
    test_count = int(0.2 * len(temp))
    validation_count = int(0.16 * len(temp))
    for i in range(5):
        test_id = temp[: test_count]
        validation_id = temp[test_count: test_count + validation_count]
        train_id = temp[test_count + validation_count:]
        temp.extend(test_id)
        temp = temp[test_count:]
        id_list.append([train_id, validation_id, test_id])
    return id_list


class DynamicHybridLoss(nn.Module):
    def __init__(self, total_steps, phase_steps, device, modality_dims=None, temp=0.1, phase=None, n_clusters=30):
        super().__init__()
        self.step = 0
        self.total_steps = total_steps
        self.phase_steps = phase_steps
        self.phase_order = ['pretrain', 'warmup', 'full']
        self.device = device
        self.temp = nn.Parameter(torch.tensor([temp])).to(device)
        self.phases = phase if phase else {
            'pretrain': {'contrast': 0.0, "intra": 0.0},
            'warmup': {'contrast': 0.3, "intra": 0.5},
            'full': {'contrast': 0.7, "intra": 1}
        }
        self.current_phase = 'pretrain'
        self.modality_dims = modality_dims
        self.num_mods = len(modality_dims) if modality_dims else 0
        self.log_vars = nn.Parameter(torch.zeros(self.num_mods))
        self.cluster_centers = {}
        if modality_dims is not None:
            self.proj_adapters = nn.ModuleDict({
                mod: nn.Linear(dim, 128).to(self.device) for mod, dim in modality_dims.items()
            })
        for mod, dim in modality_dims.items():
            init_tensor = torch.Tensor(n_clusters, dim)
            nn.init.xavier_normal_(init_tensor)
            device_tensor = init_tensor.to(device)
            self.cluster_centers[mod] = nn.Parameter(device_tensor)
        self.birgde_mod = "rna"
        if "spatial" in modality_dims.keys(): self.birgde_mod = "spatial"
        self.ifdec = False

    def forward(self, recon_losses, z_p_dict, z_dict=None):
        contrast_loss = 0.0
        intra_loss = 0.0
        de_loss = 0.0
        num_mods = self.num_mods
        modalities = list(z_p_dict.keys())
        phase = self._get_current_phase()
        weights = self._get_phase_weights(phase)
        total_loss = 0

        if self.birgde_mod in modalities:
            birgde_z = F.normalize(z_p_dict[self.birgde_mod], dim=-1)
            for mod in modalities:
                if mod != self.birgde_mod:
                    mod_z = F.normalize(z_p_dict[mod], dim=-1)
                    contrast_loss += self.contrastive_loss(mod_z, birgde_z, symmetrical=False)
        else:
            for i in range(num_mods):
                for j in range(i + 1, num_mods):
                    zi = F.normalize(z_p_dict[modalities[i]], dim=-1)
                    zj = F.normalize(z_p_dict[modalities[j]], dim=-1)
                    contrast_loss += self.contrastive_loss(zi, zj)

        if self.modality_dims is not None:
            for mod in modalities:
                z_proj = F.normalize(z_p_dict[mod], dim=-1)
                z_orig = F.normalize(z_dict[mod], dim=-1)
                intra_loss += self.modality_invariant_loss(
                    z_orig,
                    z_proj,
                    self.proj_adapters[mod]
                )

        if self.ifdec:
            qs = {}
            alpha = 1
            for mod in modalities:
                norm_squared = torch.sum((z_dict[mod].unsqueeze(1) - self.cluster_centers[self.birgde_mod]) ** 2, 2)
                numerator = 1.0 / (1.0 + (norm_squared / alpha))
                power = float(alpha + 1) / 2
                numerator = numerator ** power
                q = numerator / torch.sum(numerator, dim=1, keepdim=True)
                qs[mod] = q
                p = target_distribution(q).detach()
                de_loss += dec_loss(q, p)

            if self.birgde_mod in modalities:
                bridge_q = qs[self.birgde_mod]
                for mod in modalities:
                    if mod != self.birgde_mod:
                        de_loss += alignment_loss(qs[mod], bridge_q)
            else:
                _q = qs[modalities[0]]
                for mod in modalities:
                    if mod != modalities[0]:
                        de_loss += alignment_loss(qs[mod], _q)

        for i, loss in enumerate(recon_losses):
            total_loss += (torch.exp(-self.log_vars[i]) * loss + torch.log(1 + torch.exp(self.log_vars[i])))

        total_loss += weights['contrast'] * contrast_loss
        total_loss += weights['intra'] * intra_loss
        total_loss += weights['dec'] * de_loss

        self.step += 1
        return {
            "total": total_loss,
            "recon": sum(recon_losses),
            "contrast": contrast_loss,
            "intra": intra_loss,
            "dec": de_loss
        }

    def _get_current_phase(self):
        accum_steps = 0
        for phase in self.phase_order:
            if self.step < accum_steps + self.phase_steps.get(phase, 0):
                return phase
            accum_steps += self.phase_steps.get(phase, 0)
        return 'full'

    def _get_phase_weights(self, phase):
        if phase == 'pretrain':
            return self.phases[phase]

        prev_steps = sum([self.phase_steps[p] for p in self.phase_order if
                          p != phase and self.phase_order.index(p) < self.phase_order.index(phase)])
        phase_progress = (self.step - prev_steps) / self.phase_steps[phase]
        phase_progress = max(0.0, min(1.0, phase_progress))

        base_weights = self.phases[phase]
        if phase == 'warmup':
            return self.phases[phase]
        else:
            warmup_end_contrast = self.phases['warmup']['contrast']
            warmup_end_intra = self.phases['warmup']['intra']
            warmup_end_dec = self.phases['warmup']['dec']
            full_target_contrast = base_weights['contrast']
            full_target_intra = base_weights['intra']
            full_target_dec = base_weights['dec']
            cosine_progress = (1 - math.cos(phase_progress * math.pi)) / 2
            contrast = warmup_end_contrast + (full_target_contrast - warmup_end_contrast) * cosine_progress
            intra = warmup_end_intra + (full_target_intra - warmup_end_intra) * cosine_progress
            dec = warmup_end_dec + (full_target_dec - warmup_end_dec) * cosine_progress
        return {'contrast': contrast, 'intra': intra, 'dec': dec}

    def contrastive_loss(self, z1, z2, symmetrical=True):
        temp = torch.clamp(self.temp, min=1e-4, max=100.0)
        sim = torch.mm(z1, z2.T) / temp
        labels = torch.arange(z1.size(0), device=z1.device)
        loss_i = F.cross_entropy(sim, labels)
        if symmetrical:
            loss_j = F.cross_entropy(sim.T, labels)
            loss = (loss_i + loss_j) / 2
        else:
            loss = loss_i
        return loss

    def modality_invariant_loss(self, z_orig, z_proj, adapter):
        z_orig = adapter(z_orig)
        mse_loss = F.mse_loss(z_orig, z_proj)
        cosine_loss = 1 - F.cosine_similarity(z_orig, z_proj).mean()
        return 0.7 * cosine_loss + 0.3 * mse_loss


def map_rna_to_spatial(rna_data, spatial_data, similarity_method='hybrid',
                                coord_key=["x", "y"], add_jitter=True, density_aware=False,
                                avoid_overlap=True,use_spatial_neighbors=True,
                                   neighbor_weight=0.2):


    rna_matrix = rna_data.X.toarray() if scipy.sparse.issparse(rna_data.X) else rna_data.X
    spatial_matrix = spatial_data.X.toarray() if scipy.sparse.issparse(spatial_data.X) else spatial_data.X
    if 'X_pca' in rna_data.obsm and 'X_pca' in spatial_data.obsm:
        rna_matrix = rna_data.obsm['X_pca']
        spatial_matrix = spatial_data.obsm['X_pca']
    else:
        rna_matrix = rna_data.X.toarray() if scipy.sparse.issparse(rna_data.X) else rna_data.X
        spatial_matrix = spatial_data.X.toarray() if scipy.sparse.issparse(spatial_data.X) else spatial_data.X

    n_rna = rna_matrix.shape[0]
    n_spatial = spatial_matrix.shape[0]
    avg_cells_per_spot = n_rna / n_spatial


    chunk_size = min(5000, n_rna)
    k_keep = min(1000, n_spatial)

    rna_top_k_indices = np.zeros((n_rna, k_keep), dtype=np.int32)
    rna_top_k_scores = np.zeros((n_rna, k_keep), dtype=np.float32)

    n_chunks = (n_rna + chunk_size - 1) // chunk_size
    spatial_coords = np.array(spatial_data.obs[coord_key])

    if use_spatial_neighbors:

            k_neighbors = 20
            sp_tree = cKDTree(spatial_coords)
            sp_distances, sp_neighbor_indices = sp_tree.query(spatial_coords, k=k_neighbors+1)

            sigma_spatial = np.median(sp_distances[:, 1]) * 1.5
            

            adaptive_weights = np.zeros((n_spatial, k_neighbors), dtype=np.float32)
            
            for i in range(n_spatial):
                neighbors = sp_neighbor_indices[i, 1:]
                neighbor_dists = sp_distances[i, 1:]
                

                w_spatial = np.exp(- (neighbor_dists ** 2) / (2 * sigma_spatial ** 2))
                

                expr_i = spatial_matrix[i:i+1]
                expr_neighbors = spatial_matrix[neighbors]
                
                sim_expr = cosine_similarity(expr_i, expr_neighbors)[0]
                

                w_expr = np.maximum(sim_expr, 0.01) ** 2  

                w_combined = w_spatial * w_expr

                adaptive_weights[i] = w_combined / (w_combined.sum() + 1e-10)

    tree = cKDTree(spatial_coords)
    for chunk_id in range(n_chunks):
        start = chunk_id * chunk_size
        end = min(start + chunk_size, n_rna)

        if similarity_method == 'cosine':
            base_sim = cosine_similarity(rna_matrix[start:end], spatial_matrix).astype(np.float32)
        elif similarity_method == 'correlation':
            chunk = rna_matrix[start:end]
            chunk_centered = chunk - chunk.mean(axis=1, keepdims=True)
            spatial_centered = spatial_matrix - spatial_matrix.mean(axis=1, keepdims=True)
            chunk_norm = np.linalg.norm(chunk_centered, axis=1, keepdims=True)
            spatial_norm = np.linalg.norm(spatial_centered, axis=1, keepdims=True)
            base_sim = ((chunk_centered @ spatial_centered.T) /
                        (chunk_norm @ spatial_norm.T + 1e-10)).astype(np.float32)
        else:  # hybrid
            cos_sim = cosine_similarity(rna_matrix[start:end], spatial_matrix)
            chunk = rna_matrix[start:end]
            chunk_centered = chunk - chunk.mean(axis=1, keepdims=True)
            spatial_centered = spatial_matrix - spatial_matrix.mean(axis=1, keepdims=True)
            chunk_norm = np.linalg.norm(chunk_centered, axis=1, keepdims=True)
            spatial_norm = np.linalg.norm(spatial_centered, axis=1, keepdims=True)
            corr_sim = (chunk_centered @ spatial_centered.T) / (chunk_norm @ spatial_norm.T + 1e-10)


            base_sim = (0.8 * cos_sim + 0.2 * corr_sim).astype(np.float32)

        if use_spatial_neighbors:
            neighbor_sim = np.zeros_like(base_sim)
            
            for i in range(n_spatial):
                neighbors = sp_neighbor_indices[i, 1:]
                weights = adaptive_weights[i]

                neighbor_sim[:, i] = np.dot(base_sim[:, neighbors], weights)

            chunk_sim = (1 - neighbor_weight) * base_sim + neighbor_weight * neighbor_sim
        else:
            chunk_sim = base_sim

        # Keep Top-K
        top_k_idx = np.argpartition(-chunk_sim, k_keep-1, axis=1)[:, :k_keep]
        for i in range(chunk_sim.shape[0]):
            sorted_idx = top_k_idx[i][np.argsort(-chunk_sim[i, top_k_idx[i]])]
            rna_top_k_indices[start + i] = sorted_idx
            rna_top_k_scores[start + i] = chunk_sim[i, sorted_idx]

    spatial_coords = np.array(spatial_data.obs[coord_key])
    spatial_attraction = np.zeros(n_spatial, dtype=np.float32)

    for i in range(n_rna):
        scores = rna_top_k_scores[i]
        
        scores_shifted = np.maximum(scores - scores.min(), 1e-6)
        probs = scores_shifted / scores_shifted.sum()

        spatial_attraction[rna_top_k_indices[i]] += probs

    smooth_factor = 0.3
    uniform_attraction = np.ones(n_spatial) * avg_cells_per_spot
    
    expected_quotas = (1 - smooth_factor) * spatial_attraction + smooth_factor * uniform_attraction

    min_quota = max(2, int(avg_cells_per_spot * 0.4))
    max_quota = int(avg_cells_per_spot * 2.5)
    
    expected_quotas = np.clip(expected_quotas, min_quota, max_quota)
    quotas = np.floor(expected_quotas).astype(np.int32)
    remainder = n_rna - quotas.sum()

    if remainder > 0:

        fractional_parts = expected_quotas - quotas
        # Only add to spots not already at max
        valid_mask = quotas < max_quota
        fractional_parts[~valid_mask] = -1.0  
        
        top_indices = np.argsort(-fractional_parts)[:remainder]
        quotas[top_indices] += 1
        
    elif remainder < 0:

        excess = -remainder
        for _ in range(excess):
            valid_mask = quotas > min_quota
            if not valid_mask.any():
                break

            valid_indices = np.where(valid_mask)[0]
            max_idx = valid_indices[np.argmax(quotas[valid_mask])]
            quotas[max_idx] -= 1
    
    import heapq
    from collections import deque, defaultdict

    assignments = np.full(n_rna, -1, dtype=np.int32)
    
    proposal_idx = np.zeros(n_rna, dtype=np.int32)

    unassigned_queue = deque(range(n_rna))

    spot_accepted = defaultdict(list)

    exhausted_cells = []
    
    while unassigned_queue:
        rna_idx = unassigned_queue.popleft()
        idx = proposal_idx[rna_idx]
        
        if idx < k_keep:
            target_spot = rna_top_k_indices[rna_idx, idx]
            score = rna_top_k_scores[rna_idx, idx]
            
            proposal_idx[rna_idx] += 1
            
            spot_quota = quotas[target_spot]
            heap = spot_accepted[target_spot]
            
            if len(heap) < spot_quota:
  
                heapq.heappush(heap, (score, rna_idx))
            else:

                min_score, min_rna_idx = heap[0]
                if score > min_score:
                    heapq.heappop(heap)
                    heapq.heappush(heap, (score, rna_idx))
  
                    unassigned_queue.append(min_rna_idx)
                else:

                    unassigned_queue.append(rna_idx)
        else:

            exhausted_cells.append(rna_idx)

  
    current_counts = np.zeros(n_spatial, dtype=np.int32)
    for spot_idx, heap in spot_accepted.items():
        for score, rna_idx in heap:
            assignments[rna_idx] = spot_idx
            current_counts[spot_idx] += 1


    if exhausted_cells:

        global_available = np.where(current_counts < quotas)[0]
        avail_idx = 0
        for rna_idx in exhausted_cells:

            while current_counts[global_available[avail_idx]] >= quotas[global_available[avail_idx]]:
                avail_idx += 1
            best_spot = global_available[avail_idx]
            assignments[rna_idx] = best_spot
            current_counts[best_spot] += 1


    assert (assignments >= 0).all(), "Unassigned cells exist"
    assert np.array_equal(np.bincount(assignments, minlength=n_spatial), current_counts), "Count mismatch"


    sample_size = min(300, n_spatial)
    sample_idx = np.random.choice(n_spatial, sample_size, replace=False)
    nn_dists, _ = tree.query(spatial_coords[sample_idx], k=2)
    median_dist = np.median(nn_dists[:, 1])
    jitter_scale = median_dist * 0.35
    
    spot_groups = defaultdict(list)
    for rna_idx, sp_idx in enumerate(assignments):
        spot_groups[sp_idx].append(rna_idx)
    
    inferred_positions = np.zeros((n_rna, 2), dtype=np.float32)
    np.random.seed(42)
    
    for sp_idx, rna_list in spot_groups.items():
        base_pos = spatial_coords[sp_idx]
        n_cells = len(rna_list)
        
        if n_cells == 1:
            inferred_positions[rna_list[0]] = base_pos
        elif add_jitter and avoid_overlap:
            angles = np.linspace(0, 2*np.pi, n_cells, endpoint=False)
            angles += np.random.uniform(-0.15, 0.15, n_cells)
            
            radius_scale = np.sqrt(n_cells / 3)
            radii = jitter_scale * radius_scale * np.random.uniform(0.85, 1.15, n_cells)
            
            offsets = np.column_stack([np.cos(angles) * radii, np.sin(angles) * radii])
            inferred_positions[rna_list] = base_pos + offsets
        else:
            inferred_positions[rna_list] = base_pos

    rna_data.obs['x'] = inferred_positions[:, 0]
    rna_data.obs['y'] = inferred_positions[:, 1]
    rna_data.obs['assigned_spatial_idx'] = assignments
    
    return rna_data


def cluster_metrics(adata, reference_key="cell_type"):
    if reference_key not in adata.obs:
        raise ValueError(f"Reference label '{reference_key}' not found in adata.obs")

    if "X_pca" not in adata.obsm:
        sc.pp.pca(adata)
    if "neighbors" not in adata.uns or "connectivities" not in adata.obsp:
        sc.pp.neighbors(adata, n_neighbors=k)
    if "leiden" not in adata.obs:
        sc.tl.leiden(adata)

    y_true = np.asarray(adata.obs[reference_key])
    y_pred = np.asarray(adata.obs["leiden"])

    ARI = float(metrics.adjusted_rand_score(y_true, y_pred))
    NMI = float(metrics.normalized_mutual_info_score(y_true, y_pred))
    results = {
        "ARI": ARI,
        "NMI": NMI,
    }
    return results


def to_adata(x, xtype):

    if torch.is_tensor(x):
        if x.is_cuda:
            x = x.cpu()
        x = x.detach().numpy()


    if xtype != "adt":
        x[x < 1e-4] = 0
        x = sp.csr_matrix(x)
    else:
        pass

    return ad.AnnData(x)
