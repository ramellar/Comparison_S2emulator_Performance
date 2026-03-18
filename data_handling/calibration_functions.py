import os
import awkward as ak
import numpy as np
from scipy.optimize import lsq_linear
from configs.config import EMU_CONFIG, CALIB_CONFIGS, STRATEGIES, PU0_CONFIG_FOR_SEQ


def derive_calibration(cluster, gen, mode,
                       bounds=(-np.inf, np.inf),
                       remove_layer1=False,
                       name=""):

    lower_bound, upper_bound = bounds

    # -----------------------
    # Prepare inputs
    # -----------------------

    layer_pt = ak.flatten(cluster.layer_pt, axis=1)

    if remove_layer1:
        layer_pt = layer_pt[:, 1:13]
    else:
        layer_pt = layer_pt[:, :13]

    cluster_np = np.asarray(layer_pt)
    gen_np = np.asarray(ak.flatten(gen.pt, axis=-1))

    # -----------------------
    # PU0 calibration
    # -----------------------
    if mode == "PU0":

        regression = lsq_linear(
            cluster_np,
            gen_np,
            bounds=(lower_bound, upper_bound),
            lsmr_tol='auto',
            method='bvls'
        )
        return regression.x

    # -----------------------
    # PU200 calibration
    # -----------------------
    elif mode == "PU200":

        cluster_eta = ak.flatten(cluster.eta, axis=1)
        cluster_eta_np = -np.abs(np.asarray(cluster_eta))

        N = cluster_np.shape[0]

        A = np.hstack([
            cluster_np,
            cluster_eta_np[:, None],
            -1 * np.ones((N, 1))
        ])

        regression = lsq_linear(
            A,
            gen_np,
            bounds=(lower_bound, upper_bound),
            lsmr_tol='auto',
            method='bvls'
        )
        return regression.x

    # -----------------------
    # PU eta calibration
    # -----------------------
    elif mode == "PU_eta":

        cluster_flat = ak.flatten(getattr(cluster, f"Ecalib_{name}"), axis=-1)
        gen_flat = ak.flatten(gen.pt, axis=-1)

        residual = np.asarray(gen_flat) - np.asarray(cluster_flat)

        cluster_eta = ak.flatten(cluster.eta, axis=1)
        cluster_eta_np = -np.abs(np.asarray(cluster_eta))

        N = cluster_eta_np.shape[0]

        A = np.hstack([
            cluster_eta_np[:, None],
            -1 * np.ones((N, 1))
        ])

        regression = lsq_linear(
            A,
            residual,
            bounds=(lower_bound, upper_bound),
            lsmr_tol='auto',
            method='bvls'
        )
        return regression.x

def apply_calibration(cluster,
                      weights_layer=None,
                      weights_eta=None,
                      remove_layer1=False,
                      name=""):
    
    # -----------------------
    # Flatten inputs
    # -----------------------
    num_cluster = ak.num(cluster.layer_pt, axis=1)

    layer_pt = ak.flatten(cluster.layer_pt, axis=1)
    eta = ak.flatten(cluster.eta, axis=1)

    layer_np = ak.to_numpy(layer_pt)
    eta_np = ak.to_numpy(eta)

    # -----------------------
    # Select layers
    # -----------------------
    if remove_layer1:
        layer_np = layer_np[:, 1:13]
    else:
        layer_np = layer_np[:, :13]

    # -----------------------
    # Applying wl: per layer calibration
    # -----------------------
    if weights_layer is not None:
        weights_layer = np.asarray(weights_layer)
        E = np.sum(layer_np * weights_layer, axis=1)
    else:
        E = np.sum(layer_np, axis=1)  # fallback

    # -----------------------
    # Applying a and b : eta correction
    # -----------------------
    if weights_eta is not None:
        w_eta = weights_eta[0]
        bias = weights_eta[1]

        E = E - np.abs(eta_np) * w_eta - bias

    E = ak.unflatten(E, num_cluster)

    cluster = ak.with_field(cluster, E, f"Ecalib_{name}")

    return cluster

def save_weights(weight, base_dir, filename):
    
    path = os.path.join(base_dir, filename)
    os.makedirs(base_dir, exist_ok=True)

    ak.to_parquet(ak.Array(weight), path)


class CalibrationManager:

    def __init__(self, results_PU0, results_PU200, output_dir, configs, args):
        self.results_PU0 = results_PU0
        self.results_PU200 = results_PU200
        self.output_dir = output_dir
        self.configs = configs

    # -----------------------
    # Load weights
    # -----------------------
    def load_weight(self, filename):
        path = os.path.join(self.output_dir, filename)
        return ak.to_numpy(ak.from_parquet(path))
    

    def load(self, strategy, config_name, key):

        if strategy == "PU0":
            wl = self.load_weight(f"PU0_wl_{config_name}_{key}.parquet")
            return {"layer": wl}

        elif strategy == "PU200":
            w_all = self.load_weight(f"PU200_all_{config_name}_{key}.parquet")
            return {"all": w_all}

        elif strategy == "PU200_seq":
            if PU0_CONFIG_FOR_SEQ is None:
                PU0_cfg_name = config_name
            else:
                PU0_cfg_name = PU0_CONFIG_FOR_SEQ
            wl = self.load_weight(f"PU0_wl_{config_name}_{key}.parquet")
            ab = self.load_weight(f"PU200_seq_ab_{config_name}_with_PU0_{PU0_cfg_name}_{key}.parquet")
            return {"layer": wl, "eta": ab}

    # -----------------------
    # Apply calibration
    # -----------------------
    def apply(self, cluster, weights, strategy, remove_layer1, name):

        if strategy == "PU0":
            return apply_calibration(
                cluster,
                weights_layer=weights["layer"],
                remove_layer1=remove_layer1,
                name=name
            )

        elif strategy == "PU200":

            w = weights["all"]

            if len(w) == 15:
                w_layer = w[:13]
                w_eta = w[13:]
            else:
                w_layer = w[:12]
                w_eta = w[12:]

            return apply_calibration(
                cluster,
                weights_layer=w_layer,
                weights_eta=w_eta,
                remove_layer1=remove_layer1,
                name=name
            )

        elif strategy == "PU200_seq":

            return apply_calibration(
                cluster,
                weights_layer=weights["layer"],
                weights_eta=weights["eta"],
                remove_layer1=remove_layer1,
                name=name
            )

    # -----------------------
    # Get dataset
    # -----------------------
    # def get_data(self, strategy, key):

    #     if strategy == "PU0":
    #         return self.results_PU0[key]
    #     else:
    #         return self.results_PU200[key]

    def get_data(self, args, key):
      if args.pileup == "PU0":
        return self.results_PU0[key]
      elif args.pileup == "PU200":
        return self.results_PU200[key]


    def get_calibrated_cluster(self, strategy, config_name, key, args, name="test"):

        cfg = self.configs[config_name]

        weights = self.load(strategy, config_name, key)

        data = self.get_data(args, key)
        cluster = data["pair_cluster"]
        gen = data["pair_gen"]

        cluster_calib = self.apply(
            cluster,
            weights,
            strategy,
            remove_layer1=cfg["remove_layer1"],
            name=name
        )

        return cluster_calib, gen, weights