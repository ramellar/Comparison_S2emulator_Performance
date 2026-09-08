# Tau workflow: comandi raggruppati per parser

Questa guida raccoglie i comandi per il workflow `TauTau` del repository. I
comandi vanno eseguiti dalla directory principale del progetto:

```bash
cd /home/llr/cms/pivato/Comparison_S2emulator_Performance
```

Il workflow usa normalmente:

```text
ROOT ntuples -> load_data --tau -> Parquet -> matching_test
             -> stitch_matching_parquets -> apply_calib
             -> run_new_perfomance_plots
```

## 0. Ambiente

Attivare l'ambiente Conda usato dagli script:

```bash
conda activate s2_emulator
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
```

Per i file ROOT su EOS serve anche un proxy valido:

```bash
export X509_USER_PROXY=/percorso/al/proxy.pem
voms-proxy-info -all
```

I percorsi di input e output sono definiti in `configs/config.py`, in
particolare `PARQUET_BASE`.

## 1. Parser `scripts.load_data`

Carica gli ntuple ROOT, seleziona i tau hadronici nell'endcap e salva i dati in
Parquet. Per il workflow Tau e' obbligatorio il flag `--tau`.

### Comando locale di test

```bash
python3 -m scripts.load_data \
  -n 100 \
  --particles TauTau \
  --pileup PU200 \
  --base_path /percorso/alla/directory/root \
  --n_files 1 \
  --job_id 0 \
  --n_jobs 1 \
  --tau
```

### Opzioni del parser

| Opzione | Default | Funzione |
|---|---:|---|
| `-n` | `1` | Numero di eventi da leggere |
| `--particles` | `photons` | Nome del sample; per Tau usare `TauTau` |
| `--pileup` | `PU0` | Scenario di pileup, normalmente `PU200` |
| `--base_path` | path locale configurato nello script | Directory contenente i ROOT |
| `--name_tree` | `l1tHGCalTriggerNtuplizer/HGCalTriggerNtuple` | Nome del TTree |
| `--pt_cut` | `0` | Soglia pT dei cluster |
| `--n_files` | `10` | Numero di file ROOT disponibili |
| `--job_id` | `0` | Indice del job |
| `--n_jobs` | `1` | Numero totale di job |
| `--tau` | disattivo | Abilita la selezione e le variabili Tau |

Ogni job scrive i file in:

```text
<PARQUET_BASE>/TauTau_PU200_new_branch/single_jobs/
```

### Caricamento su HTCondor

`submit_load.sub` e' gia' impostato per `TauTau`, `PU200`, `10` job e il flag
`--tau`:

```bash
condor_submit submit_load.sub
```

Prima dell'invio controllare `ProxyPath`, `--n`, `--n_files`, `--n_jobs` e il
numero della `queue`. Il wrapper `run_load_data.sh` interpreta il primo
argomento come proxy; se si usa il wrapper direttamente:

```bash
./run_load_data.sh "$X509_USER_PROXY" \
  -n 100 --particles TauTau --pileup PU200 --n_files 1 --tau
```

Per recuperare job falliti, aggiornare gli ID in `recovery.sub` e lanciare:

```bash
condor_submit recovery.sub
```

## 2. Parser `data_handling.stich_parquets`

Unisce i Parquet prodotti dai job di caricamento in file completi.

```bash
python3 -m data_handling.stich_parquets \
  --particles TauTau \
  --pileup PU200 \
  --n_files 333 \
  --n_jobs 10
```

Opzioni principali: `-n`, `--particles`, `--pileup`, `--base_path`,
`--name_tree`, `--pt_cut`, `--n_files`, `--job_id`, `--n_jobs`.

## 3. Parser `scripts.matching_test`

Esegue il matching tra tau generati e cluster, oppure ricostruisce prima i jet
anti-$k_t$ e usa quelli per il matching.

### Matching locale

```bash
python3 -m scripts.matching_test \
  --particles TauTau \
  --pileup PU200 \
  --gen_pt_cut 20.0 \
  --matching_type antikt_jets \
  --deltaR 0.2
```

### Matching diviso in job

```bash
python3 -m scripts.matching_test \
  --particles TauTau \
  --pileup PU200 \
  --gen_pt_cut 20.0 \
  --matching_type antikt_jets \
  --job_id 0 \
  --n_jobs 20 \
  --n_events 99900
```

| Opzione | Default | Funzione |
|---|---:|---|
| `--particles` | `photons` | Sample di input |
| `--pileup` | `PU0` | Pileup di input |
| `--pt_cut` | `0` | Soglia pT cluster |
| `--gen_pt_cut` | `0` | Soglia pT del tau generato |
| `--deltaR` | `0.2` | Distanza massima per il matching |
| `--matching_type` | `gen_cluster` | `gen_cluster` oppure `antikt_jets` |
| `--total_efficiency` | disattivo | Calcola l'efficienza integrata |
| `--only_efficiency` | disattivo | Usa risultati di matching gia' salvati |
| `--job_id` | `0` | Indice del job |
| `--n_jobs` | `10` | Numero totale di job |
| `--n_events` | tutti | Limita gli eventi prima della divisione |

### Matching su HTCondor

`submit_matching.sub` e' configurato per `TauTau`, `PU200`, anti-$k_t$, `20`
job e `gen_pt_cut=20`:

```bash
condor_submit submit_matching.sub
```

## 4. Parser `data_handling.stitch_matching_parquets`

Dopo che tutti i job di matching sono terminati, unisce i risultati presenti in
`parts/`:

```bash
python3 -m data_handling.stitch_matching_parquets \
  --particles TauTau \
  --pileup PU200 \
  --gen_pt_cut 20.0
```

Per sostituire file gia' presenti:

```bash
python3 -m data_handling.stitch_matching_parquets \
  --particles TauTau --pileup PU200 --gen_pt_cut 20.0 --overwrite
```

Opzioni: `--particles`, `--pileup`, `--pt_cut`, `--gen_pt_cut` e
`--overwrite`.

## 5. Parser `scripts.apply_calib`

Applica la calibrazione Tau ai cluster matched. I file di matching devono
essere gia' stati uniti.

### Calibrazione offset

```bash
python3 -m scripts.apply_calib \
  --particles TauTau \
  --pileup PU200 \
  --gen_pt_cut 20.0 \
  --calibration offset
```

### Calibrazione offset + MC

```bash
python3 -m scripts.apply_calib \
  --particles TauTau \
  --pileup PU200 \
  --gen_pt_cut 20.0 \
  --calibration MC
```

Il parser accetta `--particles`, `--pileup`, `--pt_cut`, `--gen_pt_cut` e
`--calibration {offset,MC}`. Lo script aggiunge `pt_corrected` ai Parquet
`pair_cluster_*_matched.parquet` e salva i fit Tau nella stessa directory.

## 6. Parser `scripts.run_new_perfomance_plots`

Genera i plot usando i risultati raw, offset o MC. Per i risultati prodotti da
`apply_calib.py` usare `--pt_type offset` oppure `--pt_type mc`.

### Plot di base

```bash
python3 -m scripts.run_new_perfomance_plots \
  --particles TauTau --pileup PU200 \
  --gen_pt_cut 20.0 --matching_type antikt_jets \
  --matched --resolution_plots --pt_type offset
```

### Plot utili per Tau

```bash
# Distribuzioni pT/eta/phi per decay mode
python3 -m scripts.run_new_perfomance_plots \
  --particles TauTau --pileup PU200 --gen_pt_cut 20.0 \
  --matched --distributions_per_decaymode --pt_type offset

# Risposta per decay mode
python3 -m scripts.run_new_perfomance_plots \
  --particles TauTau --pileup PU200 --gen_pt_cut 20.0 \
  --matched --response_per_decaymode --pt_type offset

# Profili per decay mode
python3 -m scripts.run_new_perfomance_plots \
  --particles TauTau --pileup PU200 --gen_pt_cut 20.0 \
  --matched --profile_per_decaymode --pt_type offset

# Molteplicita' di cluster per decay mode
python3 -m scripts.run_new_perfomance_plots \
  --particles TauTau --pileup PU200 --gen_pt_cut 20.0 \
  --matched --mult_decaymode --pt_type raw

# Diagnostica anti-kT e calibrazione beta(|eta|)
python3 -m scripts.run_new_perfomance_plots \
  --particles TauTau --pileup PU200 --gen_pt_cut 20.0 \
  --matched --diagnostic --pt_type offset
```

Plot di efficienza e distribuzioni generiche usano rispettivamente
`--efficiency`, `--distribution`, `--scale_distribution`, `--two_d_dist`,
`--binned_distributions` e `--n_clusters_plots`. I dataset selezionabili sono
`--matched`, `--events` e `--filtered_events`.

Le opzioni complete si possono vedere con:

```bash
python3 -m scripts.run_new_perfomance_plots --help
```

Nota: questo script usa `--pt_type` con underscore; `run_calibration_plots`
usa invece `--pt-type` con trattino.

## 7. Parser `scripts.run_calibration_plots`

Confronta le strategie di calibrazione definite in `configs/config.py`. Prima
selezionare le configurazioni in `COMPARISONS`.

```bash
python3 -m scripts.run_calibration_plots \
  --particles TauTau \
  --pileup PU200 \
  --gen_pt_cut 20.0 \
  --triangle 0p03 \
  --matched \
  --resolution_plots \
  --pt-type corrected \
  --tag TauTau_PU200
```

Per tutti i triangoli usare `--all_triangles`. I plot disponibili sono
`--distribution`, `--scale_distribution`, `--resolution_plots`,
`--binned_distributions`, `--weights`, `--eta_residual` e `--weight_table`.
Le strategie si limitano con, ad esempio:

```bash
python3 -m scripts.run_calibration_plots \
  --particles TauTau --pileup PU200 --gen_pt_cut 20.0 \
  --all_triangles --matched --resolution_plots \
  --strategies raw PU200_bounds --tag TauTau_comparison
```

Per l'elenco completo dei parser:

```bash
python3 -m scripts.run_calibration_plots --help
```

## 8. Parser `scripts.derive_calibration`

Questo parser deriva i coefficienti delle calibrazioni generiche definite in
`CALIB_CONFIGS` e `STRATEGIES` in `configs/config.py`.

```bash
python3 -m scripts.derive_calibration \
  --particles TauTau \
  --pileup PU200 \
  --gen_pt_cut 20.0 \
  --pt_cut 0 \
  --offset 0
```

Opzioni: `--particles`, `--pileup`, `--tag`, `--gen_pt_cut`, `--pt_cut` e
`--offset`.

Per la calibrazione Tau descritta nella sezione 5 usare invece
`scripts.apply_calib`, che implementa direttamente le calibrazioni `offset` e
`MC` sui risultati matched.

## Ordine consigliato completo

```bash
# 1. Caricamento locale o condor
condor_submit submit_load.sub

# 2. Merge dei Parquet di caricamento
python3 -m data_handling.stich_parquets \
  --particles TauTau --pileup PU200 --n_files 333 --n_jobs 10

# 3. Matching locale oppure condor
condor_submit submit_matching.sub

# 4. Merge dei risultati di matching
python3 -m data_handling.stitch_matching_parquets \
  --particles TauTau --pileup PU200 --gen_pt_cut 20.0

# 5. Calibrazione
python3 -m scripts.apply_calib \
  --particles TauTau --pileup PU200 --gen_pt_cut 20.0 --calibration offset

# 6. Plot sui pT corretti
python3 -m scripts.run_new_perfomance_plots \
  --particles TauTau --pileup PU200 --gen_pt_cut 20.0 \
  --matched --resolution_plots --pt_type offset
```

Per verificare rapidamente la sintassi di ogni parser senza eseguire la
pipeline:

```bash
python3 -m scripts.load_data --help
python3 -m scripts.matching_test --help
python3 -m data_handling.stich_parquets --help
python3 -m data_handling.stitch_matching_parquets --help
python3 -m scripts.apply_calib --help
python3 -m scripts.run_new_perfomance_plots --help
python3 -m scripts.run_calibration_plots --help
python3 -m scripts.derive_calibration --help
```