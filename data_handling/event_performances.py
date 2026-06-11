#Loading the events for the new emulator to plot the performance
import numpy as np
import awkward as ak
import uproot
from tqdm import tqdm
from data_handling.taureco import taureco_function
from data_handling.taureco import vis_filter_function
import subprocess
import os


def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    if iteration == total: 
        print()


def provide_events_performaces( n, base_path, particle, pileup, args, n_files=300 , thr=0.0, job_id=0, n_jobs=10, name_tree="l1tHGCalTriggerNtuplizer/HGCalTriggerNtuple"):
    full_base_path = f"{base_path}{particle}_{pileup}"
    
    #Job division
    all_indices = list(range(int(n_files)))
    avg = len(all_indices) / float(n_jobs)
    chunks = []
    last = 0.0
    while last < len(all_indices):
        chunks.append(all_indices[int(last):int(last + avg)])
        last += avg
     
    my_file_indices = chunks[job_id]
    files = [f"{full_base_path}/ntuple_{i}.root:{name_tree}" for i in my_file_indices]

    print(f"--- Job {job_id}/{n_jobs} starting: Processing {len(files)} files ---")
   
    branches = [
        "event",
        "genpart_exeta",
        "genpart_exphi",
        "genpart_pt",
        "genpart_gen",
        "genpart_reachedEE",

        "cl3d_p0113Tri_eta",
        "cl3d_p0113Tri_phi",
        "cl3d_p0113Tri_pt",
        'cl3d_p0113Tri_layer_pt',

        "cl3d_p016Tri_eta",
        "cl3d_p016Tri_phi",
        "cl3d_p016Tri_pt",
        'cl3d_p016Tri_layer_pt',

        "cl3d_p03Tri_eta",
        "cl3d_p03Tri_phi",
        "cl3d_p03Tri_pt",
        'cl3d_p03Tri_layer_pt',

        "cl3d_p045Tri_eta",
        "cl3d_p045Tri_phi",
        "cl3d_p045Tri_pt",
        'cl3d_p045Tri_layer_pt',

        "cl3d_Ref_eta",
        "cl3d_Ref_phi",
        "cl3d_Ref_pt",
        'cl3d_Ref_layer_pt',
        
        
        #Taus collection of all particles
        "gen_eta",
        "gen_phi",
        "gen_pt",
        "gen_energy",
        "gen_charge",   # xxx
        "gen_pdgid",
        "gen_status",
        "gen_daughters",
        
        #gentau collection
        "gentau_pt",
        "gentau_eta",
        "gentau_phi",
        "gentau_energy",
        "gentau_mass",
        "gentau_decayMode",
        "gentau_totNproducts",  
        "gentau_totNgamma",       
        "gentau_totNpiZero",      
        "gentau_totNcharged",     
        
        #gentau products/vis collections
        "gentau_products_pt",
        "gentau_products_eta",
        "gentau_products_phi",
        "gentau_products_energy",
        "gentau_products_mass",
        "gentau_products_id",
        
        "gentau_vis_pt",
        "gentau_vis_eta",
        "gentau_vis_phi",
        "gentau_vis_energy",
        "gentau_vis_mass",
    ]

    batches = []
    total = 0

    my_file_indices = chunks[job_id]
    raw_files = [f"{full_base_path}/ntuple_{i}.root" for i in my_file_indices]
    
    valid_files = []
    print(f"Checking existence of {len(raw_files)} files...")
    
    for f_path in raw_files:
        if os.path.exists(f_path):
            # File exists, add it to our list with the tree name
            valid_files.append(f"{f_path}:{name_tree}")
        else:
            print(f"Skipping missing file: {f_path}")


    for batch in tqdm(uproot.iterate(valid_files, branches, library="ak", step_size="50 MB", allow_missing=True)):
        batches.append(batch)
        total += len(batch["event"])
        if total >= n: break

    if not batches:
        raise RuntimeError(f"Job {job_id} could not read any data. Check your Proxy!")

    data = ak.concatenate(batches)[:n]
    event_id = ak.Array(np.arange(len(data.event)))

    print("Number of events:", len(data.event))

    # -----------------------
    # GEN selection
    # Mask: Only select particles with flag "gen-particle_gen" different from -1, 
    #       only keep the gens that reached the EE and 
    #       only keep the particles whose shower will stay inside the detector
    # -----------------------
    
    filtered_gen = None
    if args.tau: 
        
        m_reco, best_match_idx, mask_clean = taureco_function(data.gen_pt, data.gen_eta, data.gen_phi, data.gen_energy, data.gen_pdgid, data.gen_status, data.gen_daughters, data.gentau_pt, data.gentau_eta, data.gentau_phi, data.gentau_energy)
        flag_vis_prod, flag_pp, gen_flag, gvis_pt, gvis_eta, gvis_phi, gvis_energy, gvis_mass, gdecaymode = vis_filter_function(0.005, 0.005, m_reco, best_match_idx, mask_clean, data.gentau_decayMode, data.gentau_vis_pt, data.gentau_vis_eta, data.gentau_vis_phi, data.gentau_vis_energy, data.gentau_vis_mass, data.gentau_products_pt, data.gentau_products_eta, data.gentau_products_phi, data.gentau_products_energy, data.gentau_products_mass, data.gentau_products_id)
        
        #Mask for endcap eta  
        eta_mask = ((abs(gvis_eta) > 1.5) & (abs(gvis_eta) < 3.0))
        had_mask = ((gdecaymode == 0) | (gdecaymode == 1) | (gdecaymode == 4) | (gdecaymode == 5))

        gen_mask = eta_mask & had_mask
        
        filtered_gen = ak.zip({    
            "event_id": event_id,                  
            "flag_vis_prod": flag_vis_prod,      #flag for all the events where m_vis =! m_products
            "flag_pp": flag_pp,      #flag for all the events with pair production                          
            "pt": gvis_pt[gen_mask],      # filtered gentau_vis_pt
            "eta": gvis_eta[gen_mask],      # filtered gentau_vis_eta
            "phi": gvis_phi[gen_mask],      # filtered gentau_vis_phi
            "energy": gvis_energy[gen_mask],      # filtered gentau_vis_energy
            "mass": gvis_mass[gen_mask],  # filtered gentau_vis_mass
            "gen_flag": gen_flag[gen_mask],  # flag for the taus (1 for the first and 2 for the second)
            "gen_decayMode": gdecaymode[gen_mask]
        })
        
        
        event_mask = ak.num(filtered_gen.pt) > 0
        filtered_gen = filtered_gen[event_mask]   
            
        #I would like to add the "event" branch also to filtered_tau_vis but it has already been used (I'm not sure they are the same ones)
        
        #If I want to remove the events with pair production I can just set a mask on flag_pp (0 -> events with no pp; 1 -> events with pp)
        
        #flag_vis_prod is an array that flags all the events where the reconstructed invariant mass from gentau_products_*
        #collection differs from the one from gentau_vis_* more than a fixed parameter


        
    else:
        
        particle_mask = (
            (data.genpart_gen != -1)
            & (data.genpart_reachedEE == 2)
            & (abs(data.genpart_exeta) > 1.6)
            & (abs(data.genpart_exeta) < 2.9)
        )
        

        filtered_gen = ak.zip({
            "event": data.event,
            "eta": data.genpart_exeta[particle_mask],
            "phi": data.genpart_exphi[particle_mask],
            "pt": data.genpart_pt[particle_mask],
            "genpart_gen": data.genpart_gen[particle_mask],
            "genpart_reachedEE": data.genpart_reachedEE[particle_mask],
        })

        event_mask = ak.num(filtered_gen.pt) > 0

        filtered_gen = filtered_gen[event_mask]
        

    # clusters
    def build_clusters(prefix):

        pt = data[f"{prefix}_pt"]
        eta = data[f"{prefix}_eta"]
        phi = data[f"{prefix}_phi"]
        layer_pt = data[f"{prefix}_layer_pt"]

        if thr > 0:
            mask = pt > thr
            pt = pt[mask]
            eta = eta[mask]
            phi = phi[mask]
            layer_pt = layer_pt[mask]

        arr = ak.zip({
            "event": data.event,
            "eta": eta,
            "phi": phi,
            "pt": pt,
        })

        arr= ak.with_field(arr, layer_pt, "layer_pt")

        return arr[event_mask]

    cl_0p0113 = build_clusters("cl3d_p0113Tri")
    cl_0p016 = build_clusters("cl3d_p016Tri")
    cl_0p03 = build_clusters("cl3d_p03Tri")
    cl_0p045 = build_clusters("cl3d_p045Tri")
    cl_ref = build_clusters("cl3d_Ref")
    # print(cl_0p0113.eta)
    # print(cl_0p0113.layer_pt)
    # print(len(cl_0p0113.eta))
    # print(len(cl_0p0113.layer_pt))
    
    #print("FIELDS BEFORE RETURN:", ak.fields(filtered_gen))
    #print("TYPE BEFORE RETURN:", ak.type(filtered_gen))

 
    return filtered_gen, cl_0p0113, cl_0p016, cl_0p03, cl_0p045, cl_ref


def apply_matching(events, gen, args, deltaR=0.2):
    #Masking events were no clusters are reconstructed
    mask_ev = ak.num(getattr(events, "eta")) > 0
    filtered_events = events[mask_ev]
    filtered_gen = gen[mask_ev]
    
    #print("filtered_gen fields:", ak.fields(filtered_gen))
    #print("filtered_gen type:", ak.type(filtered_gen))

    if args.total_efficiency:
        print("Number of events with clusters", len(filtered_gen))

    if not args.total_efficiency and args.gen_pt_cut !=0:
        #Adding a cut on the gen particle pt 
        print("Applying gen pt cut")
        mask_pt_gen = filtered_gen.pt > args.gen_pt_cut
        filtered_pt_gen = filtered_gen[mask_pt_gen]
        mask_gen = ak.num(filtered_pt_gen.pt) > 0
        filtered_gen = filtered_pt_gen[mask_gen]
        filtered_events = filtered_events[mask_gen]

    # Building structured arrays
    clusters = ak.zip({
        "eta": getattr(filtered_events, "eta"),
        "phi": getattr(filtered_events, "phi"),
        "pt": getattr(filtered_events, "pt"),
    })

    layer_pt = getattr(filtered_events, "layer_pt")
    clusters = ak.with_field(clusters, layer_pt, "layer_pt")

    genparts = ak.zip({
        "eta": getattr(filtered_gen, 'eta'),
        "phi": getattr(filtered_gen, 'phi'),
        "pt": getattr(filtered_gen, 'pt'),
        "gen_flag": ak.local_index(filtered_gen.pt, axis=1) + 1,    # c'era questo prima: "gen_flag": getattr(filtered_gen,'gen_flag'),
        "decayMode": getattr(filtered_gen, 'gen_decayMode')
    })

    if args.pt_cut !=0 :
        if args.total_efficiency:
            print("\n")
        print("Applying pt cut on clusters of", args.pt_cut, "GeV")
        #Adding a cut on the cluster pt
        mask_pt_cluster = clusters.pt > args.pt_cut
        filtered_pt_cluster = clusters[mask_pt_cluster]
        # print(min(ak.flatten(filtered_pt_cluster.pt, axis=-1)))
        # if args.total_efficiency:
        #     print(f"Number of clusters with gen cut {args.gen_pt_cut} GeV ", len(ak.flatten(clusters.pt,axis=-1)))

        mask_empty_events = ak.num(filtered_pt_cluster.pt) > 0
        clusters= filtered_pt_cluster[mask_empty_events]
        genparts = genparts[mask_empty_events]
        if args.total_efficiency:
            print(f"Number of events with gen cut {args.gen_pt_cut} GeV and cluster pt cut {args.pt_cut} GeV ", len(clusters))
            print(f"Number of gen particles after gen cut {args.gen_pt_cut} GeV and cluster pt cut {args.pt_cut} GeV ", len(ak.flatten(genparts.pt,axis=-1)))
            print(f"Number of clusters with gen cut {args.gen_pt_cut} GeV and cluster pt cut {args.pt_cut} GeV ", len(ak.flatten(clusters.pt,axis=-1)))


    # Applying the deltaR matching 
    pairs = ak.cartesian([clusters, genparts], axis=1, nested=True)
    delta_eta = np.abs(pairs['0'].eta - pairs['1'].eta)
    delta_phi = np.abs(pairs['0'].phi - pairs['1'].phi)
    delta_phi = ak.where(delta_phi > np.pi, 2 * np.pi - delta_phi, delta_phi)

    delta_r = np.sqrt(delta_eta**2 + delta_phi**2)
    mask = delta_r < deltaR

    pair_cluster_masked=pairs['0'][mask]
    pair_gen_masked=pairs['1'][mask]

    if ak.any(ak.num(pair_cluster_masked, axis=2) == 2):
        print("2 gen particles matched to the same cluster")
    if ak.any(ak.num(pair_gen_masked, axis=2) == 2):
        print("2 gen particles matched to the same cluster")

    # Remove non matched clusters in each event 
    mask_empty= (ak.num(pair_cluster_masked, axis=2)==1) & (ak.num(pair_gen_masked, axis=2)==1)
    pair_cluster_masked = pair_cluster_masked[mask_empty]
    pair_gen_masked = pair_gen_masked[mask_empty]

    # Remove non matched events in the list 
    empty_mask = ak.num(pair_cluster_masked) == 0
    has_empty = ak.any(empty_mask)
    # print("Any empty entries?", has_empty) 
    if has_empty:
        pair_cluster_masked = pair_cluster_masked[~empty_mask]
        pair_gen_masked = pair_gen_masked[~empty_mask] 
        if args.total_efficiency:
            print("Removing empty entries from the arrays")

    # Counting the number of gen particles matched in each event (this can be either 1 or 2)
    flatten_pair_gen_masked= ak.flatten(pair_gen_masked, axis = -1)
    flatten_pair_cluster_masked= ak.flatten(pair_cluster_masked, axis = 2)
    nb_matched_gen_particles = (ak.any(flatten_pair_gen_masked.gen_flag == 2, axis=-1)) & (ak.any(flatten_pair_gen_masked.gen_flag == 1, axis=-1))
    nb_matched_gen_particles = ak.where(nb_matched_gen_particles == False, 1,2)
   

    # Selecting the matched cluster with the highest pt for each gen particle
    
    delta_r_flat = ak.values_astype(
    ak.flatten(delta_r[mask][mask_empty][~empty_mask], axis=2),
    "float64"
    )


    cluster_matched = ak.zip({
        "eta": flatten_pair_cluster_masked.eta ,
        "phi": flatten_pair_cluster_masked.phi,
        "pt": flatten_pair_cluster_masked.pt,
        "gen_flag": flatten_pair_gen_masked.gen_flag,
        "delta_r": delta_r_flat,
    })

    layer_pt=flatten_pair_cluster_masked.layer_pt
    cluster_matched = ak.with_field(cluster_matched, layer_pt, "layer_pt")

    gen_matched = ak.zip({
        "eta": flatten_pair_gen_masked.eta ,
        "phi": flatten_pair_gen_masked.phi,
        "pt": flatten_pair_gen_masked.pt,
        "gen_flag": flatten_pair_gen_masked.gen_flag,
        "gen_decayMode": flatten_pair_gen_masked.decayMode,
    })

    # If there is only one gen particle matched

    one_gen_matched_mask = nb_matched_gen_particles == 1
    one_cluster_matched = cluster_matched[one_gen_matched_mask]
    one_gen_matched = gen_matched[one_gen_matched_mask]
    sorted_one_cluster_matched = one_cluster_matched[ak.argsort(one_cluster_matched.pt, ascending=False)]
    best_cluster_one_matched = ak.firsts(sorted_one_cluster_matched)
    best_gen_one_matched = one_gen_matched[:, 0]

    layer_pt_2=best_cluster_one_matched.layer_pt

    best_cluster_one_matched = ak.zip({
        'eta': ak.unflatten(best_cluster_one_matched.eta,1),
        'phi': ak.unflatten(best_cluster_one_matched.phi,1),
        'pt': ak.unflatten(best_cluster_one_matched.pt,1),
        'gen_flag': ak.unflatten(best_cluster_one_matched.gen_flag,1),
        'delta_r': ak.unflatten(best_cluster_one_matched.delta_r,1),
    })

    layer_pt_wrapped = ak.unflatten(layer_pt_2, 1)
    best_cluster_one_matched = ak.with_field(best_cluster_one_matched, layer_pt_wrapped, "layer_pt")

    best_gen_one_matched = ak.zip({
        'eta': ak.unflatten(best_gen_one_matched.eta, 1),
        'phi': ak.unflatten(best_gen_one_matched.phi, 1),
        'pt': ak.unflatten(best_gen_one_matched.pt, 1),
        'gen_flag': ak.unflatten(best_gen_one_matched.gen_flag, 1),
        'gen_decayMode': ak.unflatten(best_gen_one_matched.gen_decayMode, 1),
    })

    # If there are two gen particles that were matched

    two_gen_matched_mask = nb_matched_gen_particles == 2
    two_cluster_matched=cluster_matched[two_gen_matched_mask]
    two_gen_matched=gen_matched[two_gen_matched_mask]

    number_one_matched = two_gen_matched.gen_flag == 1
    number_two_matched = two_gen_matched.gen_flag == 2
    two_cluster_1_gen_matched = two_cluster_matched[number_one_matched]
    two_cluster_2_gen_matched = two_cluster_matched[number_two_matched]
    two_gen_1_gen_matched= two_gen_matched[number_one_matched]
    two_gen_2_gen_matched= two_gen_matched[number_two_matched]

    sorted_clusters2_1 = two_cluster_1_gen_matched[ak.argsort(two_cluster_1_gen_matched.pt, ascending=False)]
    sorted_clusters2_2 = two_cluster_2_gen_matched[ak.argsort(two_cluster_2_gen_matched.pt, ascending=False)]
    best_cluster_2_1 = ak.firsts(sorted_clusters2_1)
    best_cluster_2_2 = ak.firsts(sorted_clusters2_2)
    best_gen_2_1 = two_gen_1_gen_matched[:, 0]
    best_gen_2_2 = two_gen_2_gen_matched[:, 0]

    best_cluster_two_matched = ak.Array([[x, y] for x, y in zip(best_cluster_2_1, best_cluster_2_2)])
    best_gen_two_matched = ak.Array([[x, y] for x, y in zip(best_gen_2_1, best_gen_2_2)])

    pair_cluster_masked_highest_pt = ak.concatenate([best_cluster_one_matched, best_cluster_two_matched], axis=0)
    pair_gen_masked_highest_pt = ak.concatenate([best_gen_one_matched, best_gen_two_matched], axis=0)

    return pair_cluster_masked_highest_pt, pair_gen_masked_highest_pt, clusters, genparts