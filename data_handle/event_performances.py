#Loading the events for the new emulator to plot the performance
import numpy as np
import awkward as ak
import uproot
import math
import yaml
import cppyy
import os
import glob

def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    if iteration == total: 
        print()

with open('config_performances.yaml', "r") as afile:
    cfg_particles = yaml.safe_load(afile)["particles"]

def provide_events_performaces(n, particles, PU, thr=0.0):
    base_path = cfg_particles['base_path']
    name_tree = cfg_particles[PU][particles]["tree"]
    # filepath  = base_path + cfg_particles[PU][particles]["file"]
    # file_pattern = str(os.path.join(base_path, '*.root'))

    branches_gen = [
        'event', 'genpart_exeta', 'genpart_exphi', 'genpart_pt', 'genpart_gen', 'genpart_reachedEE'
    ]
    branches_cl_0p0113  = [
        'event', 'cl3d_p0113Tri_eta', 'cl3d_p0113Tri_phi', 'cl3d_p0113Tri_pt'
    ]
    branches_cl_0p016  = [
        'event', 'cl3d_p016Tri_eta', 'cl3d_p016Tri_phi', 'cl3d_p016Tri_pt'
    ]
    branches_cl_0p03  = [
        'event', 'cl3d_p03Tri_eta', 'cl3d_p03Tri_phi', 'cl3d_p03Tri_pt'
    ]
    branches_cl_0p045  = [
        'event', 'cl3d_p045Tri_eta', 'cl3d_p045Tri_phi', 'cl3d_p045Tri_pt'
    ]
    branches_cl_Ref  = [
        'event', 'cl3d_Ref_eta', 'cl3d_Ref_phi', 'cl3d_Ref_pt'
    ]

    all_data_gen = []
    all_data_cl_0p0113=[]
    all_data_cl_0p016=[]
    all_data_cl_0p03=[]
    all_data_cl_0p045=[]
    all_data_Ref=[]
    total_entries = 0
    file_list = sorted(glob.glob(os.path.join(base_path, '*.root')))

    printProgressBar(0, n, prefix='Reading '+str(n)+' events from ROOT file:', suffix='Complete', length=50)
    for filepath in file_list:
        if total_entries >= n:
            break
        try:
            tree = uproot.open(filepath)[name_tree]
            entries_to_read = min(n - total_entries, tree.num_entries)
            data_gen = tree.arrays(branches_gen, entry_stop=entries_to_read, library='ak')
            data_cl_0p0113  = tree.arrays(branches_cl_0p0113,  entry_stop=entries_to_read, library='ak')
            data_cl_0p016  = tree.arrays(branches_cl_0p016,  entry_stop=entries_to_read, library='ak')
            data_cl_0p03  = tree.arrays(branches_cl_0p03,  entry_stop=entries_to_read, library='ak')
            data_cl_0p045  = tree.arrays(branches_cl_0p045,  entry_stop=entries_to_read, library='ak')
            data_cl_Ref  = tree.arrays(branches_cl_Ref,entry_stop=entries_to_read, library='ak')
            all_data_gen.append(data_gen)
            all_data_cl_0p0113.append(data_cl_0p0113)
            all_data_cl_0p016.append(data_cl_0p016)
            all_data_cl_0p03.append(data_cl_0p03)
            all_data_cl_0p045.append(data_cl_0p045)
            all_data_Ref.append(data_cl_Ref)
            total_entries += entries_to_read
            printProgressBar(total_entries, n, prefix='Reading '+str(n)+' events from ROOT file:', suffix='Complete', length=50)
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
    

    data_gen = ak.concatenate(all_data_gen, axis=0)
    data_cl_0p0113 = ak.concatenate(all_data_cl_0p0113, axis=0)
    data_cl_0p016 = ak.concatenate(all_data_cl_0p016, axis=0)
    data_cl_0p03 = ak.concatenate(all_data_cl_0p03, axis=0)
    data_cl_0p045 = ak.concatenate(all_data_cl_0p045, axis=0)
    data_cl_Ref = ak.concatenate(all_data_Ref, axis=0)

    print("\n")
    print("len gen events", len(data_gen.genpart_pt))
    # print(ak.flatten(data_gen.genpart_pt,axis=-1))
    print("len gen events", len(ak.flatten(data_gen.genpart_pt,axis=-1)))

    # Mask: Only select particles with flag "gen-particle_gen" different from -1, only keep the gens that reached the EE and only keep the particles whose shower will stay inside the detector
    particle_mask = (data_gen.genpart_gen != -1) & (data_gen.genpart_reachedEE ==2) & (abs(data_gen.genpart_exeta)>1.6) & (abs(data_gen.genpart_exeta)<2.9)
    # particle_mask = (data_gen.genpart_gen != -1) & (data_gen.genpart_reachedEE ==2) 
    # particle_mask = (data_gen.genpart_gen != -1) 
    # Filter each field using the mask
    filtered_gen = ak.zip({
        'event': data_gen.event,  
        'genpart_exeta': data_gen.genpart_exeta[particle_mask],
        'genpart_exphi': data_gen.genpart_exphi[particle_mask],
        'genpart_pt': data_gen.genpart_pt[particle_mask],
        'genpart_gen': data_gen.genpart_gen[particle_mask],
        'genpart_reachedEE': data_gen.genpart_reachedEE[particle_mask]
    })

    reachedEE_ev_mask = ak.num(filtered_gen['genpart_exeta']) > 0
    filtered_gen = filtered_gen[reachedEE_ev_mask]
    data_cl_0p0113 = data_cl_0p0113[reachedEE_ev_mask]
    data_cl_0p016 = data_cl_0p016[reachedEE_ev_mask]
    data_cl_0p03 = data_cl_0p03[reachedEE_ev_mask]
    data_cl_0p045 = data_cl_0p045[reachedEE_ev_mask]
    data_cl_Ref = data_cl_Ref[reachedEE_ev_mask]

    if thr != 0:
        mask_0p0113= data_cl_0p0113.cl3d_p0113Tri_pt > thr
        data_cl_0p0113=ak.zip({
            'event': data_cl_0p0113.event, 
            'cl3d_p0113Tri_eta': data_cl_0p0113.cl3d_p0113Tri_eta[mask_0p0113], 
            'cl3d_p0113Tri_phi': data_cl_0p0113.cl3d_p0113Tri_phi[mask_0p0113], 
            'cl3d_p0113Tri_pt' : data_cl_0p0113.cl3d_p0113Tri_pt[mask_0p0113]
        })
        mask_0p016= data_cl_0p016.cl3d_p016Tri_pt > thr
        data_cl_0p016=ak.zip({
            'event': data_cl_0p016.event, 
            'cl3d_p016Tri_eta': data_cl_0p016.cl3d_p016Tri_eta[mask_0p016], 
            'cl3d_p016Tri_phi': data_cl_0p016.cl3d_p016Tri_phi[mask_0p016], 
            'cl3d_p016Tri_pt' : data_cl_0p016.cl3d_p016Tri_pt[mask_0p016]
        })
        mask_0p03= data_cl_0p03.cl3d_p03Tri_pt > thr
        data_cl_0p03=ak.zip({
            'event': data_cl_0p03.event, 
            'cl3d_p03Tri_eta': data_cl_0p03.cl3d_p03Tri_eta[mask_0p03], 
            'cl3d_p03Tri_phi': data_cl_0p03.cl3d_p03Tri_phi[mask_0p03], 
            'cl3d_p03Tri_pt' : data_cl_0p03.cl3d_p03Tri_pt[mask_0p03]
        })
        mask_0p045= data_cl_0p045.cl3d_p045Tri_pt > thr
        data_cl_0p045=ak.zip({
            'event': data_cl_0p045.event, 
            'cl3d_p045Tri_eta': data_cl_0p045.cl3d_p045Tri_eta[mask_0p045], 
            'cl3d_p045Tri_phi': data_cl_0p045.cl3d_p045Tri_phi[mask_0p045], 
            'cl3d_p045Tri_pt' : data_cl_0p045.cl3d_p045Tri_pt[mask_0p045]
        })
        mask_Ref= data_cl_Ref.cl3d_Ref_pt > thr
        data_cl_Ref=ak.zip({
            'event': data_cl_Ref.event, 
            'cl3d_Ref_eta': data_cl_Ref.cl3d_Ref_eta[mask_Ref], 
            'cl3d_Ref_phi': data_cl_Ref.cl3d_Ref_phi[mask_Ref], 
            'cl3d_Ref_pt' : data_cl_Ref.cl3d_Ref_pt[mask_Ref]
        })

    return filtered_gen, data_cl_0p0113, data_cl_0p016, data_cl_0p03, data_cl_0p045, data_cl_Ref


def apply_matching(events, att_eta, att_phi, gen, args, deltaR=0.1):
    #Masking events were no clusters are reconstructed
    mask_ev = ak.num(getattr(events, att_eta)) > 0
    filtered_events = events[mask_ev]
    filtered_gen = gen[mask_ev]
    # print(min(ak.flatten(filtered_gen.genpart_pt, axis=-1)))
    # print(filtered_gen.genpart_pt)

    if args.total_efficiency:
        print("Number of events with clusters", len(filtered_gen))

    if not args.total_efficiency and args.gen_pt_cut !=0:
        #Adding a cut on the gen particle pt 
        print("Applying gen pt cut")
        mask_pt_gen = filtered_gen.genpart_pt > args.gen_pt_cut
        filtered_pt_gen = filtered_gen[mask_pt_gen]
        mask_gen = ak.num(filtered_pt_gen.genpart_pt) > 0
        filtered_gen = filtered_pt_gen[mask_gen]
        filtered_events = filtered_events[mask_gen]
        # print(ak.type(filtered_gen))
        # print(ak.type(filtered_events))

    # Building structured arrays
    clusters = ak.zip({
        "eta": getattr(filtered_events, att_eta),
        "phi": getattr(filtered_events, att_phi),
        "pt": getattr(filtered_events, att_eta.replace("eta", "pt")),
    })
    genparts = ak.zip({
        "eta": getattr(filtered_gen, 'genpart_exeta'),
        "phi": getattr(filtered_gen, 'genpart_exphi'),
        "pt": getattr(filtered_gen, 'genpart_pt'),
        "gen_flag": getattr(filtered_gen,'genpart_gen'),
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
        # print(min(ak.flatten(filtered_pt_cluster.pt, axis=-1)))
        genparts = genparts[mask_empty_events]
        # print(ak.type(clusters))
        # print(ak.type(genparts))
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
        # raise ValueError("2 gen particles matched to the same cluster")
    if ak.any(ak.num(pair_gen_masked, axis=2) == 2):
        print("2 gen particles matched to the same cluster")
        # raise ValueError("2 gen particles matched to the same cluster")

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
    flatten_pair_cluster_masked= ak.flatten(pair_cluster_masked, axis = -1)
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

    # cluster_matched = ak.zip({
    #     "eta": flatten_pair_cluster_masked.eta ,
    #     "phi": flatten_pair_cluster_masked.phi,
    #     "pt": flatten_pair_cluster_masked.pt,
    #     "gen_flag": flatten_pair_gen_masked.gen_flag,
    # })

    # print("pt type:", flatten_pair_cluster_masked.pt.type)
    # print("delta_r type:", (delta_r[mask][mask_empty][~empty_mask]).type)
    # delta_r_clean = ak.values_astype(delta_r[mask][mask_empty][~empty_mask], "float64")
    # print("delta_r_clean type:", delta_r_clean.type)
    # print("delta_r_clean:", delta_r_clean)
    # print("cluster_matched type:", flatten_pair_cluster_masked.pt)
    # delta_r_clean = ak.values_astype(
    # ak.flatten(delta_r[mask][mask_empty][~empty_mask], axis=2),
    # "float64"
    # )
    # print("delta_r_clean type:", delta_r_clean.type)

    gen_matched = ak.zip({
        "eta": flatten_pair_gen_masked.eta ,
        "phi": flatten_pair_gen_masked.phi,
        "pt": flatten_pair_gen_masked.pt,
        "gen_flag": flatten_pair_gen_masked.gen_flag,
    })

    # If there is only one gen particle matched

    one_gen_matched_mask = nb_matched_gen_particles == 1
    one_cluster_matched = cluster_matched[one_gen_matched_mask]
    one_gen_matched = gen_matched[one_gen_matched_mask]
    sorted_one_cluster_matched = one_cluster_matched[ak.argsort(one_cluster_matched.pt, ascending=False)]
    best_cluster_one_matched = ak.firsts(sorted_one_cluster_matched)
    best_gen_one_matched = one_gen_matched[:, 0]

    best_cluster_one_matched = ak.zip({
        'eta': ak.unflatten(best_cluster_one_matched.eta,1),
        'phi': ak.unflatten(best_cluster_one_matched.phi,1),
        'pt': ak.unflatten(best_cluster_one_matched.pt,1),
        'gen_flag': ak.unflatten(best_cluster_one_matched.gen_flag,1),
        'delta_r': ak.unflatten(best_cluster_one_matched.delta_r,1),
    })
    # best_cluster_one_matched = ak.zip({
    #     'eta': ak.unflatten(best_cluster_one_matched.eta,1),
    #     'phi': ak.unflatten(best_cluster_one_matched.phi,1),
    #     'pt': ak.unflatten(best_cluster_one_matched.pt,1),
    #     'gen_flag': ak.unflatten(best_cluster_one_matched.gen_flag,1),
    # })

    best_gen_one_matched = ak.zip({
        'eta': ak.unflatten(best_gen_one_matched.eta, 1),
        'phi': ak.unflatten(best_gen_one_matched.phi, 1),
        'pt': ak.unflatten(best_gen_one_matched.pt, 1),
        'gen_flag': ak.unflatten(best_gen_one_matched.gen_flag, 1),
    })

    # If there are two gen particles matched

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

    ##Use for debugging
    # for i in range(len(max_idx1)):
    #     if ak.any(max_idx1[i]!=0):
    #         print(max_idx1[i])
    #         print(i)
    # print(result1.pt[89])
    # print(result2.pt[89])

    # print("deltar", delta_r)
    # print(pair_cluster_masked_highest_pt.delta_r)

    return pair_cluster_masked_highest_pt, pair_gen_masked_highest_pt, clusters, genparts


