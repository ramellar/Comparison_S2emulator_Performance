#Definition of the functions used for event selection in the tau case
import numpy as np
import awkward as ak
import uproot
from tqdm import tqdm
import subprocess
import os




#Function for computing the \DeltaR between two particles
def deltaR(eta1, phi1, eta2, phi2):
    dphi = (phi1 - phi2 + np.pi) % (2*np.pi) - np.pi #I add the last member to avoid to cut right events
    deta = eta1 - eta2
    return np.sqrt(deta**2 + dphi**2)




#Function that returns the final version with status 2 for each tau 
def find_tau_status2(idx, iev, id, status, daughters):
    if abs(id[iev][idx]) == 15 and status[iev][idx] == 2:
        return idx
    for d in daughters[iev][idx]:
        if abs(id[iev][d]) == 15:
            res = find_tau_status2(d, iev, id, status, daughters)
            if res is not None:
                return res
    return None




#Function that finds all the final particles comming from tau decays
def find_status1_daughters(idx, iev, status, daughters):
    # if it's already final, it keeps it
    if status[iev][idx] == 1:
        return [idx]
    final_idxs = []
    for d in daughters[iev][idx]:
        final_idxs.extend(find_status1_daughters(d, iev, status, daughters))
    return final_idxs




#Function for tau reconstruction starting from gen collection
def taureco_function(pt, eta, phi, energy, id, status, daughters, gpt, geta, gphi, genergy, gmass):
    
    #Mask to isolate all the final Higgs bosons in each event
    mask_higgs = ((id == 25) & (status == 62))
    pt = pt[mask_higgs]
    eta = eta[mask_higgs]
    phi = phi[mask_higgs]
    energy = energy[mask_higgs]
    id = id[mask_higgs]
    status = status[mask_higgs]
    daughters = daughters[mask_higgs]
    
    
    #Look for all the tau with status = 2 that come from Higgs decay
    for iev in range(len(daughters)):
        event_tau2 = []
        for group in daughters[iev]:
            for idx in group:
                if abs(id[iev][idx]) == 15:
                    res = find_tau_status2(idx, iev, id, status, daughters)
                    if res is not None:
                        event_tau2.append(res)
        tau2_idx.append(event_tau2)
    tau2_idx = ak.Array(tau2_idx)
    
    pt = pt[tau2_idx]
    eta = eta[tau2_idx]
    phi = phi[tau2_idx]
    id = id[tau2_idx]
    energy = energy[tau2_idx]
    status = status[tau2_idx]
    daughters = daughters[tau2_idx]
    
     
    #Match the reco_taus from gen_* collection with the ones from gentau_* collection
    dr_cut = 0.05
    
    gen_eta_b, reco_eta_b = ak.broadcast_arrays(geta[:, :, None], eta[:, None, :])
    gen_phi_b, reco_phi_b = ak.broadcast_arrays(gphi[:, :, None], phi[:, None, :])
    
    dR = deltaR(gen_eta_b, gen_phi_b, reco_eta_b, reco_phi_b)
    
    best_match_idx = ak.argmin(dR, axis=2) 
    
    # min dR between every gentau and tau_fh
    min_dR = ak.min(dR, axis=2) 
    
    mask_clean = min_dR < dr_cut
    
    gpt = gpt[mask_clean]
    geta = geta[mask_clean]
    gphi = gphi[mask_clean]
    genergy = genergy[mask_clean]
    gid = gid[mask_clean]
    gstatus = gstatus[mask_clean]
    gdaughters = gdaughters[mask_clean]
    
    pt = pt[best_match_idx][mask_clean]
    eta = eta[best_match_idx][mask_clean]
    phi = phi[best_match_idx][mask_clean]
    energy = energy[best_match_idx][mask_clean]
    status = status[best_match_idx][mask_clean]
    id = id[best_match_idx][mask_clean]
    daughters = daughters[best_match_idx][mask_clean]   
    
    
    #Mask to isolate all the Higgs bosons for every event
    tau_final_products_idx = []
    for iev in range(len(daughters)):
        event_products = []
        for tau_daus in daughters[iev]:
            tau_products = []
            for idx in tau_daus:
                tau_products.extend(find_status1_daughters(idx, iev, status, daughters))
            tau_products = sorted(set(tau_products))   # opzionale, per evitare duplicati
            event_products.append(tau_products)
        tau_final_products_idx.append(event_products)
    tau_final_products_idx = ak.Array(tau_final_products_idx)
    
    
    #Selection of final particles resulting from tau decays from gen_* collection
    tau_final_products_id = ak.Array([[id[iev][idxs] for idxs in tau_final_products_idx[iev]] for iev in range(len(tau_final_products_idx))])
    tau_final_products_pt = ak.Array([[pt[iev][idxs] for idxs in tau_final_products_idx[iev]]for iev in range(len(tau_final_products_idx))])
    tau_final_products_eta = ak.Array([[eta[iev][idxs] for idxs in tau_final_products_idx[iev]] for iev in range(len(tau_final_products_idx))])
    tau_final_products_phi = ak.Array([[phi[iev][idxs] for idxs in tau_final_products_idx[iev]] for iev in range(len(tau_final_products_idx))])
    tau_final_products_energy = ak.Array([[energy[iev][idxs] for idxs in tau_final_products_idx[iev]] for iev in range(len(tau_final_products_idx))])
    #tau_final_products_status = ak.Array([[status[iev][idxs] for idxs in tau_final_products_idx[iev]] for iev in range(len(tau_final_products_idx))])
    
    #Selection of the right final particles (no neutrinos)
    mask_vis = ((abs(tau_final_products_id) != 12) & (abs(tau_final_products_id) != 14) & (abs(tau_final_products_id) != 16))
    
    tau_fp_vis_pt = tau_final_products_pt[mask_vis]
    tau_fp_vis_eta = tau_final_products_eta[mask_vis]
    tau_fp_vis_phi = tau_final_products_phi[mask_vis]
    tau_fp_vis_energy = tau_final_products_energy[mask_vis]
    #tau_fp_vis_id = tau_final_products_id[mask_vis]
    #tau_fp_vis_status = tau_final_products_status[mask_vis]
    #tau_fp_vis_idx = tau_final_products_idx[mask_vis]
    
    
    #Calculate the Higgs invariant mass based on the final particles reaching the detector (using gen_*)
    px_reco = tau_fp_vis_pt * np.cos(tau_fp_vis_phi)
    py_reco = tau_fp_vis_pt * np.sin(tau_fp_vis_phi)
    pz_reco = tau_fp_vis_pt * np.sinh(tau_fp_vis_eta)
    E_reco  = tau_fp_vis_energy 
    
    px_reco_tot = ak.sum(px_reco, axis=-1)
    py_reco_tot = ak.sum(py_reco, axis=-1)
    pz_reco_tot = ak.sum(pz_reco, axis=-1)
    E_reco_tot  = ak.sum(E_reco, axis=-1)
    
    px_reco_H = ak.sum(px_reco_tot, axis=-1)
    py_reco_H = ak.sum(py_reco_tot, axis=-1)
    pz_reco_H = ak.sum(pz_reco_tot, axis=-1)
    E_reco_H  = ak.sum(E_reco_tot, axis=-1)
    
    m2_reco_H = E_reco_H**2 - px_reco_H**2 - py_reco_H**2 - pz_reco_H**2
    m_reco = np.sqrt(ak.where(m2_reco_H > 0, m2_reco_H, 0))
    
    
    return m_reco, best_match_idx, mask_clean #, pt, eta, phi, energy, status, id, daughters, gpt, geta, gphi, gid, genergy, gstatus, gdaughters


def vis_filter_function(filter0, filter1, m_reco, best_match_idx, mask_clean, gvis_pt, gvis_eta, gvis_phi, gvis_energy, gvis_mass, gprod_pt, gprod_eta, gprod_phi, gprod_energy, gprod_mass, gprod_id):
    
    #Applying the filter to gentau_vis_* and gentau_products_* to correctly sort the particles
    gvis_pt = gvis_pt[best_match_idx][mask_clean]
    gvis_eta = gvis_eta[best_match_idx][mask_clean]
    gvis_phi = gvis_phi[best_match_idx][mask_clean]
    gvis_energy = gvis_energy[best_match_idx][mask_clean]
    gvis_mass = gvis_mass[best_match_idx][mask_clean]
    
    gprod_pt =  gprod_pt[best_match_idx][mask_clean]
    gprod_eta =  gprod_eta[best_match_idx][mask_clean]
    gprod_phi = gprod_phi[best_match_idx][mask_clean]
    gprod_energy = gprod_energy[best_match_idx][mask_clean]
    gprod_mass = gprod_mass[best_match_idx][mask_clean]
    gprod_id = gprod_id[best_match_idx][mask_clean]


    #Calculate the Higgs invariant mass based on the final particles reaching the detector (using gen_prod_*)
    px_gentau = gprod_pt * np.cos(gprod_phi)
    py_gentau = gprod_pt * np.sin(gprod_phi)
    pz_gentau = gprod_pt * np.sinh(gprod_eta)
    E_gentau  = gprod_energy
    
    px_gentau_tot = ak.sum(px_gentau, axis=-1)
    py_gentau_tot = ak.sum(py_gentau, axis=-1)
    pz_gentau_tot = ak.sum(pz_gentau, axis=-1)
    E_gentau_tot  = ak.sum(E_gentau, axis=-1)
    
    px_gentau_H = ak.sum(px_gentau_tot, axis=-1)
    py_gentau_H = ak.sum(py_gentau_tot, axis=-1)
    pz_gentau_H = ak.sum(pz_gentau_tot, axis=-1)
    E_gentau_H  = ak.sum(E_gentau_tot, axis=-1)
    
    m2_gentau_H = E_gentau_H**2 - px_gentau_H**2 - py_gentau_H**2 - pz_gentau_H**2
    m_gentau_vis = np.sqrt(ak.where(m2_gentau_H > 0, m2_gentau_H, 0))
    
    
    #Calculate the Higgs invariant mass based on the final particles reaching the detector (using gen_prod_*)
    px_gentau_totv = gvis_pt * np.cos(gvis_phi)
    py_gentau_totv = gvis_pt * np.sin(gvis_phi)
    pz_gentau_totv = gvis_pt * np.sinh(gvis_eta)
    E_gentau_totv  = gvis_energy
    
    px_gentau_Hv = ak.sum(px_gentau_totv, axis=-1)
    py_gentau_Hv = ak.sum(py_gentau_totv, axis=-1)
    pz_gentau_Hv = ak.sum(pz_gentau_totv, axis=-1)
    E_gentau_Hv  = ak.sum(E_gentau_totv, axis=-1)
    
    m2_gentau_Hv = E_gentau_Hv**2 - px_gentau_Hv**2 - py_gentau_Hv**2 - pz_gentau_Hv**2
    m_gentau_visv = np.sqrt(ak.where(m2_gentau_Hv > 0, m2_gentau_Hv, 0))
    
    
    #Flag all the events where m_vis =! m_prod
    filter_value_vp = filter0
    flag_vis_prod = ak.values_astype(abs(m_gentau_visv - m_gentau_vis) > filter_value_vp, np.int32)


    #Flag all the events where m_vis =! m_reco
    filter_value = filter1
    flag_pp = ak.values_astype(abs(m_reco - m_gentau_visv) > filter_value, np.int32)
    
    return flag_vis_prod, flag_pp, gvis_pt, gvis_eta, gvis_phi, gvis_energy, gvis_mass #, gprod_pt, gprod_eta, gprod_phi, gprod_energy, gprod_mass, gprod_id



# I removed some outputs from both the functions cause maybe they are useless -> If I'll need them it's easy to add them back.