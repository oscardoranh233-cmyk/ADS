"""Optional RDKit features for new SMILES; locked training CSV values are preserved."""
from __future__ import annotations
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors, MACCSkeys, rdFreeSASA, rdMolDescriptors

def calc_sasa_3d(mol: Chem.Mol, seed: int = 0) -> tuple[float, bool]:
    mol3d = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    ok = AllChem.EmbedMolecule(mol3d, params)
    if ok != 0:
        return float(rdMolDescriptors.CalcLabuteASA(mol)), False
    try:
        AllChem.UFFOptimizeMolecule(mol3d, maxIters=300)
    except Exception:
        pass
    radii = rdFreeSASA.classifyAtoms(mol3d)
    return float(rdFreeSASA.CalcSASA(mol3d, radii)), True

def calc_charges(mol: Chem.Mol) -> tuple[float, float]:
    charged = Chem.AddHs(mol)
    AllChem.ComputeGasteigerCharges(charged)
    values: list[float] = []
    for atom in charged.GetAtoms():
        if atom.HasProp("_GasteigerCharge"):
            value = float(atom.GetProp("_GasteigerCharge"))
            if np.isfinite(value):
                values.append(value)
    if not values:
        return 0.0, 0.0
    return float(max(values)), float(min(values))

def molecular_features(smiles: str) -> dict[str, float | int | str | bool]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    canonical = Chem.MolToSmiles(mol, canonical=True)
    max_charge, min_charge = calc_charges(mol)
    sasa_3d, embed_ok = calc_sasa_3d(mol)

    features: dict[str, float | int | str | bool] = {
        "canonical_smiles": canonical,
        "MolLogP": float(Crippen.MolLogP(mol)),
        "MolMR": float(Crippen.MolMR(mol)),
        "SASA_3D": float(sasa_3d),
        "rdkit3d_embed_ok": bool(embed_ok),
        "Max_Charge": max_charge,
        "Min_Charge": min_charge,
        "Num_HBD": int(rdMolDescriptors.CalcNumHBD(mol)),
        "Num_HBA": int(rdMolDescriptors.CalcNumHBA(mol)),
        "Rotatable_Bonds": int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        "TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
        "MolWt": float(Descriptors.MolWt(mol)),
        "ExactMolWt": float(Descriptors.ExactMolWt(mol)),
        "HeavyAtomCount": int(Descriptors.HeavyAtomCount(mol)),
        "RingCount": int(rdMolDescriptors.CalcNumRings(mol)),
        "AromaticRingCount": int(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "AliphaticRingCount": int(rdMolDescriptors.CalcNumAliphaticRings(mol)),
        "SaturatedRingCount": int(rdMolDescriptors.CalcNumSaturatedRings(mol)),
        "FractionCSP3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
        "NHOHCount": int(Descriptors.NHOHCount(mol)),
        "NOCount": int(Descriptors.NOCount(mol)),
        "NumHeteroatoms": int(Descriptors.NumHeteroatoms(mol)),
        "NumValenceElectrons": int(Descriptors.NumValenceElectrons(mol)),
        "BalabanJ": float(Descriptors.BalabanJ(mol)),
        "BertzCT": float(Descriptors.BertzCT(mol)),
        "Chi0": float(Descriptors.Chi0(mol)),
        "Chi1": float(Descriptors.Chi1(mol)),
        "Kappa1": float(Descriptors.Kappa1(mol)),
        "Kappa2": float(Descriptors.Kappa2(mol)),
        "Kappa3": float(Descriptors.Kappa3(mol)),
        "LabuteASA": float(rdMolDescriptors.CalcLabuteASA(mol)),
    }

    maccs = MACCSkeys.GenMACCSKeys(mol)
    for i in range(1, maccs.GetNumBits()):
        features[f"maccs_{i:03d}"] = int(maccs.GetBit(i))
    return features
