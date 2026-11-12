"""
Build an equilibrated ligand-in-water system with a3fe's protocol, then write one
SOMD run folder per (stage, λ, replicate), all on the dense λ grid.

    python sims/prep.py configs/toluene.yaml
"""
import os, sys, shutil, json, yaml
import a3fe as a3
from a3fe.run import system_prep as sp
import BioSimSpace.Sandpit.Exscientia as BSS
from rdkit import Chem
from rdkit.Chem import AllChem

LEG, ENGINE = a3.LegType.FREE, a3.EngineType.SOMD
STAGES = {"discharge": a3.StageType.DISCHARGE, "vanish": a3.StageType.VANISH}

cfg = yaml.safe_load(open(sys.argv[1]))
base = f"{cfg['workdir']}/{cfg['name']}"
inp = f"{base}/input"
os.makedirs(inp, exist_ok=True)

# ---- Job 1a: molecule. SMILES -> 3D structure file a3fe can read -------------
sdf = f"{inp}/ligand.sdf"
if not os.path.exists(sdf):
    mol = Chem.AddHs(Chem.MolFromSmiles(cfg["smiles"]))
    AllChem.EmbedMolecule(mol, randomSeed=42)     # guess 3D coordinates
    AllChem.MMFFOptimizeMolecule(mol)             # clean up the geometry
    Chem.MolToMolFile(mol, sdf)

prep_cfg = a3.SomdSystemPreparationConfig(slurm=False,
                                          ensemble_equilibration_time=cfg["equil_ps"])
prep_cfg.dump(inp, LEG)        # a3fe's prep functions read their settings from here

# ---- Job 1b–e: parameterize -> solvate -> minimize -> heat/pre-equilibrate ----
for fn in [sp.parameterise_input, sp.solvate_input,
           sp.minimise_input, sp.heat_and_preequil_input]:
    print("running", fn.__name__, flush=True)
    fn(LEG, ENGINE, inp, inp)

# ---- Job 1f: one independent equilibration per replicate ---------------------
eq_dirs = []
for r in range(cfg["replicates"]):
    d = f"{base}/equil_{r+1}"
    if not os.path.exists(f"{d}/somd.rst7"):
        os.makedirs(d, exist_ok=True)
        sp.run_ensemble_equilibration(LEG, ENGINE, inp, d)
    eq_dirs.append(d)

# ---- Job 2: SOMD run folders, one per (stage, λ, replicate) -------------------
# TODO (needs the env): mark toluene as the alchemical molecule, write SOMD's
# topology + perturbation files per stage, then for every λ in cfg["lambda"][stage]
# and every replicate: copy those files + that replicate's equilibrated coordinates
# into  {base}/{stage}/lambda_{λ:.4f}/rep_{r}/  and write a SOMD config whose
# λ list is the FULL dense grid (so u_kl is recorded at every window).
