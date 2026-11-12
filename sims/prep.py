"""
Build an equilibrated ligand-in-water system with a3fe's protocol, then write one
SOMD run folder per (stage, λ, replicate), all on the dense λ grid.

    python sims/prep.py configs/toluene.yaml

Resumable: every step is skipped if its output already exists. Mirrors a3fe's
Leg.setup -> Leg.setup_stages -> LambdaWindow/Simulation setup (a3fe/run/leg.py,
lambda_window.py, simulation.py), minus the adaptive machinery.
"""
import os, sys, glob, shutil, json, subprocess, yaml
import a3fe as a3
from a3fe.run import system_prep as sp
from a3fe.configuration import PreparationStage
import BioSimSpace.Sandpit.Exscientia as BSS
from rdkit import Chem
from rdkit.Chem import AllChem

LEG, ENGINE = a3.LegType.FREE, a3.EngineType.SOMD
STAGES = {"discharge": a3.StageType.DISCHARGE, "vanish": a3.StageType.VANISH}

cfg = yaml.safe_load(open(sys.argv[1]))
base = f"{cfg['workdir']}/{cfg['name']}"
inp = f"{base}/input"
os.makedirs(inp, exist_ok=True)


def have(stage):
    """True if a3fe's output files for this preparation stage exist in inp."""
    return all(os.path.exists(f"{inp}/{f}")
               for f in stage.get_simulation_input_files(LEG))


# ---- Job 1a: molecule. SMILES -> 3D structure file a3fe can read -------------
sdf = f"{inp}/ligand.sdf"
if not os.path.exists(sdf):
    mol = Chem.AddHs(Chem.MolFromSmiles(cfg["smiles"]))
    if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:   # guess 3D coordinates
        raise RuntimeError("RDKit embedding failed")
    AllChem.MMFFOptimizeMolecule(mol)                    # clean up the geometry
    w = Chem.SDWriter(sdf); w.write(mol); w.close()

prep_cfg = a3.SomdSystemPreparationConfig(slurm=False,
                                          ensemble_equilibration_time=cfg["equil_ps"])
prep_cfg.dump(inp, LEG)        # a3fe's prep functions read their settings from here

# ---- Job 1b–e: parameterize -> solvate -> minimize -> heat/pre-equilibrate ----
# Skipping finished steps matters: re-solvating would change the atom count and
# silently invalidate equilibrated coordinates that already exist.
for stage, fn in [(PreparationStage.PARAMETERISED, sp.parameterise_input),
                  (PreparationStage.SOLVATED, sp.solvate_input),
                  (PreparationStage.MINIMISED, sp.minimise_input),
                  (PreparationStage.PREEQUILIBRATED, sp.heat_and_preequil_input)]:
    if have(stage):
        print("skip", fn.__name__, "(done)", flush=True)
        continue
    print("running", fn.__name__, flush=True)
    fn(LEG, ENGINE, inp, inp)

# ---- Job 1f: one independent equilibration per replicate ---------------------
# Free leg + SOMD: GROMACS runs in a temp dir, final coords saved as {d}/somd.rst7.
eq_dirs = []
for r in range(1, cfg["replicates"] + 1):
    d = f"{base}/equil_{r}"
    if not os.path.exists(f"{d}/somd.rst7"):
        os.makedirs(d, exist_ok=True)
        print(f"running ensemble equilibration {r} ({cfg['equil_ps']} ps)", flush=True)
        sp.run_ensemble_equilibration(LEG, ENGINE, inp, d)
    eq_dirs.append(d)

# ---- Job 2: SOMD run folders, one per (stage, λ, replicate) -------------------
# As in Leg.run_ensemble_equilibration / Leg.setup_stages: load the pre-equilibrated
# system and mark the ligand (first molecule, named LIG) for decoupling.
system = BSS.IO.readMolecules([f"{inp}/{f}" for f in
                               PreparationStage.PREEQUILIBRATED.get_simulation_input_files(LEG)])
lig = BSS.Align.decouple(system[0], intramol=True)
system.updateMolecule(0, lig)
lig = system[0]
assert lig._sire_object.name().value() == "LIG", "first molecule is not the ligand"
lig_charge = round(lig.charge().value())
if lig_charge != 0:
    raise NotImplementedError("charged ligand: a3fe would need co-alchemical ions + PME")
perturbed_resnum = system.getIndex(lig) + 1

ns = float(cfg["ns_per_tape"])
manifest = {"name": cfg["name"], "smiles": cfg["smiles"], "leg": "free",
            "replicates": cfg["replicates"], "ns_per_tape": ns,
            "equil_ps": cfg["equil_ps"], "a3fe_version": a3.__version__,
            "a3fe_lambda": {}, "lambda": {}, "runs": []}

for name, stype in STAGES.items():
    grid = [float(x) for x in cfg["lambda"][name]]
    a3fe_grid = prep_cfg.lambda_values[LEG][stype]
    assert set(a3fe_grid) <= set(grid), f"{name}: dense grid must contain a3fe's values"
    manifest["lambda"][name] = grid
    manifest["a3fe_lambda"][name] = a3fe_grid

    # Per-stage SOMD topology + perturbation file, written by BSS exactly as a3fe
    # does (setup_only, dummy λ/runtime, velocities dropped), then hoisted out of
    # the lambda_* dir BSS creates.
    sin = f"{base}/{name}/input"
    if not all(os.path.exists(f"{sin}/{f}") for f in ("somd.prm7", "somd.pert")):
        os.makedirs(sin, exist_ok=True)
        print(f"writing SOMD inputs for {name} ({stype.bss_perturbation_type})", flush=True)
        protocol = BSS.Protocol.FreeEnergy(runtime=0.001 * BSS.Units.Time.nanosecond,
                                           lam_vals=[0.0],
                                           perturbation_type=stype.bss_perturbation_type)
        BSS.FreeEnergy.AlchemicalFreeEnergy(system, protocol, engine="somd",
                                            work_dir=sin, setup_only=True,
                                            property_map={"velocity": "foo"})
        for f in glob.glob(f"{sin}/lambda_*/*"):
            if not f.endswith(".cfg"):
                shutil.copy(f, sin)
        for dd in glob.glob(f"{sin}/lambda_*"):
            subprocess.run(["rm", "-rf", dd], check=True)

    # Stage config = a3fe's SomdConfig defaults, with the FULL dense grid as the
    # λ array so SOMD records u_kl at every window.
    scfg = a3.SomdConfig()
    scfg.set_ligand_charge(lig_charge)
    scfg.perturbed_residue_number = perturbed_resnum
    scfg.use_boresch_restraints = False
    scfg.turn_on_receptor_ligand_restraints = False
    scfg.lambda_values = grid
    scfg.setup_lambda_arrays(stype)
    manifest["somd"] = {"timestep_fs": scfg.timestep,
                        "energy_frequency": scfg.energy_frequency,
                        "sample_interval_ps": scfg.gradient_output_interval * 1000,
                        "temperature_K": scfg.analysis_temperature,
                        "cutoff_type": scfg.cutoff_type}

    for li, lam in enumerate(grid):
        for r in range(1, cfg["replicates"] + 1):
            rd = f"{base}/{name}/lambda_{lam:.4f}/rep_{r}"
            manifest["runs"].append({"stage": name, "lam": lam, "lam_index": li,
                                     "a3fe_window": lam in a3fe_grid, "rep": r,
                                     "dir": os.path.relpath(rd, base), "ns": ns})
            if os.path.exists(f"{rd}/simfile.dat"):
                continue                      # already has output: don't touch it
            os.makedirs(rd, exist_ok=True)
            # Like LambdaWindow: link topology/coords, copy everything else.
            for src, dst in [(f"{sin}/somd.prm7", "somd.prm7"),
                             (f"{eq_dirs[r-1]}/somd.rst7", "somd.rst7")]:
                if os.path.lexists(f"{rd}/{dst}"):
                    os.remove(f"{rd}/{dst}")
                os.symlink(os.path.relpath(src, rd), f"{rd}/{dst}")
            shutil.copy(f"{sin}/somd.pert", f"{rd}/somd.pert")
            scfg.write_config(run_dir=rd, lambda_val=lam, runtime=ns,
                              top_file="somd.prm7", coord_file="somd.rst7",
                              morph_file="somd.pert")

with open(f"{base}/manifest.json", "w") as f:
    json.dump(manifest, f, indent=1)
print(f"wrote {len(manifest['runs'])} run folders + {base}/manifest.json "
      f"({len(manifest['runs']) * ns:g} ns total)")
