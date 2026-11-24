"""
Build equilibrated systems with a3fe's protocol, then write one SOMD run folder per
(leg, stage, λ, replicate), each stage on its own dense λ grid.

    python sims/prep.py configs/toluene.yaml     # free leg only, from SMILES
    python sims/prep.py configs/mdm2.yaml        # bound + free, from a3fe's example inputs

Two config shapes:
  - legacy (toluene): top-level `lambda: {stage: grid}` -> one free leg, folders at
    {base}/{stage}/lambda_X/rep_R (what notebooks/replay.ipynb reads).
  - `legs: {bound: {stage: grid}, free: {...}}` -> folders at
    {base}/{leg}/{stage}/lambda_X/rep_R.

Resumable: every step is skipped if its output already exists. Mirrors a3fe's
Leg.setup -> Leg.run_ensemble_equilibration -> Leg.setup_stages -> LambdaWindow /
Simulation setup (a3fe/run/leg.py, lambda_window.py, simulation.py), minus the
adaptive machinery.
"""
import os, sys, glob, shutil, json, subprocess, yaml
# GROMACS (run by BSS as plain `gmx mdrun`, no -nt) starts one thread-MPI rank per core.
# On many-core hosts that gives rank counts like 62 (= 2·31), which mdrun rejects
# ("large prime factor"). Capping at ~16 ranks with several OpenMP threads each keeps
# the domain decomposition sane on any core count; mdrun reads -ntomp from here.
_ncpu = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
os.environ.setdefault("OMP_NUM_THREADS", str(max(1, _ncpu // 16)))
import a3fe as a3
from a3fe.run import system_prep as sp
from a3fe.configuration import PreparationStage
import BioSimSpace.Sandpit.Exscientia as BSS

ENGINE = a3.EngineType.SOMD
STAGE_TYPES = {"restrain": a3.StageType.RESTRAIN,
               "discharge": a3.StageType.DISCHARGE,
               "vanish": a3.StageType.VANISH}
PREP_ORDER = [PreparationStage.PARAMETERISED, PreparationStage.SOLVATED,
              PreparationStage.MINIMISED, PreparationStage.PREEQUILIBRATED]
PREP_FNS = {PreparationStage.PARAMETERISED: sp.parameterise_input,
            PreparationStage.SOLVATED: sp.solvate_input,
            PreparationStage.MINIMISED: sp.minimise_input,
            PreparationStage.PREEQUILIBRATED: sp.heat_and_preequil_input}

cfg = yaml.safe_load(open(sys.argv[1]))
base = f"{cfg['workdir']}/{cfg['name']}"
legacy = "legs" not in cfg
legs = {"free": cfg["lambda"]} if legacy else cfg["legs"]
reps = cfg["replicates"]
ns = float(cfg["ns_per_tape"])


def leg_dir(leg_name):
    return base if legacy else f"{base}/{leg_name}"


def have(inp, stage, LEG):
    """True if a3fe's output files for this preparation stage exist in inp."""
    return all(os.path.exists(f"{inp}/{f}") for f in stage.get_simulation_input_files(LEG))


def stage_inputs(inp, LEG):
    """Bring the leg's starting structures into inp: from SMILES or a3fe's example data."""
    os.makedirs(inp, exist_ok=True)
    if "a3fe_example" in cfg:
        src = os.path.join(os.path.dirname(a3.__file__), "data", "example_calc_set",
                           cfg["a3fe_example"], "input")
        for f in PreparationStage.PREEQUILIBRATED.get_simulation_input_files(LEG):
            if not os.path.exists(f"{inp}/{f}"):
                shutil.copy(f"{src}/{f}", f"{inp}/{f}")
    elif "smiles" in cfg:
        if LEG != a3.LegType.FREE:
            raise ValueError("a SMILES-only config can only build the free leg")
        sdf = f"{inp}/ligand.sdf"
        if not os.path.exists(sdf):
            from rdkit import Chem
            from rdkit.Chem import AllChem
            mol = Chem.AddHs(Chem.MolFromSmiles(cfg["smiles"]))
            if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:   # guess 3D coordinates
                raise RuntimeError("RDKit embedding failed")
            AllChem.MMFFOptimizeMolecule(mol)                    # clean up the geometry
            w = Chem.SDWriter(sdf); w.write(mol); w.close()
    else:
        raise ValueError("config needs `smiles` or `a3fe_example`")


def search_restraint(inp, eq_dir, system, prep_cfg, LEG):
    """Boresch restraint from one equilibration trajectory, as in
    Leg.run_ensemble_equilibration (a3fe then uses run 1's restraint for every run)."""
    top = f"{inp}/{PreparationStage.PREEQUILIBRATED.get_simulation_input_files(LEG)[0]}"
    traj = BSS.Trajectory.Trajectory(topology=top, trajectory=f"{eq_dir}/gromacs.xtc",
                                     system=system)
    restraint = BSS.FreeEnergy.RestraintSearch.analyse(
        method="BSS", system=system, traj=traj, work_dir=eq_dir,
        temperature=298.15 * BSS.Units.Temperature.kelvin,
        append_to_ligand_selection=prep_cfg.append_to_ligand_selection)
    if restraint is None:
        raise ValueError(f"no Boresch restraint found from {eq_dir}")
    return restraint


manifest = {"name": cfg["name"], "replicates": reps, "ns_per_tape": ns,
            "equil_ps": cfg["equil_ps"], "a3fe_version": a3.__version__,
            "legs": {}, "runs": []}
for k in ("smiles", "a3fe_example"):
    if k in cfg:
        manifest[k] = cfg[k]

for leg_name, grids in legs.items():
    LEG = a3.LegType[leg_name.upper()]
    ldir = leg_dir(leg_name)
    inp = f"{ldir}/input"
    print(f"===== {leg_name} leg -> {ldir}", flush=True)
    stage_inputs(inp, LEG)

    prep_cfg = a3.SomdSystemPreparationConfig(slurm=False,
                                              ensemble_equilibration_time=cfg["equil_ps"])
    prep_cfg.dump(inp, LEG)    # a3fe's prep functions read their settings from here
    assert set(grids) == {s.name.lower() for s in prep_cfg.required_stages[LEG]}, \
        f"{leg_name}: config stages {sorted(grids)} != a3fe's required stages"

    # ---- Job 1: parameterize -> solvate -> minimize -> heat/pre-equilibrate -------
    # Start after the most advanced stage already present. Re-running an earlier step
    # (e.g. re-solvating) would change the atom count and silently invalidate
    # equilibrated coordinates that already exist.
    done = [i for i, st in enumerate(PREP_ORDER) if have(inp, st, LEG)]
    for st in PREP_ORDER[(max(done) + 1) if done else 0:]:
        print("running", PREP_FNS[st].__name__, flush=True)
        PREP_FNS[st](LEG, ENGINE, inp, inp)

    # ---- Job 1f: one independent equilibration per replicate ---------------------
    # GROMACS, 2 fs. Coordinates land in {d}/somd.rst7; for the bound leg a3fe also
    # keeps the trajectory ({d}/gromacs.xtc) in d, for the restraint search.
    # The replicates' equilibrations are independent, and one mdrun scales poorly past
    # ~16 cores, so on a many-core host they run concurrently, each in its own python
    # process pinned (taskset) to a disjoint slice of cores; mdrun sizes itself to its
    # affinity mask. With few cores this degenerates to one at a time.
    eq_dirs = [f"{ldir}/equil_{r}" for r in range(1, reps + 1)]
    todo = [(r, d) for r, d in enumerate(eq_dirs, 1) if not os.path.exists(f"{d}/somd.rst7")]
    cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    per = len(cpus) // max(1, len(todo)) if todo else 0
    if todo and per >= 8 and len(todo) > 1:
        procs = []
        for i, (r, d) in enumerate(todo):
            os.makedirs(d, exist_ok=True)
            mine = cpus[i * per:(i + 1) * per]
            env = dict(os.environ, OMP_NUM_THREADS=str(max(1, per // 4)))
            code = ("import a3fe as a3; from a3fe.run import system_prep as sp; "
                    f"sp.run_ensemble_equilibration(a3.LegType.{LEG.name}, "
                    f"a3.EngineType.{ENGINE.name}, {inp!r}, {d!r})")
            print(f"running {leg_name} ensemble equilibration {r} ({cfg['equil_ps']} ps) "
                  f"on {len(mine)} cores, concurrently", flush=True)
            log = open(f"{d}/equil_driver.log", "w")
            procs.append((r, subprocess.Popen(
                ["taskset", "-c", ",".join(map(str, mine)), sys.executable, "-c", code],
                env=env, stdout=log, stderr=subprocess.STDOUT)))
        bad = [r for r, p in procs if p.wait() != 0]
        if bad:
            raise RuntimeError(f"{leg_name} equilibration(s) {bad} failed; "
                               f"see {ldir}/equil_*/equil_driver.log")
    else:
        for r, d in todo:
            os.makedirs(d, exist_ok=True)
            print(f"running {leg_name} ensemble equilibration {r} ({cfg['equil_ps']} ps)",
                  flush=True)
            sp.run_ensemble_equilibration(LEG, ENGINE, inp, d)

    # Load the pre-equilibrated system and mark the ligand (first molecule, LIG)
    # for decoupling, as in Leg.run_ensemble_equilibration / Leg.setup_stages.
    system = BSS.IO.readMolecules([f"{inp}/{f}" for f in
                                   PreparationStage.PREEQUILIBRATED.get_simulation_input_files(LEG)])
    lig = BSS.Align.decouple(system[0], intramol=True)
    system.updateMolecule(0, lig)
    lig = system[0]
    assert lig._sire_object.name().value() == "LIG", "first molecule is not the ligand"
    lig_charge = round(lig.charge().value())
    perturbed_resnum = system.getIndex(lig) + 1

    # ---- Bound leg: Boresch restraint (run 1's, used for every run, as a3fe does) --
    restraint, restraint_info = None, None
    if LEG == a3.LegType.BOUND:
        rfile = f"{eq_dirs[0]}/restraint_1.txt"
        jfile = f"{eq_dirs[0]}/restraint_1.json"
        need_obj = not all(os.path.exists(f"{ldir}/{s}/input/somd.pert") for s in grids)
        if need_obj or not (os.path.exists(rfile) and os.path.exists(jfile)):
            print("searching for Boresch restraint (equil_1 trajectory)", flush=True)
            restraint = search_restraint(inp, eq_dirs[0], system, prep_cfg, LEG)
            with open(rfile, "w") as f:
                f.write(restraint.toString(engine="SOMD"))
            # Analytical correction is what a3fe adds to the bound leg's dG.
            restraint_info = {
                "file": os.path.relpath(rfile, base),
                "correction_analytical_kcal": restraint.getCorrection(method="analytical").value(),
                "correction_numerical_kcal": restraint.getCorrection(method="numerical").value()}
            json.dump(restraint_info, open(jfile, "w"), indent=1)
        restraint_info = json.load(open(jfile))
        print(f"restraint correction (analytical): "
              f"{restraint_info['correction_analytical_kcal']:.3f} kcal/mol", flush=True)

    leg_man = {"lambda": {}, "a3fe_lambda": {}, "ligand_charge": lig_charge}
    if restraint_info:
        leg_man["restraint"] = restraint_info

    # ---- Job 2: SOMD run folders, one per (stage, λ, replicate) -------------------
    for name, grid in grids.items():
        stype = STAGE_TYPES[name]
        grid = [float(x) for x in grid]
        a3fe_grid = prep_cfg.lambda_values[LEG][stype]
        assert set(a3fe_grid) <= set(grid), f"{leg_name}/{name}: grid must contain a3fe's values"
        leg_man["lambda"][name] = grid
        leg_man["a3fe_lambda"][name] = a3fe_grid

        # Per-stage SOMD topology + perturbation file, written by BSS exactly as a3fe
        # does (setup_only, dummy λ/runtime, velocities dropped), then hoisted out of
        # the lambda_* dir BSS creates.
        sdir = f"{ldir}/{name}"
        sin = f"{sdir}/input"
        if not all(os.path.exists(f"{sin}/{f}") for f in ("somd.prm7", "somd.pert")):
            os.makedirs(sin, exist_ok=True)
            print(f"writing SOMD inputs for {leg_name}/{name} ({stype.bss_perturbation_type})",
                  flush=True)
            protocol = BSS.Protocol.FreeEnergy(runtime=0.001 * BSS.Units.Time.nanosecond,
                                               lam_vals=[0.0],
                                               perturbation_type=stype.bss_perturbation_type)
            BSS.FreeEnergy.AlchemicalFreeEnergy(system, protocol, engine="somd",
                                                restraint=restraint,
                                                work_dir=sin, setup_only=True,
                                                property_map={"velocity": "foo"})
            for f in glob.glob(f"{sin}/lambda_*/*"):
                if not f.endswith(".cfg"):
                    shutil.copy(f, sin)
            for dd in glob.glob(f"{sin}/lambda_*"):
                subprocess.run(["rm", "-rf", dd], check=True)

        # Stage config = a3fe's SomdConfig defaults, with the FULL dense grid as the
        # λ array so SOMD records u_kl at every window of this stage.
        scfg = a3.SomdConfig()
        if lig_charge != 0:
            # SOMD's charged-ligand correction requires PME (a3fe validates this);
            # a3fe's Leg.setup_stages suggests e.g. a 10 Å cutoff. SomdConfig
            # re-validates the whole model on every assignment, so PME must be set
            # BEFORE the charge, or the intermediate state (charge, no PME) is rejected.
            scfg.cutoff_type = "PME"
            scfg.cutoff_distance = float(cfg.get("cutoff_distance", 10.0))
        scfg.set_ligand_charge(lig_charge)
        if "energy_frequency" in cfg:
            scfg.energy_frequency = int(cfg["energy_frequency"])
        if "buffered_coords_freq" in cfg:
            scfg.buffered_coords_freq = int(cfg["buffered_coords_freq"])
        scfg.perturbed_residue_number = perturbed_resnum
        scfg.use_boresch_restraints = LEG == a3.LegType.BOUND
        scfg.turn_on_receptor_ligand_restraints = (LEG == a3.LegType.BOUND
                                                   and stype == a3.StageType.RESTRAIN)
        if LEG == a3.LegType.BOUND:
            scfg.load_boresch_restraints(f"{eq_dirs[0]}/restraint_1.txt")
        scfg.lambda_values = grid
        scfg.setup_lambda_arrays(stype)
        leg_man["somd"] = {"timestep_fs": scfg.timestep,
                           "energy_frequency": scfg.energy_frequency,
                           "sample_interval_ps": scfg.gradient_output_interval * 1000,
                           "temperature_K": scfg.analysis_temperature,
                           "cutoff_type": scfg.cutoff_type,
                           "cutoff_distance_A": scfg.cutoff_distance,
                           "ligand_charge_option": scfg.ligand_charge}

        for li, lam in enumerate(grid):
            for r in range(1, reps + 1):
                rd = f"{sdir}/lambda_{lam:.4f}/rep_{r}"
                manifest["runs"].append({"leg": leg_name, "stage": name, "lam": lam,
                                         "lam_index": li, "a3fe_window": lam in a3fe_grid,
                                         "rep": r, "dir": os.path.relpath(rd, base), "ns": ns})
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
    manifest["legs"][leg_name] = leg_man

# Legacy single-free-leg manifests keep their old top-level keys, which
# notebooks/replay.ipynb reads (lambda, a3fe_lambda, somd, leg).
if legacy:
    fl = manifest["legs"]["free"]
    manifest.update({"leg": "free", "lambda": fl["lambda"],
                     "a3fe_lambda": fl["a3fe_lambda"], "somd": fl["somd"]})

with open(f"{base}/manifest.json", "w") as f:
    json.dump(manifest, f, indent=1)
print(f"wrote {len(manifest['runs'])} run folders + {base}/manifest.json "
      f"({len(manifest['runs']) * ns:g} ns total)")
