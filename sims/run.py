"""
Run every SOMD folder listed in a prep manifest, several at once on one GPU.

    python sims/run.py configs/toluene.yaml --jobs 4
    python sims/run.py configs/mdm2.yaml --jobs 1 --rep 2     # one pod per replicate

- Order is replicate-major (all of rep 1, then rep 2, ...; within a replicate,
  leg then stage then λ), so a partial run still yields complete replicates.
- Resumable: a run is done when somd.out reports "Simulation took" and the last
  simfile.dat step equals the expected step count. Anything else found in a
  folder is treated as an interrupted run: its outputs are wiped and it restarts
  from scratch (no partial-tape stitching).
- Every finished run appends a line to {base}/run_log[_<pod id>].jsonl (wall time, ns/day);
  aggregate ns/day for this session is printed as runs complete.
"""
import argparse, glob, json, os, signal, subprocess, sys, threading, time, yaml
from concurrent.futures import ThreadPoolExecutor, as_completed

OUTPUTS = ["simfile.dat", "simfile_equilibrated.dat", "moves.dat", "gradients.dat",
           "gradients.s3", "sim_restart.s3", "sim_restart.s3.previous", "latest.pdb",
           "traj*.dcd", "*.s3", "*.s3.previous", "somd.out", "somd.err"]

ap = argparse.ArgumentParser()
ap.add_argument("config")
ap.add_argument("--jobs", type=int, default=4, help="concurrent SOMD processes")
ap.add_argument("--leg", choices=["bound", "free"])
ap.add_argument("--stage", choices=["restrain", "discharge", "vanish"])
ap.add_argument("--lam", type=float, help="only this λ value")
ap.add_argument("--rep", type=int, help="only this replicate")
ap.add_argument("--max-runs", type=int, help="stop after launching this many")
ap.add_argument("--dry-run", action="store_true")
args = ap.parse_args()

cfg = yaml.safe_load(open(args.config))
base = f"{cfg['workdir']}/{cfg['name']}"
man = json.load(open(f"{base}/manifest.json"))
# Legacy (toluene) manifests have one top-level "somd" block; multi-leg ones have one per leg.
timestep_fs = {leg: v["somd"]["timestep_fs"] for leg, v in man.get("legs", {}).items()}
if not timestep_fs:
    timestep_fs = {"free": man["somd"]["timestep_fs"]}
leg_order = {"bound": 0, "free": 1}
stage_order = {"restrain": 0, "discharge": 1, "vanish": 2}


def steps_expected(r):
    return round(r["ns"] * 1e6 / timestep_fs[r.get("leg", "free")])


def last_step(path):
    try:
        out = subprocess.check_output(["tail", "-n", "1", path]).decode().split()
        return int(out[0]) if out and not out[0].startswith("#") else -1
    except (subprocess.CalledProcessError, ValueError, IndexError):
        return -1


def is_done(r):
    rd = f"{base}/{r['dir']}"
    try:
        ok = "Simulation took" in open(f"{rd}/somd.out").read()
    except FileNotFoundError:
        return False
    return ok and last_step(f"{rd}/simfile.dat") == steps_expected(r)


def clean(rd):
    for pat in OUTPUTS:
        for f in glob.glob(f"{rd}/{pat}"):
            os.remove(f)


runs = sorted(man["runs"], key=lambda r: (r["rep"], leg_order[r.get("leg", "free")],
                                          stage_order[r["stage"]], r["lam_index"]))
runs = [r for r in runs
        if (args.leg is None or r.get("leg", "free") == args.leg)
        and (args.stage is None or r["stage"] == args.stage)
        and (args.lam is None or abs(r["lam"] - args.lam) < 1e-9)
        and (args.rep is None or r["rep"] == args.rep)]
todo = [r for r in runs if not is_done(r)]
if args.max_runs is not None:
    todo = todo[:args.max_runs]
print(f"{len(runs)} selected, {len(runs) - len(todo)} already done, {len(todo)} to run "
      f"({sum(r['ns'] for r in todo):g} ns) with {args.jobs} concurrent", flush=True)
if args.dry_run or not todo:
    for r in todo[:10]:
        print("  would run", r["dir"])
    sys.exit(0)

procs, lock, stopping = set(), threading.Lock(), threading.Event()
# One log per pod, so logs from pods recording different replicates don't collide
# when they're pulled into one folder.
pod = os.environ.get("RUNPOD_POD_ID")
log_path = f"{base}/run_log_{pod}.jsonl" if pod else f"{base}/run_log.jsonl"


def run_one(r):
    rd = f"{base}/{r['dir']}"
    clean(rd)
    t0 = time.time()
    with open(f"{rd}/somd.out", "w") as out, open(f"{rd}/somd.err", "w") as err:
        p = subprocess.Popen(["somd-freenrg", "-C", "somd.cfg", "-l", str(r["lam"]), "-p", "CUDA"],
                             cwd=rd, stdout=out, stderr=err, start_new_session=True)
        with lock:
            procs.add(p)
        rc = p.wait()
        with lock:
            procs.discard(p)
    wall = time.time() - t0
    ok = rc == 0 and is_done(r)
    rec = {"dir": r["dir"], "leg": r.get("leg", "free"), "stage": r["stage"], "lam": r["lam"], "rep": r["rep"],
           "ns": r["ns"], "rc": rc, "ok": ok, "wall_s": round(wall, 1),
           "ns_per_day": round(r["ns"] / wall * 86400, 1), "finished": time.time()}
    with lock:
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
    return rec


def shutdown(*_):
    stopping.set()
    with lock:
        for p in procs:
            os.killpg(p.pid, signal.SIGTERM)
    print("\ninterrupted: killed running SOMD jobs; rerun to resume", flush=True)


signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)

t_start, ns_done, n_ok, n_fail = time.time(), 0.0, 0, 0
with ThreadPoolExecutor(max_workers=args.jobs) as ex:
    futs = []
    for r in todo:
        futs.append(ex.submit(lambda r=r: None if stopping.is_set() else run_one(r)))
    for fut in as_completed(futs):
        rec = fut.result()
        if rec is None:
            continue
        if rec["ok"]:
            n_ok += 1
            ns_done += rec["ns"]
        else:
            n_fail += 1
        el = time.time() - t_start
        agg = ns_done / el * 86400
        left = len(todo) - n_ok - n_fail
        eta = (left * man["ns_per_tape"]) / agg * 24 if agg > 0 else float("nan")  # hours
        print(f"[{n_ok + n_fail}/{len(todo)}] {'ok  ' if rec['ok'] else 'FAIL'} {rec['dir']} "
              f"{rec['wall_s']:.0f}s ({rec['ns_per_day']:.0f} ns/day/run) | "
              f"aggregate {agg:.0f} ns/day, ETA {eta:.1f} h", flush=True)

el = time.time() - t_start
print(f"done: {n_ok} ok, {n_fail} failed, {ns_done:g} ns in {el / 3600:.2f} h "
      f"= {ns_done / el * 86400:.0f} ns/day aggregate")
sys.exit(1 if n_fail else 0)
