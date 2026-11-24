#!/usr/bin/env bash
# One-time setup on a fresh RunPod pod (Community or Secure), after the laptop has
# pushed the repo with `sims/pods.sh push-code`:
#
#   bash /workspace/free-energy-resource-allocation/sims/pod_setup.sh
#
# Installs micromamba + the `fera` env (environment.yml) + a3fe 0.5.0 under
# /workspace, which is the pod's volume disk: it survives pod stop/restart, but not
# terminate. Re-running is safe: each step is skipped if already done.
# Needs a host driver with CUDA >= 13.2 (environment.yml pins cuda-version=13.2), so
# create GPU pods with minCudaVersion 13.2.
set -eo pipefail   # no -u: micromamba activation hooks reference unset vars

REPO=/workspace/free-energy-resource-allocation
export MAMBA_ROOT_PREFIX=/workspace/mamba
A3FE_TAG=0.5.0

stamp() { echo "[$(date '+%F %T')] $*"; }

if [ ! -x /workspace/bin/micromamba ]; then
    stamp "installing micromamba"
    mkdir -p /workspace/bin
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
        | tar -xj -C /workspace bin/micromamba
fi
eval "$(/workspace/bin/micromamba shell hook -s bash)"

if [ ! -d "$MAMBA_ROOT_PREFIX/envs/fera" ]; then
    # environment.yml pins cuda-version=13.2 (the toluene host's driver). The pin must
    # not exceed this host's driver CUDA, or OpenMM's CUDA platform fails
    # (CUDA_ERROR_UNSUPPORTED_PTX_VERSION), so re-pin it to the driver's version.
    # Pin = min(driver CUDA, 13.2): 13.2 is the version known to solve and run (13.5 broke
    # OpenMM's CUDA platform); below ~13 the openbiosim dev packages don't solve at all,
    # so only rent hosts whose driver reports CUDA >= 13.2.
    drv=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || true)
    pin=$(python3 -c "import sys; d=sys.argv[1]; print(min(d, '13.2', key=lambda v: tuple(map(int, v.split('.')))) if d else '13.2')" "$drv" 2>/dev/null || echo 13.2)
    drv_note=$drv
    sed "s/cuda-version=[0-9.]*/cuda-version=${pin}/" "$REPO/environment.yml" > /tmp/fera_env.yml
    stamp "creating env fera (cuda-version=${pin}, host driver CUDA ${drv_note:-unknown}; ~10-20 min)"
    micromamba create -y -f /tmp/fera_env.yml
    micromamba clean -a -y -q   # drop the package cache (most of the 41 GB toluene's took)
fi
micromamba activate fera

if ! python -c "import a3fe, sys; sys.exit(a3fe.__version__ != '$A3FE_TAG')" 2>/dev/null; then
    stamp "installing a3fe $A3FE_TAG"
    [ -d /workspace/a3fe ] || git clone -q https://github.com/michellab/a3fe.git /workspace/a3fe
    git -C /workspace/a3fe checkout -q "$A3FE_TAG"
    pip install -q --no-deps -e /workspace/a3fe
fi

stamp "checks"
python -c "import a3fe; print('a3fe', a3fe.__version__)"
which somd-freenrg gmx
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || echo "no GPU visible"
python -m openmm.testInstallation 2>&1 | tail -8
stamp "setup done"
