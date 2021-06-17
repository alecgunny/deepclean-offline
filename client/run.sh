#! /bin/bash -e

cmd="$1"
shift

CONDA_PREFIX=$HOME/conda

init() {
    # this line below should be written in giant
    # letters on the masthead of Anaconda's
    # documentation, but alas
    source $CONDA_PREFIX/etc/profile.d/conda.sh
    cd "${0%/*}"
}


if [[ "$cmd" == "install" ]]; then
    wget -q -O $HOME/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash $HOME/miniconda.sh -b -p $CONDA_PREFIX 2>/dev/null
    $CONDA_PREFIX/bin/conda update -q -n base -y conda 2>/dev/null
    $CONDA_PREFIX/bin/conda init 2>/dev/null
    rm $HOME/miniconda.sh
elif [[ "$cmd" == "create" ]]; then
    init
    conda env create -q -f environment.yaml
elif [[ "$cmd" == "list" ]]; then
    init
    conda list
elif [[ "$cmd" == "run" ]]; then
    init
    conda activate deepclean-offline-client
    python client.py "$@"
fi