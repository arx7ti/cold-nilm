with import <nixpkgs> {};
let
  unstable = import <unstable> {};
  jupyter_latex_envs = python38.pkgs.buildPythonPackage rec {
    pname = "jupyter_latex_envs";
    version = "1.4.6";
    src = python38.pkgs.fetchPypi {
      inherit pname version;
      sha256 = "070a31eb2dc488bba983915879a7c2939247bf5c3b669b398bdb36a9b5343872";
    };
    doCheck = false;
    propagatedBuildInputs = with python38Packages; [
      ipython
      notebook
      nbconvert
      traitlets
      jupyter_core
    ];
  };
  jupyter_nbextensions_configurator = python38.pkgs.buildPythonPackage rec {
    pname = "jupyter_nbextensions_configurator";
    version = "0.4.1";
    src = fetchFromGitHub {
      owner  = "Jupyter-contrib";
      repo   = "jupyter_nbextensions_configurator";
      rev    = "5608b95c2b14998efc38294aae9f5d1b45cb6299";
      sha256 = "0w4kbzq16p3v5a0pnwbi6qcxg2kcqdbkpzc8am0zi5m0qjs7zxph";
    };
    doCheck = false;
    propagatedBuildInputs = with python38Packages; [
      jupyter_core
      jupyter_contrib_core
      pyyaml
      tornado
      lxml
      traitlets
    ];
  };
  jupyter_contrib_core = python38.pkgs.buildPythonPackage rec {
    pname = "jupyter_contrib_core";
    version = "0.3.3";
    src = python38.pkgs.fetchPypi {
      inherit pname version;
      sha256 = "e65bc0e932ff31801003cef160a4665f2812efe26a53801925a634735e9a5794";
    };
    doCheck = false;
    propagatedBuildInputs = with python38Packages; [
      tornado
      notebook
      traitlets
    ];
  };
  jupyter_highlight_selected_word = python38.pkgs.buildPythonPackage rec {
    pname = "jupyter_highlight_selected_word";
    version = "0.2.0";
    src = fetchFromGitHub {
      owner  = "jcb91";
      repo   = "jupyter_highlight_selected_word";
      rev    = "156a4fe84edd70880be1f5fb6a92d69796232e54";
      sha256 = "12x0azy149hv9rpmnijffab2irv4g3ac14lwp4x2w48z4byky0ra";
    };
    doCheck = false;
  };
  nbextensions = python38.pkgs.buildPythonPackage rec {
    pname = "jupyter_contrib_nbextensions";
    version = "0.5.1";
    src = fetchFromGitHub {
      owner  = "ipython-contrib";
      repo   = "jupyter_contrib_nbextensions";
      rev    = "882fbb011308b88217a656a46ef2bcb5a6031d84";
      sha256 = "1dm3zflnm8ssrpanxixzif6is3cwyadxap967r1b9hsh5xippnjq";
    };
    doCheck = false;
    propagatedBuildInputs = with python38Packages; [
      jupyter_nbextensions_configurator
      jupyter_contrib_core
      jupyter_latex_envs
      jupyter_highlight_selected_word
      tornado
      lxml
      traitlets
    ];
  };
  libs = [
    stdenv.cc.cc.lib
    xorg.libX11
    ncurses5
    linuxPackages.nvidia_x11
    libGL
    libzip
    glib
  ];
in
  stdenv.mkDerivation rec {
    name = "cold";
    dependencies = [
      python3
    ] ++ (with python38Packages; [
      pip
      virtualenv
      nbextensions
      ipykernel
      jupyter
      jupyterlab
      autopep8
      # librosa
    ]);
    nativeBuildInputs = [ cudatoolkit_10_1 ];
    buildInputs = dependencies;
    LD_LIBRARY_PATH = "${stdenv.lib.makeLibraryPath libs}";
    CUDA_PATH = "${cudatoolkit_10_1}";
    NUMEXPR_MAX_THREADS = 24;
    src = null;
    shellHook = ''
    export PIP_PREFIX=$(pwd)/_build/pip_packages
    export PYTHONPATH="$PIP_PREFIX/${pkgs.python3.sitePackages}:$PYTHONPATH"
    export PATH="$PIP_PREFIX/bin:$PATH"
    unset SOURCE_DATE_EPOCH
    if [ ! -d "./.venv" ]; then
      python -m venv .venv
    fi
    source .venv/bin/activate
    pip install --upgrade pip 
    pip install -r requirements.txt
    patchelf --set-interpreter "$(cat $NIX_CC/nix-support/dynamic-linker)" ./_build/pip_packages/lib/python3.8/site-packages/ray/core/src/ray/thirdparty/redis/src/redis-server
    patchelf --set-interpreter "$(cat $NIX_CC/nix-support/dynamic-linker)" ./_build/pip_packages/lib/python3.8/site-packages/ray/core/src/ray/gcs/gcs_server
    patchelf --set-interpreter "$(cat $NIX_CC/nix-support/dynamic-linker)" ./_build/pip_packages/lib/python3.8/site-packages/ray/core/src/ray/raylet/raylet
    patchelf --set-interpreter "$(cat $NIX_CC/nix-support/dynamic-linker)" ./_build/pip_packages/lib/python3.8/site-packages/ray/core/src/plasma/plasma_store_server
  '';
}
