#!/usr/bin/env bash

ar=
build_dir=
build_type=
cc=
cflags=
compiler=
compiler_version=
cxx=
cxxflags=
destdir=
jobs=
nvcc=
ranlib=
sanitize=
verbose=

while (( $# )); do

  # Options that require an argument
  if [[ "$1" =~ ^--(ar|build-dir|cc|cflags|cxx|cxxflags|destdir|jobs|nvcc|ranlib|sanitize)(=(.*))?$ ]]; then
    if [[ "${BASH_REMATCH[-2]}" ]]; then
      arg="${BASH_REMATCH[-1]}"
    else
      shift
      arg="$1"
    fi

    case "${BASH_REMATCH[1]}" in
    'ar') ar="$arg" ;;
    'build-dir') build_dir="$arg" ;;
    'cc') cc="$arg" ;;
    'cflags') cflags="$arg" ;;
    'cxx') cxx="$arg" ;;
    'cxxflags') cxxflags="$arg" ;;
    'destdir') destdir="$arg" ;;
    'jobs') jobs="$arg" ;;
    'nvcc') nvcc="$arg" ;;
    'ranlib') ranlib="$arg" ;;
    'sanitize')
      for a in ${arg//,/ }; do
        sanitize="$sanitize -fsanitize=$a"
      done
      ;;
    *) ;;
    esac

  # Options that don't accept an argument
  elif [[ "$1" =~ ^--(debug|quiet|release|verbose)$ ]]; then
    case "${BASH_REMATCH[1]}" in
    'quiet') verbose='OFF' ;;
    'verbose') verbose='ON' ;;
    *) build_type="${BASH_REMATCH[1]}" ;;
    esac

  # Compiler selection accepts an optional version argument
  elif [[ "$1" =~ ^--(clang|gcc)(=([1-9].*))?$ ]]; then
    compiler="${BASH_REMATCH[1]}"

    if [[ "${BASH_REMATCH[-2]}" ]]; then
      compiler_version="-${BASH_REMATCH[-1]}"
    elif [[ "$2" =~ ^[1-9] ]]; then
      shift
      compiler_version="-$1"
    else
      compiler_version=
    fi
  fi

  shift

done

if (( jobs < 1 )); then
  jobs="$(nproc)"
  (( jobs > 0 )) || jobs=1
fi

[[ "$build_type" ]] || build_type=release
build_type_lc="${build_type,,}"
build_type="${build_type_lc^}"

if [[ "$build_dir" ]]; then
  build_dir="$(realpath "$build_dir")"
else
  build_dir="$(realpath "$(dirname "$0")")/build_$build_type_lc"
fi

if [[ "$destdir" ]]; then
  destdir="$(realpath "$destdir")"
else
  destdir="$(realpath "$(dirname "$0")")/install_$build_type_lc"
fi

have_cflags=$((${#cflags} != 0))
have_cxxflags=$((${#cxxflags} != 0))

(( have_cflags   )) || cflags="-march=native -mtune=native -Wall -Wextra"
(( have_cxxflags )) || cxxflags="-march=native -mtune=native -Wall -Wextra"
[[ "$compiler"   ]] || compiler='gcc'

case "$compiler" in
'clang')
  [[ "$cc"  ]] || cc="clang$compiler_version"
  [[ "$cxx" ]] || cxx="clang++$compiler_version"
  if [[ "$compiler_version" ]]; then
    [[ "$ar"     ]] || ar="llvm-ar$compiler_version"
    [[ "$ranlib" ]] || ranlib="llvm-ranlib$compiler_version"
  fi
  (( have_cflags   )) || cflags="$cflags${cflags:+ }-Weverything -Wno-covered-switch-default"
  (( have_cxxflags )) || cxxflags="$cxxflags${cxxflags:+ }-Weverything -Wno-covered-switch-default -Wno-c++98-compat"
  ;;

'gcc')
  [[ "$cc"     ]] || cc="gcc$compiler_version"
  [[ "$cxx"    ]] || cxx="g++$compiler_version"
  [[ "$ar"     ]] || ar="gcc-ar$compiler_version"
  [[ "$ranlib" ]] || ranlib="gcc-ranlib$compiler_version"
  ;;
*) ;;
esac

[[ "$nvcc" ]] || nvcc='/usr/local/cuda/bin/nvcc'
[[ "$verbose" ]] || verbose='ON'

bypass_vcpkg=true
force_cpp_build=false

if [[ "$OSTYPE" == "darwin"* ]]; then
  vcpkg_triplet="x64-osx"
else
  vcpkg_triplet="x64-linux"
fi

if [[ ! -z "${VCPKG_ROOT}" ]] && [ -d ${VCPKG_ROOT} ] && [ ! "$bypass_vcpkg" = true ]
then
  vcpkg_path="${VCPKG_ROOT}"
  vcpkg_define="-DCMAKE_TOOLCHAIN_FILE=${vcpkg_path}/scripts/buildsystems/vcpkg.cmake"
  vcpkg_triplet_define="-DVCPKG_TARGET_TRIPLET=$vcpkg_triplet"
  echo "Found vcpkg in VCPKG_ROOT: ${vcpkg_path}"
  additional_defines="-DBUILD_SHARED_LIBS=OFF"
elif [[ ! -z "${WORKSPACE}" ]] && [ -d ${WORKSPACE}/vcpkg ] && [ ! "$bypass_vcpkg" = true ]
then
  vcpkg_path="${WORKSPACE}/vcpkg"
  vcpkg_define="-DCMAKE_TOOLCHAIN_FILE=${vcpkg_path}/scripts/buildsystems/vcpkg.cmake"
  vcpkg_triplet_define="-DVCPKG_TARGET_TRIPLET=$vcpkg_triplet"
  echo "Found vcpkg in WORKSPACE/vcpkg: ${vcpkg_path}"
  additional_defines="-DBUILD_SHARED_LIBS=OFF"
elif [ ! "$bypass_vcpkg" = true ]
then
  (>&2 echo "darknet is unsupported without vcpkg, use at your own risk!")
fi

if [ "$force_cpp_build" = true ]
then
  additional_build_setup="-DBUILD_AS_CPP:BOOL=TRUE"
fi

additional_build_setup="$additional_build_setup \
	-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
	-DCMAKE_COLOR_MAKEFILE=ON \
	-DCMAKE_VERBOSE_MAKEFILE=$verbose \
	-DCUDA_VERBOSE_BUILD=$verbose"

mkdir -p "$build_dir" &&
cd "$build_dir" &&
cmake -v .. -DCMAKE_BUILD_TYPE="$build_type" \
            -DCMAKE_INSTALL_PREFIX="$destdir" \
            -DCMAKE_C_COMPILER="$cc" \
            -DCMAKE_CXX_COMPILER="$cxx" \
            -DCMAKE_C_COMPILER_AR="$ar" \
            -DCMAKE_CXX_COMPILER_AR="$ar" \
            -DCMAKE_C_COMPILER_RANLIB="$ranlib" \
            -DCMAKE_CXX_COMPILER_RANLIB="$ranlib" \
            ${vcpkg_define} \
            ${vcpkg_triplet_define} \
            ${additional_defines} \
            ${additional_build_setup} \
	    -DCMAKE_CUDA_COMPILER="$nvcc" \
            -DCMAKE_C_FLAGS="$cflags $sanitize" \
            -DCMAKE_CXX_FLAGS="$cxxflags $sanitize" &&
mkdir -p "$destdir" &&
cmake --build . --target install -- -j${jobs}
rm -f DarknetConfig.cmake
rm -f DarknetConfigVersion.cmake
cd ..
cp cmake/Modules/*.cmake "$destdir/share/darknet/"
