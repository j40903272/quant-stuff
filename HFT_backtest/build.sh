#!/bin/bash

function EchoThenEval() {
    echo "${1}"
    eval "${1}"
}

mode="release"
perfect=false
check_memory=false
perf=false
coverage=false
full=true
cached=0
njobs=8
target=""
PROGNAME=$(basename $0)

function usage() {
cat <<EOF

  Usage: $PROGNAME [options]

  Options:

    -h, --help            Usage
    -c, --cache           Use cache to build
    -d, --debug           Debug mode
    -j, --jobs            Allow N jobs at once (default:$njobs)
    -p, --perfect         Build only with perfect alpha
    -m, --memory_check    Build with address sanitizer
    -t, --target          Build certain target not all
    -v, --coverage        Build with test coverage
    -f, --full            Build all targets

EOF
}


# ndevs=0
# if command -v git log > /dev/null; then
#     ndevs=`git log 2>/dev/null | grep Author | awk -F '[><]' '{print $2}'| sort | uniq -i | wc -l`
# fi
# njobs="$(expr $(nproc --all) / ${ndevs})"


for flag in "$@"; do
    case $flag in
        -h|--help)
            usage
            exit 0
            ;;
        -c|--cache)
            cached=1
            shift
            ;;
        -d|--debug)
            mode="debug"
            shift
            ;;
        -p|--perfect)
            perfect=true
            shift
            ;;
        -j|--jobs)
            njobs="$2"
            shift
            shift
            ;;
        -m|--memory_check)
            check_memory=true
            shift
            ;;
        -P|--perf)
            perf=true
            shift
            ;;
        -t|--target)
            target="$2"
            shift
            shift
            ;;
        -v|--coverage)
            coverage=true
            shift
            ;;
        -f|--full)
            full=true
            shift
            ;;
    esac
done

if [ ! -d "./build" ]; then
    comd="mkdir build"
    EchoThenEval "${comd}"
fi

comd="cd ./build"
EchoThenEval "${comd}"

if [ "${cached}" -eq 0 ]; then
    comd="make clean"
    EchoThenEval "${comd}"
fi

if [ "${perfect}" == true ]; then
    comd="cmake -DALPHA_PERFECT=true"
else
    comd="cmake -DALPHA_PERFECT=false"
fi

if [ "${coverage}" == true ]; then
    comd="${comd} -DCOVERAGE=true"
else
    comd="${comd} -DCOVERAGE=false"
fi

if [ "${check_memory}" == true ]; then
    comd="${comd} -DCHECK_MEMORY=true"
else
    comd="${comd} -DCHECK_MEMORY=false"
fi

if [ "${perf}" == true ]; then
    comd="${comd} -DPERF=true"
else
    comd="${comd} -DPERF=false"
fi

if [ "${mode}" == "debug" ]; then
    comd="${comd} -DCMAKE_BUILD_TYPE=Debug ../"
else
    comd="${comd} -DCMAKE_BUILD_TYPE=Release ../"
fi

if [ "${full}" == true ]; then
    comd="${comd} -DFULL=true ../"
else
    comd="${comd} -DFULL=false ../"
fi

EchoThenEval "${comd}"

if [[ $target ]]; then
    comd="make -j ${njobs} ${target}"
else
    comd="make -j ${njobs}"
fi

EchoThenEval "${comd}"