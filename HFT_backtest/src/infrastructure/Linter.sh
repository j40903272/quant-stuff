
#!/bin/bash
COLOR_REST='\e[0m';
COLOR_BLUE='\e[1;34m';
COLOR_RED='\e[1;31m';
COLOR_GREEN='\e[1;32m';

function usage() {
    echo "Usage: ./build.sh"
    echo "  -h, --help      usage"
    echo "  -r, --reformat  reformat source file"
}

reformat=false
diff_bin=""
clang_bin=""

if command -v diff > /dev/null; then
    diff_bin=$(command -v diff)
else
    echo "cannot find diff binary on this machine"
    exit 1
fi

if command -v clang-format > /dev/null; then
    clang_bin=$(command -v clang-format)
else
    echo "cannot find clang-format binary on this machine"
    exit 1
fi
for flag in "$@"; do
    case $flag in
        -h|--help)
            usage
            exit 0
            ;;
        -r|--reformat)
            reformat=true
            shift
            ;;
    esac
done

source_files=`find . * | grep -v -e "^\." | grep -v -e "^build" | grep -v -e "^personal" | grep -e "\.h$\|\.c$\|\.cpp$\|\.hpp$" | grep -v "gmock\|gtest\|json\|spdlog\|protobuf\|websocketpp\|benchmark\|eigen"`
return_code=0
for f in ${source_files}; do
    check_cmd="${diff_bin} -u <(cat ${f}) <(${clang_bin} ${f} -style=file)"
    if ! diff=$(bash -c "${check_cmd}"); then
        echo -e "----- In file ${COLOR_BLUE}${f}${COLOR_REST} -----"
        echo "${diff}" | tail -n +4
        ((return_code=return_code+1))
        if [ "${reformat}" == true ]; then
            bash -c "${clang_bin} ${f} -i -style=file"
            echo -e "----- Auto-format ${COLOR_BLUE}${f}${COLOR_REST} done. -----"
        fi
    fi
done

if [ ${return_code} -ne 0 ]; then
    echo -e "........ linter ${COLOR_RED}FAILED${COLOR_REST}"
else
    echo -e "........ linter ${COLOR_GREEN}PASSED${COLOR_REST}"
fi

exit ${return_code}
