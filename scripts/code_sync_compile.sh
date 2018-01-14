#! /bin/bash

set -e

# Reset
Color_Off='\033[0m'       # Text Reset

# Regular Colors
Black='\033[0;30m'        # Black
Red='\033[0;31m'          # Red
Green='\033[0;32m'        # Green
Yellow='\033[0;33m'       # Yellow
Blue='\033[0;34m'         # Blue
Purple='\033[0;35m'       # Purple
Cyan='\033[0;36m'         # Cyan
White='\033[0;37m'        # White

# Bold
BBlack='\033[1;30m'       # Black
BRed='\033[1;31m'         # Red
BGreen='\033[1;32m'       # Green
BYellow='\033[1;33m'      # Yellow
BBlue='\033[1;34m'        # Blue
BPurple='\033[1;35m'      # Purple
BCyan='\033[1;36m'        # Cyan
BWhite='\033[1;37m'       # White

#################################### utility script function
function log {
    echo -e  "${BBlue}------------------------- $1 -------------------------${Color_Off}"
}

function log_multi {
    echo ""
    echo -e  "${BBlue}------------------------- -------------------------${Color_Off}"
    echo -e  "${BBlue}$1${Color_Off}"
    echo -e  "${BBlue}------------------------- -------------------------${Color_Off}"
}

function exitIfError {
    if [[ $? -ne 0 ]] ; then
       log_multi "Errors detected. Exiting script. The software might have not been successfully installed."
        exit 1
    fi
}

function isinstalled_which {
    if which "$@" >/dev/null 2>&1; then
	true
    else
	false
    fi
}


function mkdir_if_not_exist {
    # usage: mkdir_if_not_exist path [1 - remove if existed]
    if [[ -d $1 && $# -eq 2 ]]; then
	if [ $2 == 1 ]; then
	    rm -R $1
	fi
    fi

    if [ ! -d $1 ]; then
        log "Create a directory $1"
        mkdir -p $1
        if [ $? -ne 0 ] ; then
            log "Failed to create $1"
            exit
        fi
    fi
}

function download {
    BN=`basename "$1"`
    if [ ! -f $BN ]; then
	echo "Downloading $BN ..."
        #wget --quiet "$U" -O $BN
	wget "$1"
	if [ ! -f $BN ]; then
            echo "Failed to download $BN"
            exit
	fi
    fi
}

function git_a_repo {
    #parameters: directory git_repo_url [repo_base] [checkout]
    #echo "Function name:  ${FUNCNAME}"
    #echo "The number of positional parameter : $#"
    #echo "All parameters or arguments passed to the function: '$@'"
    #FILERELPATH="${1}"
    #FILEPATH="`abspath ${FILERELPATH}`"
    #FILEDIR="`dirname "${FILEPATH}"`/"
    #FILENAME="`basename "${FILEPATH}"`"
    #FILEBASE="${FILENAME%.*}"
    #FILEEXT="${FILENAME:${#FILEBASE}}"

    cd $1
    if [ $# == 3 ]; then
        repo_base=$3
    else
        repo_basename=`basename $2`
        repo_base="${repo_basename%.*}"
    fi
    if [ ! -d "$repo_base" ]; then
        log "clone $repo_base"
        git clone $2 ./$repo_base --recursive
        if [  $# == 4 ]; then
            cd $repo_base
            git checkout $4
        fi
	true
    else
        log "update $repo_base"
        cd $repo_base
        git pull  | grep "Already up-to-date." #--recurse-submodules
	updated=$?
	git submodule update --init --recursive
	if [[ $updated -ne 0 ]]; then
	    true
	else
	    false
	fi
    fi
}

function compare_files_in_two_dir {
    if [ $# != 3 ]; then
	log "Usage: src_dir dst_dir ext_name"
	exit 1
    fi
    SRC_DIR=$1
    DST_DIR=$2
    EXT_NAME=$3
    for file in $SRC_DIR/*
    do
        #echo "$file"
	fname=`basename $file`
	fbase="${fname%.*}"
	fext="${fname:${#fbase}}"
	if [ "$fext" == "$EXT_NAME" ]; then
	    if [ ! -f "$DST_DIR/$fname" ]; then
		cp -f $file $DST_DIR
	    else
		if [ "$file" -nt "$DST_DIR/$fname" ]; then
		    cp -f $file $DST_DIR
		else
		    cp -f $file $file.bak
		    cp -f $DST_DIR/$fname $file
		fi
	    fi
	fi
    done

}

function cp_with_prefix_added {
    if [ $# != 4 ]; then
	log "Usage: dst_dir prefix src_dir extname"
	exit 1
    fi
    DST_DIR=$1
    PREFIX=$2
    SRC_DIR=$3
    EXT_NAME=$4

    for file in $SRC_DIR/*
    do
	#echo "$file"
	fname=`basename $file`
	fbase="${fname%.*}"
	fext="${fname:${#fbase}}"
	if [ "$fext" == "$EXT_NAME" ]; then
	    cp -f $file $DST_DIR/$PREFIX\_$fname
	fi
    done
}

function cp_dir_struct {
    if [ $# != 2 ]; then
	log "Usage: dst_root_dir src_root_dir"
	exit 1
    fi

    DST_ROOT_DIR=$1
    SRC_ROOT_DIR=$2

    pushd $SRC_ROOT_DIR
    shopt -s dotglob
    find * -prune -type d | while read d; do
	mkdir_if_not_exist $DST_ROOT_DIR/$d
	if [ -f $SRC_ROOT_DIR/$d/__init__.py ]; then
	    cp $SRC_ROOT_DIR/$d/__init__.py $DST_ROOT_DIR/$d
	fi
    done
    popd
}

function remove_files {
    if [ $# != 2 ]; then
	log "Usage: ref_dir rm_dir"
	exit 1
    fi
    REF_DIR=$1
    RM_DIR=$2
    for file in $REF_DIR/*
    do
	fname=`basename $file`
	if [ "$fname" != "*" ]; then
	    fileCopy=$RM_DIR/$fname
	    if [ -f $fileCopy ]; then
		rm $fileCopy
	    fi
	fi
    done

}

function compile_mxnet_with_contrib {
    if [ $# != 4 ]; then
	log "Usage: src_operators_dir mxnet_root_dir num_cores src_external_mxnet"
	exit 1
    fi
    src_operator=$1
    mxnet_root_dir=$2
    num_cores=$3
    mxnetWrapper=$4
    mxnet_contrib=$mxnet_root_dir/src/operator/contrib
    if [ ! -d $src_operator ]; then
	log "not found $src_operator"
	exit 1
    fi
    if [ ! -d $mxnet_contrib ]; then
	log "not found $mxnet_contrib"
	exit 1
    fi

    # copy extra operators
    if [ "$(ls -A $src_operator)" ]; then
	 cp -rf  $src_operator/* $mxnet_contrib
    fi

    # compile mxnet
    log "Compile mxnet"
    #cp $2/make/config.mk $2
    #echo "USE_CUDA=1" >>$2/config.mk
    #echo "USE_CUDA_PATH=/usr/local/cuda" >>$2/config.mk
    #echo "USE_CUDNN=1" >>$2/config.mk
    # echo "EXTRA_OPERATORS = $SRC_PATH/Deformable-ConvNets/rfcn/operator_cxx" >> $SRC_PATH/mxnet/config.mk
    cd $mxnet_root_dir
    colormake -j$num_cores USE_OPENCV=1 USE_BLAS=openblas USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1 USE_PROFILER=1 USE_GPERFTOOLS=1 2>&1 | grep -E --color=always 'error|$' #
    # USE_DIST_KVSTORE
    # USE_GPERFTOOLS
    
    #remove_files $src_operator $mxnet_contrib

    #exitIfError

    # copy mxnet python wrapper  (application specific import)
    log "Copy mxnet python to $mxnetWrapper"
    mkdir_if_not_exist $mxnetWrapper
    cp -rf $mxnet_root_dir/python/mxnet $mxnetWrapper
    cp -f $mxnet_root_dir/lib/*  $mxnetWrapper/mxnet
    cp -f $mxnet_root_dir/nnvm/lib/*   $mxnetWrapper/mxnet
}

####################################

log "Checking Ubuntu Version"
ubuntu_version="$(lsb_release -r)"
echo "Ubuntu $ubuntu_version"
if [[ $ubuntu_version == *"14."* ]]; then
    ubuntu_le_14=true
elif [[ $ubuntu_version == *"16."* || $ubuntu_version == *"15."* || $ubuntu_version == *"17."* || $ubuntu_version == *"18."* ]]; then
    ubuntu_le_14=false
else
    echo "Ubuntu release older than version 14. This installation script might fail."
    ubuntu_le_14=true
fi
exitIfError
log "Ubuntu Version Checked"
echo ""

log "Checking Number of Processors"
NUM_CORES=$(grep -c ^processor /proc/cpuinfo 2>/dev/null || sysctl -n hw.ncpu)
echo "$NUM_CORES cores"
exitIfError
log "Number of Processors Checked"
echo ""

####################################

SRC_PATH=$(pwd)/..
echo "SRC_PATH $SRC_PATH"

updateGit=1

# colorize make output
if ! isinstalled_which 'colormake' ; then
    sudo apt-get install colormake
fi


#################################### get mxnet
mxnet_updated=false
if [ $updateGit == 1 ]; then
    log "Get the latest mxnet repo"
    if git_a_repo $SRC_PATH https://github.com/apache/incubator-mxnet.git mxnet; then
	mxnet_updated=true
    fi
fi

#################################### D2RGM
#git_a_repo $SRC_PATH  https://github.ncsu.edu/twu19/D2RGM.git

# compile
cd $SRC_PATH/AOGNet
mkdir_if_not_exist $SRC_PATH/AOGNet/aognet/operator_cxx
#chmod +x ./init.sh
#./init.sh
compile_mxnet_with_contrib $SRC_PATH/AOGNet/aognet/operator_cxx $SRC_PATH/mxnet $NUM_CORES $SRC_PATH/AOGNet
cp $SRC_PATH/AOGNet/mxnet $SRC_PATH/AOGNet/image_classification/mxnet -r

###################################
use_jupyter=false
if [[ $use_jupyter == true ]]; then
    # set up jupyter notebook
    cd $EXAMPLE

    log_multi "Please run the commond below in your local machine:  ssh -N -f -L 127.0.0.1:8889:127.0.0.1:8889 microway@10.76.80.31"

    jupyter notebook --no-browser --port=8889
fi
