TRAIN_TEST_SPLIT=navtest

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/mnt/zhaozc_workspace/project/FastWAM/data/navsim/maps"
export NAVSIM_EXP_ROOT="/mnt/zhaozc_workspace/project/FastWAM/evaluate_results"
export NAVSIM_DEVKIT_ROOT="/mnt/zhaozc_workspace/project/FastWAM/navsim"
export OPENSCENE_DATA_ROOT="/mnt/zhaozc_workspace/project/FastWAM/data/navsim"

CACHE_PATH="/mnt/zhaozc_workspace/project/FastWAM/data/metric_cache_navtest"

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
train_test_split=$TRAIN_TEST_SPLIT \
cache.cache_path=$CACHE_PATH