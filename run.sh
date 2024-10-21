SRC_DIR="../depth_pro"
REF_DIR="../depth_rs"
PARAM_PATH="../cam_params.json"
OUTPUT_DIR="../depth_adjusted"
MAX_DEPTH=8.0
MIN_DEPTH=0.0

python main.py \
    --ref_dir $REF_DIR \
    --src_dir $SRC_DIR \
    --param_path $PARAM_PATH \
    --output_dir $OUTPUT_DIR \
    --max_depth $MAX_DEPTH \
    --min_depth $MIN_DEPTH