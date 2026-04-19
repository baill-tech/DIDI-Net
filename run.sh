CUDA_VISIBLE_DEVICES=0 python train.py \
  --train_manifest data/manifest_train.json \
  --val_manifest data/manifest_val.json \
  --output_dir outputs/run \
  --batch_size 8 \
  --epochs 40 \
  --lr 1e-5 \
  --num_workers 0 \
  --image_size 256 \
  --template_size 224 \
  --sam_canvas_size 512 \
  --use_real_sam \
  --sam_model_alias sam_model \
  --sam_device cuda \
  --amp

CUDA_VISIBLE_DEVICES=0 python infer.py \
  --checkpoint outputs/run/checkpoints/epoch_001.pt \
  --scene_image data/example/scene.png \
  --template_image data/example/defect_source.png \
  --template_mask data/example/defect_source_mask.png \
  --loc_mask data/example/location_mask.png \
  --output_image outputs/infer/result.png \
  --output_grid outputs/infer/grid.png \
  --num_inference_steps 50 \
  --use_real_sam \
  --sam_model_alias sam_model \
  --sam_device cuda \
  --device cuda

CUDA_VISIBLE_DEVICES=0 python batch_infer_mvtec.py \
  --checkpoint outputs/run/checkpoints/epoch_001.pt \
  --mvtec_root data/mvtec \
  --output_dir outputs/mvtec_batch_infer \
  --pair_mode same_name_then_cyclic \
  --num_inference_steps 50 \
  --use_real_sam \
  --sam_model_alias sam_model \
  --sam_device cuda \
  --device cuda \
  --save_grid


CUDA_VISIBLE_DEVICES=0 python evaluate_metrics.py \
  --summary_json outputs/mvtec_batch_infer/summary.json \
  --output_dir outputs/mvtec_eval \
  --device cuda