timbreremap-train-sdss fit -c cfg/onset_mapping_808.yaml --model.preset cfg/presets/808_snare_1.json --data.onset_feature.frame_size 2048 --trainer.default_root_dir experiment/logs_mlp2048_train
timbreremap-train-sdss fit -c cfg/onset_mapping_808.yaml --model.preset cfg/presets/808_snare_2.json --data.onset_feature.frame_size 2048 --trainer.default_root_dir experiment/logs_mlp2048_train
timbreremap-train-sdss fit -c cfg/onset_mapping_808.yaml --model.preset cfg/presets/808_snare_3.json --data.onset_feature.frame_size 2048 --trainer.default_root_dir experiment/logs_mlp2048_train
timbreremap-train-sdss fit -c cfg/onset_mapping_808.yaml --model.preset cfg/presets/808_open_snare.json --data.onset_feature.frame_size 2048 --trainer.default_root_dir experiment/logs_mlp2048_train
timbreremap-train-sdss fit -c cfg/onset_mapping_808.yaml --model.preset cfg/presets/808_noisy_snare.json --data.onset_feature.frame_size 2048 --trainer.default_root_dir experiment/logs_mlp2048_train

timbreremap-train-sdss fit -c cfg/onset_mapping_808.yaml --model.preset cfg/presets/808_snare_1.json --trainer.default_root_dir experiment/logs_mlp_train
timbreremap-train-sdss fit -c cfg/onset_mapping_808.yaml --model.preset cfg/presets/808_snare_2.json --trainer.default_root_dir experiment/logs_mlp_train
timbreremap-train-sdss fit -c cfg/onset_mapping_808.yaml --model.preset cfg/presets/808_snare_3.json --trainer.default_root_dir experiment/logs_mlp_train
timbreremap-train-sdss fit -c cfg/onset_mapping_808.yaml --model.preset cfg/presets/808_open_snare.json --trainer.default_root_dir experiment/logs_mlp_train
timbreremap-train-sdss fit -c cfg/onset_mapping_808.yaml --model.preset cfg/presets/808_noisy_snare.json --trainer.default_root_dir experiment/logs_mlp_train

timbreremap-test experiment/logs_mlp2048_train/lightning_logs experiment/test_logs_mlp2048
timbreremap-test experiment/logs_mlp_train/lightning_logs experiment/test_logs_mlp
