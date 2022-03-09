### ----------------- COCO Testing Part ------------------------- ###
for shots in 1 2 3 5
do
	# generate few-shot support
	rm support_dir/support_feature.pkl
	CUDA_VISIBLE_DEVICES=0 python3 fsod_train_net.py --num-gpus 1 \
		--config-file configs/fsod/R101/test_R_101_C4_1x_coco${shots}.yaml \
		--eval-only 2>&1 | tee log/fsod_101_test_log_coco${shots}.txt
	# evaluation
	CUDA_VISIBLE_DEVICES=0,1,2,3 python3 fsod_train_net.py --num-gpus 4 \
		--config-file configs/fsod/R101/test_R_101_C4_1x_coco${shots}.yaml \
		--eval-only 2>&1 | tee log/fsod_101_test_log_coco${shots}.txt
done

### ----------------- VOC Testing Part ------------------------- ###
for shots in 1 2 3 5
do
	# generate few-shot support
	rm support_dir/support_feature.pkl
	CUDA_VISIBLE_DEVICES=0 python3 fsod_train_net.py --num-gpus 1 \
		--config-file configs/fsod/R101/test_R_101_C4_1x_voc${shots}.yaml \
		--eval-only 2>&1 | tee log/fsod_101_test_log_voc${shots}.txt

	CUDA_VISIBLE_DEVICES=0,1,2,3 python3 fsod_train_net.py --num-gpus 4 \
		--config-file configs/fsod/R101/test_R_101_C4_1x_voc${shots}.yaml \
		--eval-only 2>&1 | tee log/fsod_101_test_log_voc${shots}.txt
done
