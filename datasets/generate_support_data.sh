# COCO
## Change to your COCO dataset path
DATA_ROOT=/data/datasets/coco_2017
# cd coco
# # # Train
ln -s $DATA_ROOT/train2017 ./
ln -s $DATA_ROOT/val2017 ./
ln -s $DATA_ROOT/annotations ./
# # Few-shot samples
# cp /home/user/ws/FewX/support/final_submission/coco/overall/* new_annotations/

python3 1_split_filter.py ./ 
python3 3_gen_support_pool.py ./
# ## Train and test support
for shots in 1 2 3 5 10
do
	python 4_gen_support_pool_10_shot.py --shot ${shots}
done
cd ..

# VOC
## Change to your PASCAL VOC dataset path
# VOC is only used for testing, prepare images is enough
VOC_ROOT=/data/datasets/bowenli/pascal_voc
cd voc 
ln -s $VOC_ROOT/JPEGImages ./
ln -s $VOC_ROOT/annotations ./
# Few-shot samples
mkdir new_annotations
cp /home/user/ws/FewX/support/final_submission/voc/* new_annotations/
## Test support
for shots in 1 2 3 5
do
	python 4_gen_support_pool_10_shot.py --shot ${shots}
done

