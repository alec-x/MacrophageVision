
md -Force .\output\test\efficientnet_b0


python .\ML\timm_alt_models\inference_alt.py .\data\processed\dataset_split\validation\M0 --model efficientnet_b0 --input-size 3 96 96 --checkpoint .\output\train\20220202-134654-efficientnet_b0-96\model_best.pth.tar --num-classes 3 -b 50 --topk 1 --output_dir .\output\test\efficientnet_b0 --output_name ./M0
python .\ML\timm_alt_models\inference_alt.py .\data\processed\dataset_split\validation\M1 --model efficientnet_b0 --input-size 3 96 96 --checkpoint .\output\train\20220202-134654-efficientnet_b0-96\model_best.pth.tar --num-classes 3 -b 50 --topk 1 --output_dir .\output\test\efficientnet_b0 --output_name ./M1
python .\ML\timm_alt_models\inference_alt.py .\data\processed\dataset_split\validation\M2 --model efficientnet_b0 --input-size 3 96 96 --checkpoint .\output\train\20220202-134654-efficientnet_b0-96\model_best.pth.tar --num-classes 3 -b 50 --topk 1 --output_dir .\output\test\efficientnet_b0 --output_name ./M2

md -Force '.\output\test\efficientnet_b2'
python .\ML\timm_alt_models\inference_alt.py .\data\processed\dataset_split\validation\M0 --model efficientnet_b2 --input-size 3 96 96 --checkpoint .\output\train\20220202-193206-efficientnet_b2-96\model_best.pth.tar --num-classes 3 -b 50 --topk 1 --output_dir .\output\test\efficientnet_b2 --output_name ./M0
python .\ML\timm_alt_models\inference_alt.py .\data\processed\dataset_split\validation\M1 --model efficientnet_b2 --input-size 3 96 96 --checkpoint .\output\train\20220202-193206-efficientnet_b2-96\model_best.pth.tar --num-classes 3 -b 50 --topk 1 --output_dir .\output\test\efficientnet_b2 --output_name ./M1
python .\ML\timm_alt_models\inference_alt.py .\data\processed\dataset_split\validation\M2 --model efficientnet_b2 --input-size 3 96 96 --checkpoint .\output\train\20220202-193206-efficientnet_b2-96\model_best.pth.tar --num-classes 3 -b 50 --topk 1 --output_dir .\output\test\efficientnet_b2 --output_name ./M2

md -Force '.\output\test\inception_v4'
python .\ML\timm_alt_models\inference_alt.py .\data\processed\dataset_split\validation\M0 --model inception_v4 --input-size 3 96 96 --checkpoint .\output\train\20220203-121645-inception_v4-96\model_best.pth.tar --num-classes 3 -b 50 --topk 1 --output_dir .\output\test\inception_v4 --output_name ./M0
python .\ML\timm_alt_models\inference_alt.py .\data\processed\dataset_split\validation\M1 --model inception_v4 --input-size 3 96 96 --checkpoint .\output\train\20220203-121645-inception_v4-96\model_best.pth.tar --num-classes 3 -b 50 --topk 1 --output_dir .\output\test\inception_v4 --output_name ./M1
python .\ML\timm_alt_models\inference_alt.py .\data\processed\dataset_split\validation\M2 --model inception_v4 --input-size 3 96 96 --checkpoint .\output\train\20220203-121645-inception_v4-96\model_best.pth.tar --num-classes 3 -b 50 --topk 1 --output_dir .\output\test\inception_v4 --output_name ./M2

md -Force '.\output\test\pnasnet'
python .\ML\timm_alt_models\inference_alt.py .\data\processed\dataset_split\validation\M0 --model pnasnet5large --input-size 3 96 96 --checkpoint .\output\train\20220203-103440-pnasnet5large-96\model_best.pth.tar --num-classes 3 -b 50 --topk 1 --output_dir .\output\test\pnasnet --output_name ./M0
python .\ML\timm_alt_models\inference_alt.py .\data\processed\dataset_split\validation\M1 --model pnasnet5large --input-size 3 96 96 --checkpoint .\output\train\20220203-103440-pnasnet5large-96\model_best.pth.tar --num-classes 3 -b 50 --topk 1 --output_dir .\output\test\pnasnet --output_name ./M1
python .\ML\timm_alt_models\inference_alt.py .\data\processed\dataset_split\validation\M2 --model pnasnet5large --input-size 3 96 96 --checkpoint .\output\train\20220203-103440-pnasnet5large-96\model_best.pth.tar --num-classes 3 -b 50 --topk 1 --output_dir .\output\test\pnasnet --output_name ./M2

md -Force '.\output\test\resnext'
python .\ML\timm_alt_models\inference_alt.py .\data\processed\dataset_split\validation\M0 --model resnext101_32x8d --input-size 3 96 96 --checkpoint .\output\train\20220203-124112-resnext101_32x8d-96\model_best.pth.tar --num-classes 3 -b 50 --topk 1 --output_dir .\output\test\resnext --output_name ./M0
python .\ML\timm_alt_models\inference_alt.py .\data\processed\dataset_split\validation\M1 --model resnext101_32x8d --input-size 3 96 96 --checkpoint .\output\train\20220203-124112-resnext101_32x8d-96\model_best.pth.tar --num-classes 3 -b 50 --topk 1 --output_dir .\output\test\resnext --output_name ./M1
python .\ML\timm_alt_models\inference_alt.py .\data\processed\dataset_split\validation\M2 --model resnext101_32x8d --input-size 3 96 96 --checkpoint .\output\train\20220203-124112-resnext101_32x8d-96\model_best.pth.tar --num-classes 3 -b 50 --topk 1 --output_dir .\output\test\resnext --output_name ./M2