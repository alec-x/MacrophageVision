
$architecture = @(
    'efficientnet_b0', 
    'efficientnet_b2', 
    'inception_v4', 
    'pnasnet5large', 
    'resnext101_32x8d'
)

$model = @(
    '.\output\train\20220202-134654-efficientnet_b0-96\model_best.pth.tar', 
    '.\output\train\20220202-193206-efficientnet_b2-96\model_best.pth.tar', 
    '.\output\train\20220203-121645-inception_v4-96\model_best.pth.tar', 
    '.\output\train\20220203-103440-pnasnet5large-96\model_best.pth.tar', 
    '.\output\train\20220203-124112-resnext101_32x8d-96\model_best.pth.tar'
)

$output_dir = @(
    '.\output\test\efficientnet_b0', 
    '.\output\test\efficientnet_b2', 
    '.\output\test\inception_v4', 
    '.\output\test\pnasnet', 
    '.\output\test\resnext', 
    '.\output\test\resnext'
)

$output_name = 'M0', 'M1', 'M2'
$fold = 1..5

conda activate MacVis2

foreach($i in 0..4){
    md -Force $output_dir[$i]
    foreach($k in 1..5){
        python .\ML\timm_alt_models\train_alt.py `
        ".\data\processed\dataset_split\fold_${k}" `
        --model $architecture[$i] `
        --input-size 3 96 96 `
        --epochs 50 `
        --num-classes 3 -b 50 `
        --train-split "train" `
        --val-split "validation" `
        --weight-decay 1e-4 `
        --opt "adam" `
        --checkpoint-hist 1
    }
}