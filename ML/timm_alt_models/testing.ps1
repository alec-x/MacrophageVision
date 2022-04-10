
$architecture = @(
    'efficientnet_b0', 
    'efficientnet_b2', 
    'inception_v4', 
    'pnasnet5large', 
    'resnext101_32x8d'
)

$model = @{}
$model['efficientnet_b0'] = @(
    '.\output\train\20220217-212448-efficientnet_b0-96\model_best.pth.tar', 
    '.\output\train\20220217-213735-efficientnet_b0-96\model_best.pth.tar', 
    '.\output\train\20220217-215019-efficientnet_b0-96\model_best.pth.tar', 
    '.\output\train\20220217-220301-efficientnet_b0-96\model_best.pth.tar', 
    '.\output\train\20220217-221546-efficientnet_b0-96\model_best.pth.tar' 
)

$model['efficientnet_b2'] = @(  
    '.\output\train\20220217-222826-efficientnet_b2-96\model_best.pth.tar', 
    '.\output\train\20220217-224701-efficientnet_b2-96\model_best.pth.tar', 
    '.\output\train\20220217-230536-efficientnet_b2-96\model_best.pth.tar', 
    '.\output\train\20220217-232408-efficientnet_b2-96\model_best.pth.tar', 
    '.\output\train\20220217-234243-efficientnet_b2-96\model_best.pth.tar'    
)

$model['inception_v4'] = @(  
    '.\output\train\20220218-000119-inception_v4-96\model_best.pth.tar', 
    '.\output\train\20220218-002805-inception_v4-96\model_best.pth.tar', 
    '.\output\train\20220218-005456-inception_v4-96\model_best.pth.tar', 
    '.\output\train\20220218-012140-inception_v4-96\model_best.pth.tar', 
    '.\output\train\20220218-014829-inception_v4-96\model_best.pth.tar'    
)

$model['pnasnet5large'] = @(  
    '.\output\train\20220218-021523-pnasnet5large-96\model_best.pth.tar', 
    '.\output\train\20220218-035434-pnasnet5large-96\model_best.pth.tar', 
    '.\output\train\20220218-053350-pnasnet5large-96\model_best.pth.tar', 
    '.\output\train\20220218-071249-pnasnet5large-96\model_best.pth.tar', 
    '.\output\train\20220218-085147-pnasnet5large-96\model_best.pth.tar'    
)

$model['resnext101_32x8d'] = @(  
    '.\output\train\20220218-103054-resnext101_32x8d-96\model_best.pth.tar', 
    '.\output\train\20220218-113822-resnext101_32x8d-96\model_best.pth.tar', 
    '.\output\train\20220218-124557-resnext101_32x8d-96\model_best.pth.tar', 
    '.\output\train\20220218-135311-resnext101_32x8d-96\model_best.pth.tar', 
    '.\output\train\20220218-150013-resnext101_32x8d-96\model_best.pth.tar'    
)

$output_dir = @(
    '.\output\test\efficientnet_b0', 
    '.\output\test\efficientnet_b2', 
    '.\output\test\inception_v4', 
    '.\output\test\pnasnet', 
    '.\output\test\resnext'
)

$dataset = @(
    '.\data\processed\dataset_split\fold_1\',
    '.\data\processed\dataset_split\fold_2\',
    '.\data\processed\dataset_split\fold_3\',
    '.\data\processed\dataset_split\fold_4\',
    '.\data\processed\dataset_split\fold_5\'
)

conda activate MacVis2

foreach($i in 0..4){
    md -Force $output_dir[$i] > $null
    foreach($k in 0..4){
        $real_k = $k + 1
        python .\ML\timm_alt_models\inference_alt.py `
        $dataset[$k] `
        --model $architecture[$i] `
        --input-size 3 96 96 `
        --checkpoint $model[$architecture[$i]][$k] `
        --num-classes 3 -b 50 `
        --topk 1 `
        --output_dir $output_dir[$i] `
        --output_name "fold_${real_k}" `
        --log-freq 1000
    }  
}