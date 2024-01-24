from atpbind_main import single_run, write_result
import numpy as np

def lr_range_test(tenfold_iter=10, valid_fold_num=0, gn_dim_count=4):
    start_lr = 1e-8
    gamma = 10**(1/tenfold_iter)
    model = 'lm-gearnet'
    result = single_run(
        valid_fold_num=valid_fold_num,
        model=model,
        model_kwargs={
            'lm_type': 'esm-t33',
            'gearnet_hidden_dim_size': 512,
            'gearnet_hidden_dim_count': gn_dim_count,
            'lm_freeze_layer_count': 30,
        },
        pipeline_kwargs={
            'optimizer_kwargs': {
                'lr': start_lr,
            },
            'scheduler': 'exponential',
            'scheduler_kwargs': {
                'gamma': gamma,
            }
        },
        patience=10**9,
        max_epoch=int(tenfold_iter * 7.5),
        gpu=1,
    )
    
    for record in result['full_record']:
        result['record'] = record
        write_result(
            model_key=f'{model}-{gn_dim_count}-lrrange',
            valid_fold=valid_fold_num,
            result=result,
            write_inference=False,
            result_file='result_cv/result_cv_sides.csv',
        )
        

if __name__ == '__main__':
    lr_range_test(tenfold_iter=20, valid_fold_num=0, gn_dim_count=4)