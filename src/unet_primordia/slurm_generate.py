import os

hyperparam_set = [
    (True, 'crb', 0.0002, 0.0001),
    (True, 'crb', 0.0002, 0.0003),
    (True, 'brc', 0.0002, 0.0001),
    (True, 'brc', 0.0002, 0.0003)
]

out_channels = 6

def float_to_str(x):
    return str(x).replace('.', '_')


def hyperparams_to_name(hp):
    interpolate = {
        True: 'interpolate',
        False: 'no_interpolate'
    }

    layer_order = hp[1]

    lr = float_to_str(hp[2])
    wd = float_to_str(hp[3])
    return f'checkpoint_{interpolate[hp[0]]}_{layer_order}_lr{lr}_wd{wd}'


def generate_slurm_script(hyperparams,
                          prefix_dir='/g/kreshuk/wolny/workspace/models/3dunet',
                          train_script_path='/g/kreshuk/wolny/workspace/lsfm_utils/src/unet_primordia/train_unet3d.py',
                          config_dir='/g/kreshuk/wolny/workspace/models/3dunet/config'):
    slurm_template = \
    '''#!/bin/bash

#SBATCH -J {}
#SBATCH -A kreshuk                              
#SBATCH -N 1				            
#SBATCH -n 2				            
#SBATCH --mem 32G			            
#SBATCH -t 48:00:00                     
#SBATCH -o {}			        
#SBATCH -e {}			        
#SBATCH --mail-type=FAIL,BEGIN,END		    
#SBATCH --mail-user=adrian.wolny@embl.de
#SBATCH -p gpu				            
#SBATCH -C gpu=1080Ti			        
#SBATCH --gres=gpu:1	

module load cuDNN

export PYTHONPATH="/g/kreshuk/wolny/workspace/inferno:/g/kreshuk/wolny/workspace/neurofire:$PYTHONPATH"

{} {}        
    '''

    checkpoint = hyperparams_to_name(hyperparams)
    project_dir = os.path.join(prefix_dir, checkpoint)
    outfile = os.path.join(project_dir, 'outfile.out')
    errfile = os.path.join(project_dir, 'errfile.err')
    if hyperparams[0]:
        interpolate = '--interpolate'
    else:
        interpolate = ''

    layer_order = hyperparams[1]
    lr = hyperparams[2]
    wd = hyperparams[3]
    args = f'--config-dir {config_dir} --checkpoint-dir {project_dir} --validate-after-iters 100 --log-after-iters 100 --out-channels {out_channels} --layer-order {layer_order} --learning-rate {lr} --weight-decay {wd} {interpolate}'

    script_name = f'slurm_{checkpoint}.sh'
    return script_name, slurm_template.format(checkpoint, outfile, errfile, train_script_path, args)


def main():
    slurm_dir = './slurm_scripts'
    if not os.path.exists(slurm_dir):
        os.mkdir(slurm_dir)
    for hp in hyperparam_set:
        script_name, script_content = generate_slurm_script(hp)
        with open(os.path.join(slurm_dir, script_name), 'w') as f:
            f.write(script_content)


if __name__ == '__main__':
    main()