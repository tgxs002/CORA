exp=${1:-'test'}
gpu=${2:-'8'}
type=${3:-'local'} # choose slurm if you are running on a cluster with slurm scheduler

if [ "$type" == 'local' ]; then
  extra_args=${@:4:99}
else
  quotatype=${4:-'auto'} # for slurm
  partition=${5:-'1'} # for slurm
  extra_args=${@:6:99}
fi

name=${name/#configs/logs}
name=${name//.sh//$exp}
work_dir="${name}"
now=$(date +"%Y%m%d_%H%M%S")
mkdir  -p $work_dir

ncpu='16'

if [ "$quotatype" == 'reserved_normal' ]; then
  quotatype='reserved --phx-priority=normal'
fi

if [ "$type" == 'local' ]; then

  header="python -m torch.distributed.launch --nproc_per_node=${gpu} --use_env main.py "

else

  header="srun --async --partition=$partition -n${gpu} --mpi=pmi2 --gres=gpu:$gpu --ntasks-per-node=${gpu} --quotatype=$quotatype \
    --job-name=$exp --cpus-per-task=$ncpu --kill-on-bad-exit=1 -o $work_dir/phoenix-slurm-$now-%j.out python main.py "

fi
