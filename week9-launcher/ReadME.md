[Launcher](https://github.com/TACC/launcher) is a utility for performing many-task computing workflows on computing clusters. It is designed for running large collections of serial or multi-threaded applications within a single batch job. Launcher can be used as an alternative to Slurm job arrays and to pack many short-running jobs into one batch job.

With Launcher, you can run a set of defined jobs within a single batch job, even when you have more jobs than the number of requested CPUs. The number of available CPUs determines the upper limit on the number of jobs that can be run at the same time. In addition, you can easily use multiple compute nodes to increase the number of available CPUs.

### Using Launcher on CARC systems

Begin by logging in. You can find instructions for this in the [Getting Started with Discovery](/user-guides/hpc-systems/discovery/getting-started-discovery) or [Getting Started with Endeavour](/user-guides/hpc-systems/endeavour/getting-started-endeavour) user guides.

You can use Launcher by loading the corresponding software module:

```
module load launcher
```

Launcher is not a compiled program. Instead, it's a set of Bash and Python scripts, so you can use the Launcher module with any [software tree](/user-guides/hpc-systems/software/software-modules-lmod) available on CARC systems.

### Running Launcher in batch mode

In order to submit jobs to the Slurm job scheduler, you will need to use Launcher in batch mode. There are a few steps to follow:

1. Create a launcher job file that contains jobs to run (one job per line)
2. Create a Slurm job script that requests resources, configures Launcher, and runs the launcher job file
3. Submit the job script to the job scheduler using `sbatch`

A Slurm job script is a special type of Bash shell script that the Slurm job scheduler recognizes as a job. For a job running Launcher, a Slurm job script should look something like the following:

```
#!/bin/bash

#SBATCH --job-name=mnist_classify
#SBATCH --account=irahbari_1147
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-task=1
#SBATCH --time=1:00:00

module purge
module load launcher
module load gcc/11.3.0
module load hwloc/2.7.1

export LAUNCHER_DIR=$LAUNCHER_ROOT
export LAUNCHER_RMI=SLURM
export LAUNCHER_PLUGIN_DIR=$LAUNCHER_DIR/plugins
export LAUNCHER_SCHED=interleaved
export LAUNCHER_BIND=1
export LAUNCHER_WORKDIR=$PWD
export LAUNCHER_JOB_FILE=simulations.txt

eval "$(conda shell.bash hook)"
conda activate torch-env
$LAUNCHER_DIR/paramrun
```

Each line is described below:

| **Command or Slurm argument** | **Meaning** |
|---|---|
| `#!/bin/bash` | Use Bash to execute this script |
| `#SBATCH` | Syntax that allows Slurm to read your requests (ignored by Bash) |
| `--account=<project_id>` | Charge compute time to <project_id>; enter `myaccount` to view your available project IDs |
| `--partition=gpu` | Submit job to the gpu partition |
| `--nodes=2` | Use 2 compute nodes |
| `--ntasks-per-node=2` | Run 2 tasks per node |
| `--cpus-per-task=8` | Reserve 8 CPUs per task for your exclusive use |
| `--mem=32G` | Reserve 32GB memory on a node for your  use |
| `--time=1:00:00` | Reserve resources described for 1 hour |
| `module purge` | Clear [environment modules](/user-guides/hpc-systems/software/software-modules-lmod) |
| `module load launcher` | Load the `launcher` [environment module](/user-guides/hpc-systems/software/software-modules-lmod) |
| `module load gcc/11.3.0` | Load the `gcc` [environment module](/user-guides/hpc-systems/software/software-modules-lmod) |
| `module load hwloc/2.7.1` | Load the [`hwloc`](https://www.open-mpi.org/projects/hwloc/) [environment module](/user-guides/hpc-systems/software/software-modules-lmod) |
| `export LAUNCHER_DIR=$LAUNCHER_ROOT` | Set Launcher root directory |
| `export LAUNCHER_RMI=SLURM` | Use Slurm plugin |
| `export LAUNCHER_PLUGIN_DIR=$LAUNCHER_DIR/plugins` | Set plugin directory |
| `export LAUNCHER_SCHED=interleaved` | Use interleaved scheduling option |
| `export LAUNCHER_BIND=1` | Bind tasks to cores using `hwloc` |
| `export LAUNCHER_WORKDIR=$PWD` | Set working directory for job |
| `export LAUNCHER_JOB_FILE=simulations.txt` | Specify launcher job file to use |
| `eval "$(conda shell.bash hook)"` | Setup Conda |
| `conda activate torch-env` | Launch jobs |
| `$LAUNCHER_DIR/paramrun` | Launch jobs |

Adjust the resources requested based on your needs, keeping in mind that fewer resources requested leads to less queue time for your job.

In this example, the file `simulations.txt` may contain many lines like the following:

```
./sim 3 4 5 >& job-$LAUNCHER_JID.log
./sim 6 4 7 >& job-$LAUNCHER_JID.log
./sim 1 9 2 >& job-$LAUNCHER_JID.log
```

The same simulation program `sim` is being run but with varying parameter values for each run.

Launcher will schedule each line as a job on one of the tasks (CPUs) requested. In this serial example, there are 32 CPUs available across 2 compute nodes, so 32 jobs will run at one time until all jobs are completed.

In this example, the output of each job is also saved to a unique log file. For example, the `job-1.log` file would contain the output for the first line in the file.

Develop and edit Launcher job files and job scripts to run on CARC clusters:

* on your local computer and then transfer the files to one of your directories on CARC file systems.
* with the Files app available on our [OnDemand](/user-guides/carc-ondemand) service.
* with one of the available text editor modules (nano, micro, vim, or emacs).

Save the job script as `launcher.job`, for example, and then submit it to the job scheduler with Slurm's `sbatch` command:

```
[user@discovery1 ~]$ sbatch launcher.job
Submitted batch job 13589
```

Check the status of your job by entering `myqueue`. If there is no job status listed, then this means the job has completed.

The results of the job will be logged and, by default, saved to a file of the form `slurm-<jobid>.out` in the same directory where the job script is located. To view the contents of this file, enter `less slurm-<jobid>.out`, and then enter `q` to exit the viewer. In this example, each launcher job also has its own unique log file, and you can enter `less job-<$LAUNCHER_JID>.log` to view them.

For more information on job status and running jobs, see the [Running Jobs user guide](/user-guides/hpc-systems/using-our-hpc-systems/running-jobs).

### Additional resources

If you have questions about or need help with Launcher, please [submit a help ticket](/user-support/submit-ticket) and we will assist you.

* [TACC's Launcher](https://docs.tacc.utexas.edu/software/launcher/)
* [CARC fork of Launcher](https://github.com/uschpc/launcher/tree/uschpc)
* [CARC examples](https://github.com/uschpc/launcher/tree/uschpc/examples)
