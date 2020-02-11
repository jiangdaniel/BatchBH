CPUS=${1-"-1"}


./online.py --testers batchbh --task 1S-Mean3 --dir mean3-bbh --n-cpus $CPUS --n-trials 1 --rdiff --pi1s 0.1


./online.py --testers nonadaptive --task 1S-Mean3 --dir mean3-bbh --monotone --n-cpus $CPUS

./online.py --testers adaptive --task 1S-Mean3 --dir mean3-bsbh --monotone --n-cpus $CPUS

./online.py --testers bh --task 1S-Mean3 --dir mean3-bh --n-cpus $CPUS

./online.py --testers sbh --task 1S-Mean3 --dir mean3-sbh --n-cpus $CPUS


./online.py --testers nonadaptive --task 2S-GaussianMean --dir mean0-bbh --monotone --n-cpus $CPUS

./online.py --testers adaptive --task 2S-GaussianMean --dir mean0-bsbh --monotone --n-cpus $CPUS

./online.py --testers bh --task 2S-GaussianMean --dir mean0-bh --n-cpus $CPUS

./online.py --testers sbh --task 2S-GaussianMean --dir mean0-sbh --n-cpus $CPUS


./rename.sh


./online.py --testers nonadaptive --real pvals.pkl --dir real-bbh --alpha 0.1 --n-cpus $CPUS

./online.py --testers adaptive --real pvals.pkl --dir real-bsbh --alpha 0.1 --n-cpus $CPUS
