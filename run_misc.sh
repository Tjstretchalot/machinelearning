nice git pull | tee log_git_pull.txt
nice python3 -u -m clean 2>&1 | tee log_clean.txt
nice python3 -u -m gaussian_spheres.runners.ff.train_one 2>&1 | tee log.txt
