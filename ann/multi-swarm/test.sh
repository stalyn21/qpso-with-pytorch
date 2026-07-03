#!/usr/bin/env bash                                                                                                                                                                                                                      
set -e      
                                                                                                                                                                                                                                         
for seed in 123 7 99 256; do
  for tech in t1 t2 t2-jacobi t3 t4; do                                                                                                                                                                                                  
    for ds in iris wine circle breast; do                                                                                                                                                                                                
      extra=""
      [ "$tech" = "t2-jacobi" ] && extra="--n-rounds 5"                                                                                                                                                                                  
      python runner.py --technique $tech --dataset $ds --cv-folds 5 --n-sessions 1 --n-particles 100 --max-iter 500 --seed $seed $extra                                                                                                                                                                             
    done                                                                                                                                                                                                                                 
  done                                                                                                                                                                                                                                   
done 

# python runner.py --technique t1 --dataset iris --cv-folds 5 --n-sessions 1 --n-particles 100 --max-iter 500 --seed 42 

# python runner.py --technique t2 --dataset iris --cv-folds 5 --n-sessions 1 --n-particles 100 --max-iter 500 --seed 42 

# python runner.py --technique t2-jacobi --dataset iris --cv-folds 5 --n-sessions 1 --n-particles 100 --max-iter 500 --seed 42 --n-rounds 5

# python compare.py