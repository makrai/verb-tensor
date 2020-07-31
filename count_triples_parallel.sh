for i in 0 `seq 9`
do 
  for k in 0 `seq 9`
  do 
    for j in 0 `seq 9`
    do 
      time nice python count_triples_pyconll.py $i$j$k
    done 
  done &
done
