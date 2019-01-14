for i in `seq 1 10`;
do
  echo worker $i
  python 01_generate_data.py doomrnn --total_episodes 100 --start_batch $i --batch_size 100 --time_steps 1000 &
  sleep 1.0
done
