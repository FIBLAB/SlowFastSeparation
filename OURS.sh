# #--------------------------------2S2F--------------------------------
# # phase=TimeSelection
# phase=LearnDynamics
# tau_s=1.0
# slow_dim=2
# koopman_dim=$slow_dim

# model=ours
# system=2S2F
# trace_num=200
# total_t=5.1
# dt=0.01
# id_epoch=100
# learn_epoch=150
# # seed_num=10
# seed_num=3
# tau_1=0.1
# tau_N=3.0
# device=cuda
# gpu_id=0
# cpu_num=1
# data_dir=Data/$system/data/
# id_log_dir=logs/$system/TimeSelection/
# learn_log_dir=logs/$system/LearnDynamics/slow_${slow_dim}_koopman_${koopman_dim}/
# result_dir=Results/$system/slow_${slow_dim}_koopman_${koopman_dim}/


#--------------------------------1S2F--------------------------------
# phase=TimeSelection
phase=LearnDynamics
tau_s=3.0
slow_dim=1
koopman_dim=$slow_dim

model=ours
system=1S2F
trace_num=100
total_t=15.1
dt=0.01
id_epoch=30
learn_epoch=50
# seed_num=10
seed_num=3
tau_1=0.3
tau_N=7.0
device=cpu
gpu_id=1
cpu_num=1
data_dir=Data/$system/data/
id_log_dir=logs/$system/TimeSelection/
learn_log_dir=logs/$system/LearnDynamics/slow_${slow_dim}_koopman_${koopman_dim}/
result_dir=Results/$system/slow_${slow_dim}_koopman_${koopman_dim}/


CUDA_VISIBLE_DEVICES=$gpu_id python run.py \
--model $model \
--system $system \
--trace_num $trace_num \
--total_t $total_t \
--dt $dt \
--id_epoch $id_epoch \
--learn_epoch $learn_epoch \
--seed_num $seed_num \
--tau_1 $tau_1 \
--tau_N $tau_N \
--tau_s $tau_s \
--slow_dim $slow_dim \
--koopman_dim $koopman_dim \
--device $device \
--phase $phase \
--cpu_num $cpu_num \
--data_dir $data_dir \
--id_log_dir $id_log_dir \
--learn_log_dir $learn_log_dir \
--result_dir $result_dir \
--parallel