import subprocess
from matplotlib import pyplot as plt 
import numpy as np
import json 

batch_size = (1, 2, 4, 8, 16, 32, 64, 128, 256)
density = (0.7, 0.65, 0.6, 0.55, 0.5, 0.475, 0.45, 0.425, 0.4, 0.375, 0.35, 0.325, 0.3, 0.275, 0.25, 0.225, 0.2, 0.175, 0.15, 0.125, 0.1, 0.075, 0.05, 0.025, 0.02, 0.01, 0.00175, 0.0015, 0.001)
calculate_sparsity = lambda density: round((1 - density) * 100, 5)
sparsity = list(map(calculate_sparsity, density))
problem = {}
problem["qkv"] = (7680, 2560)
problem['attention_fc'] = (2560, 2560)
problem['linear1'] = (10240, 2560)
problem['linear2'] = (2560,10240)
profiled_metric = {}

peak_flop = 10496 #  Inst/cycle
peak_flops =  10496 * 2
frequency = 1.39e9    # cycle/nsecond
peak_perf = peak_flop * frequency

metrics = "l1tex__t_sectors_pipe_lsu_lookup_miss.sum,gpu__time_duration.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,smsp__sass_inst_executed_op_global_ld.sum,smsp__sass_inst_executed_op_ld.sum,smsp__sass_inst_executed_op_memory_32b.sum,smsp__inst_executed_pipe_fma.sum"
parsed_metrics = metrics.split(',')
metrics_idx = {}
print(metrics_idx)

for metric in parsed_metrics:
    profiled_metric[metric] = {}
    for layer in problem:
        profiled_metric[metric][layer] = {}
        for batch in batch_size:
            profiled_metric[metric][layer][batch] = {}
            for sparse in density:
                profiled_metric[metric][layer][batch][sparse] = 0


for layer in problem:
    m, k = problem[layer]
    print("Profile: ",layer, m, k)
    for batch in batch_size:
        print("Batch_size:", m, batch, k)
        for sparse in density:
            print("Sparse: ", round(1 - sparse,4))
            result = subprocess.run(f"ncu --metric {metrics} ./cusparse_sgemm {m} {batch} {k} {sparse} 0", stdout=subprocess.PIPE, shell=True)  
            csr_idx = result.stdout.decode().find("CsrMMPolicy")
            for metric in parsed_metrics:
                metrics_idx[metric] = result.stdout.decode()[csr_idx:].find(metric)
                print(result.stdout.decode()[csr_idx+metrics_idx[metric]:csr_idx+metrics_idx[metric]+120].split())

                
                if result.stdout.decode()[csr_idx+metrics_idx[metric]:csr_idx+metrics_idx[metric]+120].split()[1] == 'msecond':
                    profiled_metric[metric][layer][batch][sparse] = float(result.stdout.decode()[csr_idx+metrics_idx[metric]:csr_idx+metrics_idx[metric]+120].split()[-1].replace(',','')) * 1000.0
                else:
                    profiled_metric[metric][layer][batch][sparse] = float(result.stdout.decode()[csr_idx+metrics_idx[metric]:csr_idx+metrics_idx[metric]+120].split()[-1].replace(',',''))


with open("./cusparse_profiled_metric.json", 'w') as file:
    json.dump(profiled_metric, file)