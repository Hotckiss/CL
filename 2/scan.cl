#define SWAP(a,b) {__local float *tmp = a; a = b; b = tmp; }

__kernel void scan_hillis_steele_block(int size, 
                                        __global float *input, 
                                        __global float *output, 
                                        __local float *a, 
                                        __local float *b) {
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint block_size = get_local_size(0);

    if (gid < size) {
        a[lid] = b[lid] = input[gid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
 
    for (uint size = 1; size < block_size; size <<= 1) {
        if (lid >= size) {
            b[lid] = a[lid] + a[lid - size];
        } else {
            b[lid] = a[lid];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        SWAP(a, b);
    }
    if (gid < size) {
        output[gid] = a[lid];
    }
}

__kernel void create_sum_blocks(int n, 
                                int m, 
                                __global float *input, 
                                __global float *output) {
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint block_size = get_local_size(0);

    uint cur_block_ind = gid / block_size + 1;

    if (gid >= n || cur_block_ind >= m) {
        return;
    }

    if (gid == cur_block_ind * block_size - 1)
        output[cur_block_ind] = input[gid];
}

__kernel void scan_propagation(int n,
                        __global float *block_sum, 
                        __global float *input, 
                        __global float *output) {
    uint gid = get_global_id(0);
    uint block_size = get_local_size(0);
    uint cur_block_ind = gid / block_size + 1;

    if (gid >= n) {
        return;
    }

    output[gid] = input[gid] + block_sum[cur_block_ind - 1];
}