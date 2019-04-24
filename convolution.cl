__kernel void convolution(int n, int hm, __global float *A, __global float *B, __global float *C) {
	int i = get_global_id(0);
	int j = get_global_id(1);

	if (i >= n || j >= n) {
		return;
	}

	float sum = 0;

	for (int k = -hm; k <= hm; k++) {
		for (int l = -hm; l <= hm; l++) {
			if (i + k < 0 || j + l < 0 || i + k >= n || j + l >= n) {
				continue;
			}

			sum += A[(i + k) * n + (j + l)] *
				B[(k + hm) * (2 * hm + 1) + (l + hm)];
		}
	}

	C[i * n + j] = sum;
}
