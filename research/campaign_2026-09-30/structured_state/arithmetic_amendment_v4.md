# Arithmetic amendment before corrected checks

Independent critic's frozen 20k half-life8192 impulse test rejected initial code:
two state-coordinate tolerance violations (atol2e-5 + rtol2e-4), max absolute
error1.3569e-4. FFT used unrounded log-radius powers; the real recurrence repeatedly
multiplied rounded coefficients. Previous actor checks used a looser absolute
5e-4 threshold and did not expose this. Preserve all original results and code.

Correct the implementation, retain the independent grader's tolerance. Define
the actual real transition coefficients as scalar `rho=exp(log_rho)` and rotation
`a=rho*cos(theta), b=rho*sin(theta)` in model dtype. The real recurrence applies
`(a*re-b*im,b*re+a*im)` directly. FFT kernels use `exp(t*log(rho))` for scalars
and `exp(t*log(complex(a,b)))` for rotations, plus incoming state multiplied by
the same rounded `rho`/`complex(a,b)` coefficient. This makes the analytic kernel
represent the floating point transition that a native port will serialize.

Model states/dense layers remain FP32 (FP64 toy oracle permitted), normalization
and injection unchanged, learned half-life/phase/projection/gate/head unchanged.
Autograd flows through coefficient operations and the log/power kernel;
independent real recurrence has the same parameter dependencies. Both init arms
receive the identical correction. Repeat actor checks, strict long impulse test,
and independent gradient/parity grade. This is a numerical identity fix, not an
evaluation-guided architecture change or changed scientific comparison.
