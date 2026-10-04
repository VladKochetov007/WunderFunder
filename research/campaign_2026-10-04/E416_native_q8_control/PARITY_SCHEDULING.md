Before starting v2 parity, the controller allocated a second independent
1 CPU / 2 GiB slot because E415 CPU work had finished. The v2 parity stage runs
on logical CPU14 while the existing v3 parity stage continues on CPU15.
The two processes write separate parity files and read identical frozen inputs.
Parity-pass elapsed times are not scientific timing trials. All actual paired
runtime trials retain CPU15 and wait for the controller's quiet-window signal.
