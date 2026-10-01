# Compute

One local RTX3060 laptop GPU, restored after reboot. At most twelve aggregate
CPU cores, shared hard36GiB RAM/no swap, observed VRAM <=5,000,000,000 bytes.
At most one GPU job; campaign total <=28,800 GPU seconds, including failed fits,
GPU prediction exports and synthetic GPU probes. Reserve >=3,240 seconds for
final fitting/verification. Research Python/torch versions differ from submission
Python3.11; deployment must be checked separately.

scripts/campaign_job.py provides durable launches, source hashes, resource
monitoring, exclusive GPU lease, accounting, timeouts and a persistent stop
barrier. scripts/limited places commands inside wnn-campaign.slice. Runtime slice
limits must be restored after reboot. No shell background runs.
