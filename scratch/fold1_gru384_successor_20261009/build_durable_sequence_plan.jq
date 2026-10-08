$q[0] as $queue
| {
    schema: "wnn.fold1_gru384_durable_sequence.r1",
    runtime: $runtime,
    disk_path: $root,
    min_free_bytes: 2000000000,
    queue_path: $queue_path,
    queue_sha256: "fd46f7533b209297241b0b0f924b8cc3a795a609232f18c3cae3bee3bbb4d272",
    stage: "wait for existing seed3 attempt; gate its terminal artifacts and launch seed4 once",
    steps: [
      {
        attempt: $queue.steps[0].attempt_id,
        expected_command: $queue.steps[0].command,
        must_not_exist: $queue.steps[0].must_not_exist_before_launch,
        output_bytes: 0
      },
      {
        attempt: $queue.steps[1].attempt_id,
        expected_command: $queue.steps[1].command,
        launch_command: $launch,
        must_not_exist: $queue.steps[1].must_not_exist_before_launch,
        output_bytes: 0
      }
    ]
  }
