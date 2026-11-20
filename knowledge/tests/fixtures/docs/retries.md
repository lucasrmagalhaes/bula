# Retries

A failed job is retried automatically until it reaches the retry limit.

## The retryLimit option

Set `retryLimit` when creating the queue. The default is 3 attempts. A value of 0 disables retries.

```ts
const emails = createQueue("emails", { retryLimit: 5 });
```

## Backoff

Between attempts Kiteq waits using exponential backoff. The first delay is one second and it doubles after each failure, up to `maxBackoffMs`.

## Dead letter queue

When a job exhausts its retries it is moved to the dead letter queue named after the original queue with the `:dead` suffix.
