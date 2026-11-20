---
title: Getting started with Kiteq
---

# Getting started

Kiteq is a fictional job queue library used only by this test suite. It does not exist on any package registry.

## Installation

Install the package and its peer dependency:

```bash
npm install kiteq kiteq-redis
```

## Creating a queue

Call `createQueue(name, options)` to create a queue. The name must be unique per connection.

```ts
import { createQueue } from "kiteq";

const emails = createQueue("emails", { concurrency: 4 });
```

### The concurrency option

`concurrency` controls how many jobs run at the same time in one worker process. The default is 1.
