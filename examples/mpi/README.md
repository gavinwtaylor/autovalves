# MPI Examples

- `hello.c` is an MPI 'Hello, World' (no communication)
- `pt2pt.c` demonstrates using MPI to send messages between two specific
  workers
- `status.c` demonstrates using MPI to receive messages from ANY worker,
  rather than a specific one.  Similar wildcards exist for tags, message size,
  etc.
- `collective.c` demonstrates several forms of collective communication
