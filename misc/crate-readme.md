Filterm
=======

Filterm lets you run a child process while piping all terminal data to and
from the child through a custom filter. This lets you modify things like
ANSI escape sequences that get sent from the child.

The main way of using Filterm is to define a custom filter by implementing
the [`Filter`] trait, and then call [`run`].

For an example of Filterm in use, see [Monoterm].

Platform support
----------------

Filterm has been tested on GNU/Linux. It may work on other Unix-like
operating systems, as it avoids using Linux- and GNU-specific functionality
and sticks to POSIX whenever possible.

[Monoterm]: https://github.com/taylordotfish/monoterm
[`Filter`]: https://docs.rs/filterm/0.5/filterm/trait.Filter.html
[`run`]: https://docs.rs/filterm/0.5/filterm/fn.run.html
