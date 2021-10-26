Filterm
=======

Filterm lets you run a child process while piping all terminal data to
and from the child through a custom filter. This lets you modify things
like ANSI escape sequences that get sent from the child.

The main way of using Filterm is to define a custom filter by implementing
the [`FilterHooks`] trait, and then call [`run`].

For an example of Filterm in use, see
[Monoterm](https://github.com/taylordotfish/monoterm).

Platform support
----------------

Filterm has been tested on GNU/Linux. It may work on other Unix-like
operating systems, as it avoids using Linux- and GNU-specific
functionality.

[`FilterHooks`]: https://docs.rs/filterm/latest/filterm/trait.FilterHooks.html
[`run`]: https://docs.rs/filterm/latest/filterm/fn.run.html
