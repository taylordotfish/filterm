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
[`Filter`]: https://docs.rs/filterm/0.4/filterm/trait.Filter.html
[`run`]: https://docs.rs/filterm/0.4/filterm/fn.run.html

Documentation
-------------

[Documentation is available on docs.rs.](https://docs.rs/filterm/0.4)

License
-------

Filterm is licensed under version 3 of the GNU General Public License,
or (at your option) any later version. See [LICENSE](LICENSE).

Contributing
------------

By contributing to Filterm, you agree that your contribution may be used
according to the terms of Filterm’s license.
