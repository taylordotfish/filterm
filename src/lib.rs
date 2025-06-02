/*
 * Copyright (C) 2021-2025 taylor.fish <contact@taylor.fish>
 *
 * This file is part of Filterm.
 *
 * Filterm is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Filterm is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Filterm. If not, see <https://www.gnu.org/licenses/>.
 */

//! Filterm lets you run a child process while piping all terminal data to and
//! from the child through a custom filter. This lets you modify things like
//! ANSI escape sequences that get sent from the child.
//!
//! The main way of using Filterm is to define a custom filter by implementing
//! the [`Filter`] trait, and then call [`run`].
//!
//! For an example of Filterm in use, see [Monoterm].
//!
//! Platform support
//! ----------------
//!
//! Filterm has been tested on GNU/Linux. It may work on other Unix-like
//! operating systems, as it avoids using Linux- and GNU-specific functionality
//! and sticks to POSIX whenever possible.
//!
//! [Monoterm]: https://github.com/taylordotfish/monoterm

use std::cell::Cell;
use std::convert::{Infallible, TryFrom};
use std::env;
use std::ffi::{CStr, CString, OsString};
use std::io;
use std::mem::MaybeUninit;
use std::ops::ControlFlow;
use std::os::fd::{AsFd, AsRawFd, BorrowedFd, OwnedFd, RawFd};
use std::os::unix::ffi::OsStringExt;
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};

use atomic_int::AtomicCInt;
use nix::errno::Errno;
use nix::fcntl::{FcntlArg, FdFlag, OFlag, fcntl};
use nix::libc::{self, c_char, c_int};
use nix::poll::{PollFd, PollFlags, PollTimeout, poll};
use nix::pty::openpty;
use nix::sys::signal::{SaFlags, SigAction, SigHandler, sigaction};
use nix::sys::signal::{SigSet, Signal, kill, raise};
use nix::sys::termios::{SetArg, Termios, cfmakeraw, tcgetattr, tcsetattr};
use nix::sys::wait::{WaitPidFlag, WaitStatus, waitpid};
use nix::unistd::{self, ForkResult, Pid, fork, setsid};
use nix::unistd::{isatty, pipe, read, write};

#[macro_use]
pub mod error;
mod utils;

use error::CallName::Ioctl;
use error::Client::*;
pub use error::Error;
use error::ErrorKind::{self, *};
use error::{ChildCommFailed, ChildCommFailedKind, UnexpectedChildStatus};

fn tiocgwinsz(fd: RawFd) -> nix::Result<libc::winsize> {
    nix::ioctl_read_bad!(ioctl, libc::TIOCGWINSZ, libc::winsize);
    let mut size = MaybeUninit::uninit();
    unsafe { ioctl(fd, size.as_mut_ptr()) }?;
    // SAFETY: Initialized by `ioctl`.
    Ok(unsafe { size.assume_init() })
}

fn tiocswinsz(fd: RawFd, size: &libc::winsize) -> nix::Result<()> {
    nix::ioctl_write_ptr_bad!(ioctl, libc::TIOCSWINSZ, libc::winsize);
    unsafe { ioctl(fd, size as *const _) }.map(|_| ())
}

fn tiocsctty(fd: RawFd) -> nix::Result<()> {
    nix::ioctl_write_int_bad!(ioctl, libc::TIOCSCTTY);
    unsafe { ioctl(fd, 0) }.map(|_| ())
}

#[allow(clippy::declare_interior_mutable_const)]
const SIGACTION_CELL: Cell<Option<SigAction>> = Cell::new(None);
thread_local! {
    static ORIG_TERM_ATTRS: Cell<Option<Termios>> = const { Cell::new(None) };
    static CHILD_PID: Cell<Option<Pid>> = const { Cell::new(None) };
    static ORIG_SIGNAL_ACTIONS: [Cell<Option<SigAction>>; 5] =
        const { [SIGACTION_CELL; 5] };
}

const SIGNALS: [Signal; 5] = [
    Signal::SIGWINCH,
    Signal::SIGCHLD,
    Signal::SIGHUP,
    Signal::SIGINT,
    Signal::SIGTERM,
];

const SIGWINCH_INDEX: usize = 0;
const SIGCHLD_INDEX: usize = 1;
const TERMINATE_INDICES: [usize; 3] = [2, 3, 4];

#[allow(clippy::declare_interior_mutable_const)]
const ATOMIC_FALSE: AtomicBool = AtomicBool::new(false);
static SIGNAL_RECEIVED: [AtomicBool; 5] = [ATOMIC_FALSE; 5];
static SIGNAL_FD: AtomicCInt = AtomicCInt::new(0);

extern "C" fn handle_signal(signal: c_int) {
    let raw = SIGNAL_FD.load(Ordering::Relaxed);
    if raw == 0 {
        return;
    }
    let fd = unsafe { BorrowedFd::borrow_raw(raw) };
    if let Some(i) = SIGNALS.iter().position(|s| signal == *s as _) {
        SIGNAL_RECEIVED[i].store(true, Ordering::Relaxed);
        ignore_error_sigsafe!(write(fd, &[signal as u8]));
        return;
    }

    let stderr = io::stderr();
    let stderr = stderr.as_fd();
    let _ = write(stderr, b"unexpected signal: ");
    let (buf, i) = utils::format_c_uint(signal as _);
    let _ = write(stderr, &buf[i..]);
    let _ = write(stderr, b"\n");
}

fn forward_signal(index: usize) {
    let old_action = ORIG_SIGNAL_ACTIONS.with(|actions| actions[index].get());
    let old_action = if let Some(action) = old_action {
        action
    } else {
        return;
    };
    let result = unsafe { sigaction(SIGNALS[index], &old_action) };
    ignore_error!(result);
    if let Ok(ref action) = result {
        ignore_error!(raise(SIGNALS[index]));
        unsafe {
            sigaction(SIGNALS[index], action).expect("sigaction() failed");
        }
    }
}

macro_rules! define_child_error_kind {
    ($($variant:ident),* $(,)?) => {
        define_child_error_kind!(impl 0, (), $($variant,)*);
    };

    (impl $len:expr, ($($variant:ident,)*),) => {
        #[repr(u32)]
        #[derive(Clone, Copy, Eq, PartialEq)]
        enum ChildErrorKind {
            $($variant,)*
        }

        impl ChildErrorKind {
            pub const VARIANTS: [Self; $len] = [$(Self::$variant),*];
        }
    };

    (impl $len:expr, ($($tt:tt)*), $first:ident, $($rest:tt)*) => {
        define_child_error_kind!(impl $len + 1, ($($tt)* $first,), $($rest)*);
    };
}

define_child_error_kind! {
    Setsid,
    Tiocsctty,
    Tcsetattr,
    Execv,
}

struct ChildError(ChildErrorKind, Errno);

impl From<ChildError> for Error {
    fn from(e: ChildError) -> Self {
        use ChildErrorKind as Kind;
        Self {
            kind: match e.0 {
                Kind::Setsid => ChildSetupFailed,
                Kind::Tiocsctty => ChildSetupFailed,
                Kind::Tcsetattr => SetAttrFailed {
                    target: Child,
                    caller: Child,
                },
                Kind::Execv => ChildExecFailed,
            },
            call: match e.0 {
                Kind::Setsid => Some("setsid".into()),
                Kind::Tiocsctty => Some(Ioctl("TIOCSCTTY")),
                Kind::Tcsetattr => Some("tcsetattr".into()),
                Kind::Execv => Some("execv".into()),
            },
            io_error: Some(e.1.into()),
        }
    }
}

impl From<ChildError> for u64 {
    fn from(e: ChildError) -> Self {
        ((e.0 as u32 as u64) << 32) | (e.1 as i32 as u32 as u64)
    }
}

impl TryFrom<u64> for ChildError {
    type Error = u64;

    fn try_from(n: u64) -> Result<Self, Self::Error> {
        let errno = Errno::from_raw(n as u32 as i32);
        match ChildErrorKind::VARIANTS.get((n >> 32) as usize) {
            Some(&kind) => Ok(Self(kind, errno)),
            None => Err(n),
        }
    }
}

fn child_exec(
    paths: &[impl AsRef<CStr>],
    args: &[impl AsRef<CStr>],
    buf: &mut [*const c_char],
    tty_fd: OwnedFd,
    attrs: &Termios,
) -> Result<Infallible, ChildError> {
    use ChildError as Error;
    use ChildErrorKind as Kind;

    if paths.is_empty() {
        let _ = write(io::stderr(), b"child_exec: empty `paths`\n");
        unsafe {
            libc::abort();
        }
    }

    if let Some(ptr) = buf.get_mut(args.len()) {
        *ptr = ptr::null();
    } else {
        let _ = write(io::stderr(), b"child_exec: bad `buf` length\n");
        unsafe {
            libc::abort();
        }
    }

    for (arg, ptr) in args.iter().zip(&mut *buf) {
        *ptr = arg.as_ref().as_ptr();
    }

    let _ = unistd::dup2_stdin(&tty_fd);
    let _ = unistd::dup2_stdout(&tty_fd);
    let _ = unistd::dup2_stderr(&tty_fd);
    drop(tty_fd);

    setsid().map_err(|e| Error(Kind::Setsid, e))?;
    tiocsctty(0).map_err(|e| Error(Kind::Tiocsctty, e))?;
    tcsetattr(io::stdin(), SetArg::TCSANOW, attrs)
        .map_err(|e| Error(Kind::Tcsetattr, e))?;

    for path in paths {
        // Note: we're using the raw `libc` function because the Nix wrapper
        // isn't async-signal-safe.
        unsafe {
            libc::execv(path.as_ref().as_ptr(), buf.as_ptr());
        }
        use Errno::*;
        match Errno::last() {
            EACCES | EINVAL | ELOOP | ENAMETOOLONG | ENOENT | ENOTDIR
            | ENOEXEC => {}
            _ => break,
        }
    }
    Err(Error(Kind::Execv, Errno::last()))
}

fn read_child_error(fd: BorrowedFd<'_>) -> Result<Option<ChildError>, Error> {
    let mut buf = [0; 8];
    let mut nread = 0;
    while nread < buf.len() {
        nread += match read(fd, &mut buf[nread..])
            .map_err(ErrorKind::from(ChildCommFailed::new()).with("read"))?
        {
            0 => return Ok(None),
            n => n,
        }
    }
    let n = u64::from_be_bytes(buf);
    Ok(Some(
        ChildError::try_from(n)
            .expect("received invalid error from child process"),
    ))
}

struct Buffers {
    pub input: Box<[u8]>,
    pub output: Vec<u8>,
}

fn handle_stdin_ready<F: Filter>(
    pty_fd: BorrowedFd<'_>,
    filter: &mut F,
    bufs: &mut Buffers,
) -> Result<ControlFlow<()>, Error> {
    let nread = match read(io::stdin(), &mut bufs.input) {
        Ok(0) => {
            return Err(Error {
                kind: EmptyParentRead,
                call: Some("read".into()),
                io_error: None,
            });
        }
        r => r.map_err(ParentReadFailed.with("read"))?,
    };

    let inbuf = &bufs.input;
    let mut write_err = false;
    chunked(
        &mut bufs.output,
        |c| filter.on_parent_data(&inbuf[..nread], |data| c.add(data)),
        |chunk| {
            if !write_err {
                write_err = write(pty_fd, chunk).is_err();
            }
        },
    );

    Ok(if write_err {
        ControlFlow::Break(())
    } else {
        ControlFlow::Continue(())
    })
}

fn handle_pty_ready<F: Filter>(
    pty_fd: BorrowedFd<'_>,
    filter: &mut F,
    bufs: &mut Buffers,
) -> Result<ControlFlow<()>, Error> {
    let nread = match read(pty_fd, &mut bufs.input) {
        Ok(0) | Err(_) => return Ok(ControlFlow::Break(())),
        Ok(n) => n,
    };

    let inbuf = &bufs.input;
    let mut write_err = None;
    chunked(
        &mut bufs.output,
        |c| filter.on_child_data(&inbuf[..nread], |data| c.add(data)),
        |chunk| {
            if write_err.is_none() {
                write_err = write(io::stdin(), chunk).err();
            }
        },
    );
    write_err
        .map(ParentWriteFailed.with("write"))
        .map_or(Ok(ControlFlow::Continue(())), Err)
}

fn update_child_winsize(pty_fd: RawFd, pid: Pid) -> Result<(), Error> {
    if !SIGNAL_RECEIVED[SIGWINCH_INDEX].swap(false, Ordering::Relaxed) {
        return Ok(());
    }

    let size = tiocgwinsz(0).map_err(GetSizeFailed.with("tiocgwinsz"))?;
    tiocswinsz(pty_fd, &size)
        .map_err(SetSizeFailed.with(Ioctl("TIOCSWINSZ")))?;

    ignore_error!(kill(pid, Signal::SIGWINCH));
    forward_signal(SIGWINCH_INDEX);
    Ok(())
}

fn handle_pending_terminate() -> Result<(), Error> {
    for i in TERMINATE_INDICES {
        if SIGNAL_RECEIVED[i].load(Ordering::Relaxed) {
            return Err(Error::from_kind(ReceivedSignal(SIGNALS[i] as _)));
        }
    }
    Ok(())
}

fn try_child_wait(pid: Pid) -> Result<Option<Exit>, Error> {
    if !SIGNAL_RECEIVED[SIGCHLD_INDEX].swap(false, Ordering::Relaxed) {
        return Ok(None);
    }

    let result = match waitpid(pid, Some(WaitPidFlag::WNOHANG))
        .map_err(GetChildStatusFailed.with("waitpid"))?
    {
        WaitStatus::Exited(_, code) => Ok(Some(Exit::Normal(code))),
        WaitStatus::Signaled(_, sig, _) => Ok(Some(Exit::Signal(sig as _))),
        WaitStatus::StillAlive => Ok(None),
        status => Err(Error {
            kind: UnexpectedChildStatus::new(status).into(),
            call: Some("waitpid".into()),
            io_error: None,
        }),
    };

    forward_signal(SIGCHLD_INDEX);
    result
}

fn set_nonblock(fd: BorrowedFd<'_>) -> Result<(), Errno> {
    let flags = OFlag::from_bits_truncate(fcntl(fd, FcntlArg::F_GETFL)?);
    fcntl(fd, FcntlArg::F_SETFL(flags | OFlag::O_NONBLOCK)).map(|_| ())
}

fn read_all_nonblocking(fd: BorrowedFd<'_>) -> Result<(), Errno> {
    let mut buf = [0; 64];
    loop {
        match read(fd, &mut buf) {
            Ok(0) | Err(Errno::EAGAIN) => return Ok(()),
            Ok(_) => {}
            Err(e) => return Err(e),
        }
    }
}

/// A trait for filtering data to and from a child terminal.
///
/// An object implementing this trait should be passed to [`run`].
pub trait Filter {
    /// Called when data from the child terminal is received.
    ///
    /// `parent_write` should be called (any number of times) to forward this
    /// data, or send different data, to the parent terminal. The default
    /// implementation of this method forwards all data unchanged.
    fn on_child_data<F>(&mut self, data: &[u8], mut parent_write: F)
    where
        F: FnMut(&[u8]),
    {
        parent_write(data);
    }

    /// Called when data from the parent terminal is received.
    ///
    /// `child_write` should be called (any number of times) to forward this
    /// data, or send different data, to the child terminal. The default
    /// implementation of this method forwards all data unchanged.
    fn on_parent_data<F>(&mut self, data: &[u8], mut child_write: F)
    where
        F: FnMut(&[u8]),
    {
        child_write(data);
    }
}

/// For use with [`chunked`]; see its documentation for more information.
struct Chunked<'a, T, OutFn> {
    buf: &'a mut Vec<T>,
    chunk_out: OutFn,
}

impl<T, OutFn> Chunked<'_, T, OutFn>
where
    T: Copy,
    OutFn: FnMut(&[T]),
{
    /// For use with [`chunked`]; see its documentation for more information.
    pub fn add(&mut self, mut data: &[T]) {
        assert!(self.buf.capacity() > 0);
        loop {
            let spare = self.buf.capacity() - self.buf.len();
            if data.len() <= spare {
                self.buf.extend(data.iter().copied());
                if self.buf.capacity() == self.buf.len() {
                    self.flush();
                }
                return;
            }
            self.buf.extend(data.iter().copied().take(spare));
            self.flush();
            data = &data[spare..];
        }
    }

    fn flush(&mut self) {
        (self.chunk_out)(self.buf);
        self.buf.clear();
    }
}

/// Given a function, `data_in`, that repeatedly calls [`Chunked::add`] with
/// pieces of data, this function groups the data into chunks using the
/// provided buffer `buf` and calls `chunk_out` with each chunk.
fn chunked<T, In, Out>(buf: &mut Vec<T>, data_in: In, chunk_out: Out)
where
    T: Copy,
    In: FnOnce(&mut Chunked<'_, T, Out>),
    Out: FnMut(&[T]),
{
    let mut chunked = Chunked {
        buf,
        chunk_out,
    };
    data_in(&mut chunked);
    if !chunked.buf.is_empty() {
        chunked.flush();
    }
}

/// Returned by [`run`] when the child process exits.
#[non_exhaustive]
pub enum Exit {
    /// The child process exited normally with the given exit code.
    Normal(i32),
    /// The child process exited with a signal.
    Signal(i32),
}

fn run_impl(
    paths: &[impl AsRef<CStr>],
    args: &[impl AsRef<CStr>],
    filter: &mut impl Filter,
) -> Result<Exit, Error> {
    if !isatty(io::stdin()).unwrap_or(false) {
        return Err(Error::from_kind(NotATty));
    }

    let term_attrs =
        tcgetattr(io::stdin()).map_err(GetAttrFailed.with("tcgetattr"))?;
    let winsize =
        tiocgwinsz(0).map_err(GetSizeFailed.with(Ioctl("TIOCGWINSZ")))?;

    // We have to use `openpty` instead of `posix_openpt` because `ptsname`
    // isn't thread-safe (and is thus unsafe to call).
    let pty = openpty(None, None).map_err(CreatePtyFailed.with("openpty"))?;
    let [pty_parent, pty_child] = [pty.master, pty.slave];

    let (sig_read, sig_write) =
        pipe().map_err(SignalSetupFailed.with("pipe"))?;
    set_nonblock(sig_read.as_fd())
        .and_then(|_| set_nonblock(sig_write.as_fd()))
        .map_err(SignalSetupFailed.with("fnctl"))?;
    SIGNAL_FD.store(sig_write.as_raw_fd(), Ordering::Relaxed);

    for (i, signal) in SIGNALS.into_iter().enumerate() {
        let orig = unsafe {
            sigaction(
                signal,
                &SigAction::new(
                    SigHandler::Handler(handle_signal),
                    SaFlags::SA_RESTART,
                    SigSet::empty(),
                ),
            )
        }
        .map_err(SignalSetupFailed.with("sigaction"))?;
        ORIG_SIGNAL_ACTIONS.with(|actions| {
            actions[i].set(Some(orig));
        });
    }

    let mut new_attrs = term_attrs.clone();
    cfmakeraw(&mut new_attrs);
    ORIG_TERM_ATTRS.with(|attrs| attrs.set(Some(term_attrs.clone())));
    tcsetattr(io::stdin(), SetArg::TCSANOW, &new_attrs).map_err(
        SetAttrFailed {
            target: Parent,
            caller: Parent,
        }
        .with("tcsetattr"),
    )?;
    tiocswinsz(pty_parent.as_raw_fd(), &winsize)
        .map_err(SetSizeFailed.with(Ioctl("TIOCSWINSZ")))?;

    let (read_fd, write_fd) = pipe()
        .map_err(ErrorKind::from(ChildCommFailed::new()).with("pipe"))?;
    let mut buf = Vec::new();
    buf.resize(args.len() + 1, ptr::null());

    let fork = unsafe { fork() }.map_err(CreateChildFailed.with("fork"))?;
    let child_pid = match fork {
        ForkResult::Child => {
            SIGNAL_FD.store(0, Ordering::Relaxed);
            drop(sig_read);
            drop(sig_write);
            drop(pty_parent);
            drop(read_fd);
            ignore_error_sigsafe!(fcntl(
                &write_fd,
                FcntlArg::F_SETFD(FdFlag::FD_CLOEXEC)
            ));
            let result: Result<Infallible, _> =
                child_exec(paths, args, &mut buf, pty_child, &term_attrs);
            let err = u64::from(result.unwrap_err()).to_be_bytes();
            ignore_error_sigsafe!(write(&write_fd, &err));
            unsafe {
                libc::_exit(libc::EXIT_FAILURE);
            }
        }
        ForkResult::Parent {
            child,
        } => child,
    };

    CHILD_PID.with(|pid| pid.set(Some(child_pid)));
    drop(pty_child);
    drop(write_fd);
    if let Some(e) = read_child_error(read_fd.as_fd())? {
        return Err(Error::from(e));
    }
    drop(read_fd);

    const BUFFER_SIZE: usize = 1024;
    let mut bufs = Buffers {
        input: vec![0; BUFFER_SIZE].into_boxed_slice(),
        output: Vec::with_capacity(BUFFER_SIZE),
    };

    let stdin = io::stdin();
    let mut poll_fds = [sig_read.as_fd(), stdin.as_fd(), pty_parent.as_fd()]
        .map(|fd| PollFd::new(fd, PollFlags::POLLIN));

    loop {
        const EMPTY: PollFlags = PollFlags::empty();

        match poll(&mut poll_fds, PollTimeout::NONE) {
            Err(Errno::EINTR) => continue,
            r => {
                r.map_err(ErrorKind::from(ChildCommFailed::new()).with("poll"))
            }
        }?;

        match poll_fds[0].revents() {
            Some(EMPTY) => {}
            Some(PollFlags::POLLIN) => {
                read_all_nonblocking(sig_read.as_fd()).map_err(
                    ChildCommFailed(
                        ChildCommFailedKind::SignalReadError.into(),
                    )
                    .with("read"),
                )?;
                handle_pending_terminate()?;
                if let Some(exit) = try_child_wait(child_pid)? {
                    return Ok(exit);
                }
                update_child_winsize(pty_parent.as_raw_fd(), child_pid)?;
            }
            Some(flags) => {
                return Err(Error::from_kind(ChildCommFailed(
                    ChildCommFailedKind::BadPollFlags(flags).into(),
                )));
            }
            None => {}
        }

        if poll_fds[1].revents() != Some(PollFlags::empty()) {
            if let ControlFlow::Break(_) =
                handle_stdin_ready(pty_parent.as_fd(), filter, &mut bufs)?
            {
                break;
            }
        }

        if poll_fds[2].revents() != Some(PollFlags::empty()) {
            if let ControlFlow::Break(_) =
                handle_pty_ready(pty_parent.as_fd(), filter, &mut bufs)?
            {
                break;
            }
        }
    }

    match waitpid(child_pid, None)
        .map_err(GetChildStatusFailed.with("waitpid"))?
    {
        WaitStatus::Exited(_, code) => Ok(Exit::Normal(code)),
        WaitStatus::Signaled(_, sig, _) => Ok(Exit::Signal(sig as _)),
        status => Err(Error {
            kind: UnexpectedChildStatus::new(status).into(),
            call: Some("waitpid".into()),
            io_error: None,
        }),
    }
}

/// Runs the command specified by `args` with the given filter.
///
/// This function behaves as if it passes `args` to `execvp()`, but it
/// internally uses `execv()` as the former is not async-signal-safe.
///
/// # Panics
///
/// This function will panic if called multiple times concurrently, either from
/// multiple threads or by a function in `filter`.
pub fn run<Args, Arg, F>(args: Args, filter: &mut F) -> Result<Exit, Error>
where
    Args: IntoIterator<Item = Arg>,
    Arg: Into<OsString>,
    F: Filter,
{
    ORIG_TERM_ATTRS.with(|attrs| attrs.set(None));
    CHILD_PID.with(|pid| pid.set(None));
    ORIG_SIGNAL_ACTIONS.with(|actions| {
        for action in actions {
            action.set(None);
        }
    });

    let args: Vec<_> = args
        .into_iter()
        .map(|a| CString::new(a.into().into_vec()).unwrap())
        .collect();
    let first =
        args.first().ok_or_else(|| Error::from_kind(ChildExecFailed))?;

    let paths = if first.as_bytes().contains(&b'/') {
        Vec::new()
    } else {
        // Strictly speaking, we should use `confstr(_CS_PATH, ...)` in the
        // case where `PATH` is unset, but that function is not exported by
        // `libc`.
        env::split_paths(
            &env::var_os("PATH").unwrap_or_else(|| "/bin:/usr/bin".into()),
        )
        .map(|path| {
            let mut path = path.into_os_string().into_vec();
            path.reserve_exact(first.as_bytes().len() + 1);
            path.push(b'/');
            path.extend_from_slice(first.as_bytes());
            CString::new(path).unwrap()
        })
        .collect()
    };

    static RUNNING: AtomicBool = AtomicBool::new(false);
    assert!(
        !RUNNING.swap(true, Ordering::Acquire),
        "filterm is already running",
    );

    let result = run_impl(
        if paths.is_empty() {
            &args[..1]
        } else {
            &paths
        },
        &args,
        filter,
    );

    if let Some(pid) = CHILD_PID.with(Cell::take) {
        if result.is_err() {
            ignore_error!(kill(pid, Signal::SIGHUP));
        }
    }

    if let Some(attrs) = ORIG_TERM_ATTRS.with(Cell::take) {
        ignore_error!(tcsetattr(io::stdin(), SetArg::TCSANOW, &attrs));
    }

    ORIG_SIGNAL_ACTIONS.with(|actions| {
        for (i, action) in actions.iter().enumerate() {
            if let Some(ref action) = action.take() {
                unsafe {
                    sigaction(SIGNALS[i], action).expect("sigaction() failed");
                }
            }
        }
    });

    for (i, received) in SIGNAL_RECEIVED.iter().enumerate() {
        if received.swap(false, Ordering::Relaxed) {
            ignore_error!(raise(SIGNALS[i]));
        }
    }

    RUNNING.store(false, Ordering::Release);
    result
}
